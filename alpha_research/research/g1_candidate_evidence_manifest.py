"""Single-reducer assembly of research-only G1 candidate evidence.

Workers may independently materialize candidate evidence, snapshot-remediation
receipts, and provider-regime receipts.  They cannot publish a manifest.  This
module is the narrow reducer boundary that content-binds those artifacts,
recomputes provider-regime receipts from the supplied original observations,
and invokes the existing screening engine exactly once over the closed
denominator.  QC receipt bodies, scalar receipt bodies, and the remediation
output frame remain upstream verifier boundaries: their hashes are bound here,
but their contents are not independently recomputed by this module.

It performs no raw-data I/O, hidden-OOS access, admission, submission, or
production action.  The cutover boundary segment is retained as diagnostic
evidence and is never counted in the pre/post stability threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import json
import math
from typing import Final, cast

import pandas as pd

from alpha_research.core.hashing import (
    canonical_json_bytes,
    canonicalize,
    hash_json,
    require_sha256,
)
from alpha_research.high_frequency.snapshot_remediation import (
    SnapshotRemediationMode,
    SnapshotRemediationReceipt,
    SnapshotRemediationSpec,
)
from alpha_research.research.research_screening_verdict import (
    ResearchScreeningCandidateEvidenceV1,
    ResearchScreeningReceiptV1,
    ResearchScreeningSpecV1,
    ResearchScreeningVerdictEngineV1,
)
from alpha_research.robustness.provider_regime_evaluation import (
    ProviderRegimeEvaluationReceipt,
    ProviderRegimeEvaluationSpec,
    verify_provider_regime_evaluation_receipt,
)


LINEAGE_SCHEMA: Final = "g1-candidate-lineage/v1"
MANIFEST_SPEC_SCHEMA: Final = "g1-candidate-evidence-manifest-spec/v1"
WORKER_ARTIFACT_SCHEMA: Final = "g1-candidate-evidence-worker-artifact/v1"
MANIFEST_ENTRY_SCHEMA: Final = "g1-candidate-evidence-manifest-entry/v1"
MANIFEST_RECEIPT_SCHEMA: Final = "g1-candidate-evidence-manifest-receipt/v1"
SOURCE_EVIDENCE_MANIFEST_SCHEMA: Final = "g1-candidate-source-evidence-manifest/v1"
PROVIDER_EVALUATION_ID_SCHEMA: Final = "g1-provider-evaluation-identity/v1"
STABILITY_SCOPES: Final = ("pre_cutover", "post_cutover")
BOUNDARY_SCOPE: Final = "cutover_boundary"


class G1CandidateEvidenceManifestError(ValueError):
    """Stable, fail-closed error raised at the G1 reducer boundary."""

    def __init__(self, code: str, detail: str) -> None:
        if (
            not isinstance(code, str)
            or not code
            or not code[0].isalpha()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in code)
        ):
            raise ValueError("G1 manifest error code is unsafe")
        self.code = code
        self.detail = str(detail)
        super().__init__(f"{code}:{detail}")


def _error(code: str, detail: str) -> G1CandidateEvidenceManifestError:
    return G1CandidateEvidenceManifestError(code, detail)


def _text(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
            for character in value
        )
    ):
        raise _error("invalid_text", f"{name} must be canonical text")
    return value


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise _error("invalid_digest", f"{name} must be a lowercase sha256 digest")
    try:
        return cast(str, require_sha256(value, name=name))
    except ValueError as exc:
        raise _error("invalid_digest", str(exc)) from exc


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error("invalid_integer", f"{name} must be an integer >= {minimum}")
    return value


def _date(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise _error("invalid_date", f"{name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _error("invalid_date", f"{name} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise _error("invalid_date", f"{name} must be canonical")
    return value


def _boundary(
    *,
    research_only: object,
    admission_claim: object,
    production_ready: object,
    submission_authorized: object,
    hidden_oos_consumed: object,
    release_authorized: object,
) -> None:
    if research_only is not True:
        raise _error("boundary_violation", "G1 evidence must remain research-only")
    if admission_claim is not False:
        raise _error("admission_forbidden", "G1 evidence cannot claim admission")
    if production_ready is not False:
        raise _error("production_forbidden", "G1 evidence cannot claim production readiness")
    if submission_authorized is not False:
        raise _error("submission_forbidden", "G1 evidence cannot authorize submission")
    if hidden_oos_consumed is not False:
        raise _error("hidden_oos_forbidden", "G1 evidence cannot consume hidden OOS")
    if release_authorized is not False:
        raise _error("release_forbidden", "G1 evidence cannot authorize release")


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    if not isinstance(value, Mapping):
        raise _error("invalid_mapping", f"{name} must be a mapping")
    observed = frozenset(value)
    if observed != expected:
        raise _error(
            "field_set_mismatch",
            f"{name} fields differ; missing={sorted(expected - observed)!r}; "
            f"extra={sorted(observed - expected)!r}",
        )


def _canonical_mapping(payload: bytes, *, name: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error("invalid_wire_payload", f"{name} is not canonical JSON") from exc
    if not isinstance(decoded, dict):
        raise _error("invalid_wire_payload", f"{name} must decode to an object")
    if canonical_json_bytes(decoded) != payload:
        raise _error("noncanonical_wire_payload", f"{name} is not canonical JSON")
    return cast(Mapping[str, object], decoded)


def _record_dict(value: object) -> dict[str, object]:
    """Use the repository's canonical dataclass projection in one place."""

    projected = canonicalize(value)
    if not isinstance(projected, dict):  # pragma: no cover - dataclass invariant
        raise TypeError("canonical record projection must be an object")
    return cast(dict[str, object], projected)


def expected_provider_evaluation_id(campaign_id: str, candidate_id: str) -> str:
    """Return the only accepted candidate/provider evaluation identity."""

    identity_hash = hash_json(
        {
            "schema_version": PROVIDER_EVALUATION_ID_SCHEMA,
            "campaign_id": _text(campaign_id, name="campaign_id"),
            "candidate_id": _text(candidate_id, name="candidate_id"),
        }
    )
    return f"provider-regime.{identity_hash}"


@dataclass(frozen=True, slots=True)
class G1CandidateLineageV1:
    """Preregistered identity and evidence hashes for one candidate."""

    candidate_id: str
    family_id: str
    factor_spec_hash: str
    label_spec_hash: str
    split_spec_hash: str
    cost_spec_hash: str
    qc_quarantine_receipt_hash: str
    qc_quarantined_frame_hash: str
    remediation_spec_hash: str
    provider_evaluation_spec_hash: str
    scalar_source_evidence_hash: str
    schema_version: str = LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != LINEAGE_SCHEMA:
            raise _error("unsupported_schema", "unsupported G1 candidate lineage")
        for name in ("candidate_id", "family_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "factor_spec_hash",
            "label_spec_hash",
            "split_spec_hash",
            "cost_spec_hash",
            "qc_quarantine_receipt_hash",
            "qc_quarantined_frame_hash",
            "remediation_spec_hash",
            "provider_evaluation_spec_hash",
            "scalar_source_evidence_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "G1CandidateLineageV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        return cls(**cast(dict[str, object], dict(value)))  # type: ignore[arg-type]


def expected_source_evidence_manifest_hash(
    campaign_id: str,
    lineage: G1CandidateLineageV1,
    remediation_receipt: SnapshotRemediationReceipt,
    provider_evaluation_receipt: ProviderRegimeEvaluationReceipt,
) -> str:
    """Bind screening scalars to their complete, content-addressed lineage.

    The upstream scalar receipt remains an opaque hash in ``lineage``.  This
    manifest additionally binds the exact remediation receipt/output and the
    provider receipt/original-observation hash verified by this reducer.
    """

    if type(lineage) is not G1CandidateLineageV1:
        raise TypeError("lineage must be an exact G1 V1 record")
    if type(remediation_receipt) is not SnapshotRemediationReceipt:
        raise TypeError("remediation_receipt must be an exact receipt")
    if type(provider_evaluation_receipt) is not ProviderRegimeEvaluationReceipt:
        raise TypeError("provider_evaluation_receipt must be an exact receipt")
    return cast(
        str,
        hash_json(
            {
                "schema_version": SOURCE_EVIDENCE_MANIFEST_SCHEMA,
                "campaign_id": _text(campaign_id, name="campaign_id"),
                "candidate_lineage": lineage.to_dict(),
                "candidate_lineage_hash": lineage.content_hash,
                "remediation_receipt_hash": remediation_receipt.receipt_id,
                "remediated_frame_hash": remediation_receipt.output_frame_hash,
                "provider_evaluation_receipt_hash": (
                    provider_evaluation_receipt.content_hash
                ),
                "provider_input_hash": provider_evaluation_receipt.input_hash,
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class G1CandidateEvidenceManifestSpecV1:
    """Closed denominator and reducer policy for one G1 campaign."""

    campaign_id: str
    screening_spec_hash: str
    candidates: tuple[G1CandidateLineageV1, ...]
    minimum_stability_rank_ic_dates: int
    provider_cutover_date: str = "2021-06-07"
    required_remediation_mode: SnapshotRemediationMode | str = (
        SnapshotRemediationMode.MASK_TO_NULL
    )
    provider_stability_scope_ids: tuple[str, ...] = STABILITY_SCOPES
    provider_boundary_scope_id: str = BOUNDARY_SCOPE
    boundary_contributes_to_stability_thresholds: bool = False
    publisher_role: str = "single_reducer"
    worker_publish_authorized: bool = False
    research_only: bool = True
    admission_claim: bool = False
    production_ready: bool = False
    submission_authorized: bool = False
    hidden_oos_consumed: bool = False
    release_authorized: bool = False
    schema_version: str = MANIFEST_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SPEC_SCHEMA:
            raise _error("unsupported_schema", "unsupported G1 manifest spec")
        object.__setattr__(self, "campaign_id", _text(self.campaign_id, name="campaign_id"))
        object.__setattr__(
            self,
            "screening_spec_hash",
            _digest(self.screening_spec_hash, name="screening_spec_hash"),
        )
        candidates = tuple(self.candidates)
        if not candidates or any(type(item) is not G1CandidateLineageV1 for item in candidates):
            raise _error("invalid_denominator", "candidates must be exact lineage records")
        if tuple(item.candidate_id for item in candidates) != tuple(
            sorted(item.candidate_id for item in candidates)
        ):
            raise _error("noncanonical_denominator", "candidate lineage must be sorted")
        if len({item.candidate_id for item in candidates}) != len(candidates):
            raise _error("duplicate_candidate", "candidate lineage is duplicated")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "minimum_stability_rank_ic_dates",
            _integer(
                self.minimum_stability_rank_ic_dates,
                name="minimum_stability_rank_ic_dates",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "provider_cutover_date",
            _date(self.provider_cutover_date, name="provider_cutover_date"),
        )
        object.__setattr__(
            self,
            "required_remediation_mode",
            SnapshotRemediationMode(self.required_remediation_mode),
        )
        if self.required_remediation_mode is SnapshotRemediationMode.ORDERFLOW_RECONSTRUCT:
            raise _error(
                "orderflow_reconstruction_forbidden",
                "G1 assembly cannot authorize event replay or reconstruction",
            )
        if self.provider_stability_scope_ids != STABILITY_SCOPES:
            raise _error("invalid_provider_policy", "stability scopes must be pre/post only")
        if self.provider_boundary_scope_id != BOUNDARY_SCOPE:
            raise _error("invalid_provider_policy", "cutover boundary scope differs")
        if self.boundary_contributes_to_stability_thresholds is not False:
            raise _error("boundary_threshold_forbidden", "boundary is diagnostic only")
        if self.publisher_role != "single_reducer" or self.worker_publish_authorized is not False:
            raise _error("invalid_publisher_policy", "only the single reducer may publish")
        _boundary(
            research_only=self.research_only,
            admission_claim=self.admission_claim,
            production_ready=self.production_ready,
            submission_authorized=self.submission_authorized,
            hidden_oos_consumed=self.hidden_oos_consumed,
            release_authorized=self.release_authorized,
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "G1CandidateEvidenceManifestSpecV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        raw_candidates = value["candidates"]
        raw_scopes = value["provider_stability_scope_ids"]
        if not isinstance(raw_candidates, list) or any(
            not isinstance(item, Mapping) for item in raw_candidates
        ):
            raise _error("invalid_mapping", "candidate lineage must be a list of objects")
        if not isinstance(raw_scopes, list):
            raise _error("invalid_sequence", "provider stability scopes must be a list")
        payload = dict(value)
        payload["candidates"] = tuple(
            G1CandidateLineageV1.from_mapping(cast(Mapping[str, object], item))
            for item in raw_candidates
        )
        payload["provider_stability_scope_ids"] = tuple(
            cast(Sequence[str], raw_scopes)
        )
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class G1CandidateEvidenceWorkerArtifactV1:
    """Independent worker output; it deliberately carries no publish authority."""

    campaign_id: str
    worker_id: str
    candidate_id: str
    family_id: str
    candidate_lineage_hash: str
    screening_evidence: ResearchScreeningCandidateEvidenceV1
    remediation_spec: SnapshotRemediationSpec
    remediation_receipt: SnapshotRemediationReceipt
    provider_evaluation_spec: ProviderRegimeEvaluationSpec
    provider_evaluation_receipt: ProviderRegimeEvaluationReceipt
    publish_authorized: bool = False
    research_only: bool = True
    admission_claim: bool = False
    production_ready: bool = False
    submission_authorized: bool = False
    hidden_oos_consumed: bool = False
    release_authorized: bool = False
    schema_version: str = WORKER_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_ARTIFACT_SCHEMA:
            raise _error("unsupported_schema", "unsupported G1 worker artifact")
        for name in ("campaign_id", "worker_id", "candidate_id", "family_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "candidate_lineage_hash",
            _digest(self.candidate_lineage_hash, name="candidate_lineage_hash"),
        )
        if type(self.screening_evidence) is not ResearchScreeningCandidateEvidenceV1:
            raise TypeError("screening_evidence must be an exact V1 record")
        if type(self.remediation_spec) is not SnapshotRemediationSpec:
            raise TypeError("remediation_spec must be an exact specification")
        if type(self.remediation_receipt) is not SnapshotRemediationReceipt:
            raise TypeError("remediation_receipt must be an exact receipt")
        if type(self.provider_evaluation_spec) is not ProviderRegimeEvaluationSpec:
            raise TypeError("provider_evaluation_spec must be an exact specification")
        if type(self.provider_evaluation_receipt) is not ProviderRegimeEvaluationReceipt:
            raise TypeError("provider_evaluation_receipt must be an exact receipt")
        if (
            self.screening_evidence.candidate_id != self.candidate_id
            or self.screening_evidence.family_id != self.family_id
        ):
            raise _error("candidate_identity_mismatch", "screening candidate identity differs")
        if self.remediation_receipt.spec_hash != self.remediation_spec.content_hash:
            raise _error("remediation_spec_mismatch", "remediation receipt spec differs")
        if (
            self.provider_evaluation_receipt.evaluation_spec_hash
            != self.provider_evaluation_spec.content_hash
        ):
            raise _error("provider_spec_mismatch", "provider receipt spec differs")
        if self.publish_authorized is not False:
            raise _error("worker_publish_forbidden", "worker artifacts cannot publish")
        _boundary(
            research_only=self.research_only,
            admission_claim=self.admission_claim,
            production_ready=self.production_ready,
            submission_authorized=self.submission_authorized,
            hidden_oos_consumed=self.hidden_oos_consumed,
            release_authorized=self.release_authorized,
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        result = _record_dict(self)
        # SnapshotRemediationSpec deliberately serializes two policy constants
        # not stored as dataclass fields, so preserve its public wire contract.
        result["remediation_spec"] = self.remediation_spec.to_dict()
        return result

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "G1CandidateEvidenceWorkerArtifactV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        nested_fields = (
            "screening_evidence",
            "remediation_spec",
            "remediation_receipt",
            "provider_evaluation_spec",
            "provider_evaluation_receipt",
        )
        for name in nested_fields:
            if not isinstance(value[name], Mapping):
                raise _error(
                    "invalid_mapping",
                    f"G1 worker artifact {name} must be an object",
                )
        payload = dict(value)
        try:
            payload["screening_evidence"] = (
                ResearchScreeningCandidateEvidenceV1.from_mapping(
                    cast(Mapping[str, object], value["screening_evidence"])
                )
            )
            payload["remediation_spec"] = SnapshotRemediationSpec.from_mapping(
                cast(Mapping[str, object], value["remediation_spec"])
            )
            payload["remediation_receipt"] = SnapshotRemediationReceipt.from_mapping(
                cast(Mapping[str, object], value["remediation_receipt"])
            )
            payload["provider_evaluation_spec"] = (
                ProviderRegimeEvaluationSpec.from_mapping(
                    cast(Mapping[str, object], value["provider_evaluation_spec"])
                )
            )
            payload["provider_evaluation_receipt"] = (
                ProviderRegimeEvaluationReceipt.from_mapping(
                    cast(
                        Mapping[str, object],
                        value["provider_evaluation_receipt"],
                    )
                )
            )
            result = cls(**payload)  # type: ignore[arg-type]
        except G1CandidateEvidenceManifestError:
            raise
        except (TypeError, ValueError) as exc:
            raise _error("worker_artifact_invalid", str(exc)) from exc
        if result.to_dict() != dict(value):
            raise _error(
                "noncanonical_worker_artifact",
                "G1 worker artifact mapping is not canonical",
            )
        return result

    @classmethod
    def from_wire_bytes(
        cls,
        payload: bytes,
        *,
        expected_hash: str,
    ) -> "G1CandidateEvidenceWorkerArtifactV1":
        result = cls.from_mapping(_canonical_mapping(payload, name="G1 worker artifact"))
        if result.to_wire_bytes() != payload:
            raise _error(
                "noncanonical_wire_payload",
                "G1 worker artifact is not canonical JSON",
            )
        if result.content_hash != _digest(
            expected_hash, name="expected_worker_artifact_hash"
        ):
            raise _error(
                "worker_artifact_hash_mismatch",
                "G1 worker artifact identity differs",
            )
        return result


@dataclass(frozen=True, slots=True)
class G1CandidateEvidenceManifestEntryV1:
    """Transparent lineage entry emitted by the single reducer."""

    lineage: G1CandidateLineageV1
    worker_id: str
    worker_artifact_hash: str
    remediation_receipt_hash: str
    provider_evaluation_receipt_hash: str
    provider_input_hash: str
    screening_evidence_hash: str
    schema_version: str = MANIFEST_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_ENTRY_SCHEMA:
            raise _error("unsupported_schema", "unsupported G1 manifest entry")
        if type(self.lineage) is not G1CandidateLineageV1:
            raise _error("invalid_lineage", "manifest entry lineage must be exact V1")
        object.__setattr__(self, "worker_id", _text(self.worker_id, name="worker_id"))
        for name in (
            "worker_artifact_hash",
            "remediation_receipt_hash",
            "provider_evaluation_receipt_hash",
            "provider_input_hash",
            "screening_evidence_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))

    @property
    def candidate_id(self) -> str:
        return self.lineage.candidate_id

    @property
    def family_id(self) -> str:
        return self.lineage.family_id

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "G1CandidateEvidenceManifestEntryV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        raw_lineage = value["lineage"]
        if not isinstance(raw_lineage, Mapping):
            raise _error("invalid_mapping", "manifest entry lineage must be an object")
        payload = dict(value)
        payload["lineage"] = G1CandidateLineageV1.from_mapping(raw_lineage)
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class G1CandidateEvidenceManifestReceiptV1:
    """Content-addressed reducer receipt over the exact candidate denominator."""

    campaign_id: str
    manifest_spec_hash: str
    screening_spec_hash: str
    denominator_candidate_ids: tuple[str, ...]
    entries: tuple[G1CandidateEvidenceManifestEntryV1, ...]
    tested_count: int
    screening_evidence_set_hash: str
    screening_receipt_hash: str
    provider_stability_scope_ids: tuple[str, ...] = STABILITY_SCOPES
    provider_boundary_scope_id: str = BOUNDARY_SCOPE
    boundary_contributes_to_stability_thresholds: bool = False
    publisher_role: str = "single_reducer"
    worker_publish_authorized: bool = False
    research_only: bool = True
    admission_claim: bool = False
    production_ready: bool = False
    submission_authorized: bool = False
    hidden_oos_consumed: bool = False
    release_authorized: bool = False
    schema_version: str = MANIFEST_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_RECEIPT_SCHEMA:
            raise _error("unsupported_schema", "unsupported G1 manifest receipt")
        object.__setattr__(self, "campaign_id", _text(self.campaign_id, name="campaign_id"))
        for name in (
            "manifest_spec_hash",
            "screening_spec_hash",
            "screening_evidence_set_hash",
            "screening_receipt_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        denominator = tuple(
            _text(item, name="denominator_candidate_id")
            for item in self.denominator_candidate_ids
        )
        entries = tuple(self.entries)
        if not denominator or denominator != tuple(sorted(denominator)):
            raise _error("invalid_denominator", "receipt denominator is not canonical")
        if len(set(denominator)) != len(denominator):
            raise _error("duplicate_candidate", "receipt denominator is duplicated")
        if any(type(item) is not G1CandidateEvidenceManifestEntryV1 for item in entries):
            raise _error("invalid_entry", "receipt entries must use exact V1 records")
        if tuple(item.candidate_id for item in entries) != denominator:
            raise _error("candidate_set_drift", "receipt entries differ from denominator")
        if len({item.worker_artifact_hash for item in entries}) != len(entries):
            raise _error("duplicate_worker_artifact", "worker artifact is reused")
        object.__setattr__(self, "denominator_candidate_ids", denominator)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "tested_count", _integer(self.tested_count, name="tested_count"))
        if self.tested_count != len(denominator):
            raise _error("count_mismatch", "tested count differs from denominator")
        if self.provider_stability_scope_ids != STABILITY_SCOPES:
            raise _error("invalid_provider_policy", "receipt stability scopes differ")
        if self.provider_boundary_scope_id != BOUNDARY_SCOPE:
            raise _error("invalid_provider_policy", "receipt boundary scope differs")
        if self.boundary_contributes_to_stability_thresholds is not False:
            raise _error("boundary_threshold_forbidden", "boundary is diagnostic only")
        if self.publisher_role != "single_reducer" or self.worker_publish_authorized is not False:
            raise _error("invalid_publisher_policy", "receipt publisher policy differs")
        _boundary(
            research_only=self.research_only,
            admission_claim=self.admission_claim,
            production_ready=self.production_ready,
            submission_authorized=self.submission_authorized,
            hidden_oos_consumed=self.hidden_oos_consumed,
            release_authorized=self.release_authorized,
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "G1CandidateEvidenceManifestReceiptV1":
        _exact_fields(value, frozenset(cls.__dataclass_fields__), name=cls.__name__)
        raw_denominator = value["denominator_candidate_ids"]
        raw_entries = value["entries"]
        raw_scopes = value["provider_stability_scope_ids"]
        if not isinstance(raw_denominator, list) or not isinstance(raw_scopes, list):
            raise _error("invalid_sequence", "receipt identifiers must be JSON lists")
        if not isinstance(raw_entries, list) or any(
            not isinstance(item, Mapping) for item in raw_entries
        ):
            raise _error("invalid_mapping", "receipt entries must be a list of objects")
        payload = dict(value)
        payload["denominator_candidate_ids"] = tuple(
            cast(Sequence[str], raw_denominator)
        )
        payload["entries"] = tuple(
            G1CandidateEvidenceManifestEntryV1.from_mapping(
                cast(Mapping[str, object], item)
            )
            for item in raw_entries
        )
        payload["provider_stability_scope_ids"] = tuple(
            cast(Sequence[str], raw_scopes)
        )
        return cls(**payload)  # type: ignore[arg-type]

    @classmethod
    def from_wire_bytes(
        cls,
        payload: bytes,
        *,
        expected_hash: str | None = None,
    ) -> "G1CandidateEvidenceManifestReceiptV1":
        result = cls.from_mapping(_canonical_mapping(payload, name="G1 manifest receipt"))
        if expected_hash is not None and result.content_hash != _digest(
            expected_hash, name="expected_manifest_receipt_hash"
        ):
            raise _error("receipt_hash_mismatch", "G1 manifest receipt hash differs")
        return result


@dataclass(frozen=True, slots=True)
class G1CandidateEvidenceAssemblyV1:
    """In-memory research result; no method in this type publishes externally."""

    manifest_receipt: G1CandidateEvidenceManifestReceiptV1
    screening_receipt: ResearchScreeningReceiptV1
    screening_evidence: tuple[ResearchScreeningCandidateEvidenceV1, ...]

    def __post_init__(self) -> None:
        evidence = tuple(self.screening_evidence)
        if tuple(item.candidate_id for item in evidence) != (
            self.manifest_receipt.denominator_candidate_ids
        ):
            raise _error("candidate_set_drift", "assembly evidence differs from manifest")
        if self.screening_receipt.content_hash != self.manifest_receipt.screening_receipt_hash:
            raise _error("screening_receipt_mismatch", "screening receipt hash differs")
        if (
            self.screening_receipt.evidence_set_hash
            != self.manifest_receipt.screening_evidence_set_hash
        ):
            raise _error("screening_evidence_mismatch", "screening evidence set differs")
        verdict_hashes = {
            item.candidate_id: item.evidence_hash
            for item in self.screening_receipt.candidate_verdicts
        }
        if verdict_hashes != {item.candidate_id: item.content_hash for item in evidence}:
            raise _error("screening_evidence_mismatch", "screening verdict evidence differs")
        object.__setattr__(self, "screening_evidence", evidence)


class G1CandidateEvidenceReducerV1:
    """Verify independent worker artifacts and emit one canonical G1 receipt."""

    __slots__ = ("_manifest_spec", "_screening_spec")

    def __init__(
        self,
        manifest_spec: G1CandidateEvidenceManifestSpecV1,
        screening_spec: ResearchScreeningSpecV1,
    ) -> None:
        if type(manifest_spec) is not G1CandidateEvidenceManifestSpecV1:
            raise TypeError("manifest_spec must be an exact V1 specification")
        if type(screening_spec) is not ResearchScreeningSpecV1:
            raise TypeError("screening_spec must be an exact V1 specification")
        manifest_snapshot = G1CandidateEvidenceManifestSpecV1.from_mapping(
            manifest_spec.to_dict()
        )
        screening_snapshot = ResearchScreeningSpecV1.from_wire_bytes(
            screening_spec.to_wire_bytes()
        )
        self._validate_specs(manifest_snapshot, screening_snapshot)
        self._manifest_spec = manifest_snapshot
        self._screening_spec = screening_snapshot

    @staticmethod
    def _validate_specs(
        manifest: G1CandidateEvidenceManifestSpecV1,
        screening: ResearchScreeningSpecV1,
    ) -> None:
        if manifest.screening_spec_hash != screening.content_hash:
            raise _error("screening_spec_mismatch", "manifest screening spec hash differs")
        if manifest.campaign_id != screening.campaign_id:
            raise _error("campaign_mismatch", "manifest and screening campaigns differ")
        expected_ids = screening.candidate_ids
        observed_ids = tuple(item.candidate_id for item in manifest.candidates)
        if observed_ids != tuple(sorted(expected_ids)):
            raise _error("candidate_set_drift", "manifest and screening denominators differ")
        family_by_candidate = screening.family_by_candidate
        if any(
            item.family_id != family_by_candidate[item.candidate_id]
            for item in manifest.candidates
        ):
            raise _error("family_binding_mismatch", "manifest family binding differs")
        if screening.min_provider_regime_count != len(STABILITY_SCOPES):
            raise _error(
                "provider_policy_mismatch",
                "screening must require both pre/post provider regimes",
            )

    @property
    def manifest_spec_hash(self) -> str:
        return self._manifest_spec.content_hash

    def reduce(
        self,
        artifacts: Sequence[G1CandidateEvidenceWorkerArtifactV1],
        *,
        provider_observations: Mapping[str, pd.DataFrame],
    ) -> G1CandidateEvidenceAssemblyV1:
        supplied_artifacts = list(artifacts)
        if any(type(item) is not G1CandidateEvidenceWorkerArtifactV1 for item in supplied_artifacts):
            raise TypeError("artifacts must use exact G1 worker records")
        artifact_snapshots: list[G1CandidateEvidenceWorkerArtifactV1] = []
        for item in supplied_artifacts:
            if (
                type(item.screening_evidence)
                is not ResearchScreeningCandidateEvidenceV1
                or type(item.remediation_spec) is not SnapshotRemediationSpec
                or type(item.remediation_receipt) is not SnapshotRemediationReceipt
                or type(item.provider_evaluation_spec)
                is not ProviderRegimeEvaluationSpec
                or type(item.provider_evaluation_receipt)
                is not ProviderRegimeEvaluationReceipt
            ):
                raise TypeError("worker artifact nested records must use exact V1 types")
            artifact_snapshots.append(
                G1CandidateEvidenceWorkerArtifactV1.from_wire_bytes(
                    item.to_wire_bytes(), expected_hash=item.content_hash
                )
            )
        observed_ids = [item.candidate_id for item in artifact_snapshots]
        if len(set(observed_ids)) != len(observed_ids):
            raise _error("duplicate_candidate_artifact", "candidate artifact is duplicated")
        expected_ids = tuple(item.candidate_id for item in self._manifest_spec.candidates)
        if set(observed_ids) != set(expected_ids):
            raise _error("candidate_set_drift", "worker artifacts differ from denominator")
        if set(provider_observations) != set(expected_ids):
            raise _error("candidate_set_drift", "provider observations differ from denominator")
        if len({item.content_hash for item in artifact_snapshots}) != len(artifact_snapshots):
            raise _error("duplicate_worker_artifact", "worker artifact is duplicated")

        by_candidate = {item.candidate_id: item for item in artifact_snapshots}
        evidence: list[ResearchScreeningCandidateEvidenceV1] = []
        entries: list[G1CandidateEvidenceManifestEntryV1] = []
        for lineage in self._manifest_spec.candidates:
            artifact = by_candidate[lineage.candidate_id]
            candidate_evidence, entry = self._verify_candidate(
                lineage,
                artifact,
                pd.DataFrame(provider_observations[lineage.candidate_id]),
            )
            evidence.append(candidate_evidence)
            entries.append(entry)

        screening_receipt = ResearchScreeningVerdictEngineV1(
            self._screening_spec
        ).screen(tuple(evidence))
        receipt = G1CandidateEvidenceManifestReceiptV1(
            campaign_id=self._manifest_spec.campaign_id,
            manifest_spec_hash=self._manifest_spec.content_hash,
            screening_spec_hash=self._screening_spec.content_hash,
            denominator_candidate_ids=expected_ids,
            entries=tuple(entries),
            tested_count=len(entries),
            screening_evidence_set_hash=screening_receipt.evidence_set_hash,
            screening_receipt_hash=screening_receipt.content_hash,
        )
        return G1CandidateEvidenceAssemblyV1(
            manifest_receipt=receipt,
            screening_receipt=screening_receipt,
            screening_evidence=tuple(evidence),
        )

    def _verify_candidate(
        self,
        lineage: G1CandidateLineageV1,
        artifact: G1CandidateEvidenceWorkerArtifactV1,
        observations: pd.DataFrame,
    ) -> tuple[
        ResearchScreeningCandidateEvidenceV1,
        G1CandidateEvidenceManifestEntryV1,
    ]:
        if artifact.campaign_id != self._manifest_spec.campaign_id:
            raise _error("campaign_mismatch", "worker campaign differs")
        if artifact.candidate_id != lineage.candidate_id or artifact.family_id != lineage.family_id:
            raise _error("candidate_identity_mismatch", "worker candidate identity differs")
        if artifact.candidate_lineage_hash != lineage.content_hash:
            raise _error("lineage_hash_mismatch", "worker lineage hash differs")

        # Snapshot nested records again at reducer time.  This catches objects
        # mutated after worker construction before any receipt can be emitted.
        evidence = ResearchScreeningCandidateEvidenceV1.from_mapping(
            artifact.screening_evidence.to_dict()
        )
        remediation_spec = SnapshotRemediationSpec.from_mapping(
            artifact.remediation_spec.to_dict()
        )
        try:
            remediation_receipt = SnapshotRemediationReceipt.from_mapping(
                artifact.remediation_receipt.to_dict()
            )
        except (TypeError, ValueError) as exc:
            raise _error("remediation_receipt_invalid", str(exc)) from exc
        provider_spec = ProviderRegimeEvaluationSpec.from_wire_bytes(
            artifact.provider_evaluation_spec.to_wire_bytes()
        )
        provider_receipt = ProviderRegimeEvaluationReceipt.from_wire_bytes(
            artifact.provider_evaluation_receipt.to_wire_bytes()
        )

        if remediation_spec.content_hash != lineage.remediation_spec_hash:
            raise _error("remediation_spec_mismatch", "remediation policy differs")
        if remediation_spec.mode is not self._manifest_spec.required_remediation_mode:
            raise _error("remediation_mode_mismatch", "remediation mode differs from policy")
        if remediation_receipt.spec_hash != remediation_spec.content_hash:
            raise _error("remediation_spec_mismatch", "remediation receipt spec differs")
        remediation_mode = cast(SnapshotRemediationMode, remediation_spec.mode)
        if remediation_receipt.mode != remediation_mode.value:
            raise _error("remediation_mode_mismatch", "remediation receipt mode differs")
        if remediation_receipt.input_frame_hash != lineage.qc_quarantined_frame_hash:
            raise _error("qc_frame_hash_mismatch", "remediation input differs from QC frame")
        if provider_spec.content_hash != lineage.provider_evaluation_spec_hash:
            raise _error("provider_spec_mismatch", "provider evaluation policy differs")
        if provider_spec.cutover_date != self._manifest_spec.provider_cutover_date:
            raise _error("provider_cutover_mismatch", "provider cutover date differs")
        expected_evaluation_id = expected_provider_evaluation_id(
            self._manifest_spec.campaign_id, lineage.candidate_id
        )
        if provider_spec.evaluation_id != expected_evaluation_id:
            raise _error("provider_identity_mismatch", "provider candidate identity differs")
        try:
            verified_provider = verify_provider_regime_evaluation_receipt(
                provider_spec,
                observations,
                provider_receipt,
            )
        except (TypeError, ValueError) as exc:
            raise _error("provider_receipt_invalid", str(exc)) from exc
        source_manifest_hash = expected_source_evidence_manifest_hash(
            self._manifest_spec.campaign_id,
            lineage,
            remediation_receipt,
            verified_provider,
        )
        if evidence.source_evidence_hash != source_manifest_hash:
            raise _error(
                "source_evidence_manifest_mismatch",
                "screening evidence is not bound to the verified source manifest",
            )
        regime_count, regime_pass_count, worst_regime_ic = self._provider_scalars(
            verified_provider
        )
        if (
            evidence.provider_regime_count != regime_count
            or evidence.provider_regime_pass_count != regime_pass_count
            or evidence.worst_provider_regime_ic != worst_regime_ic
        ):
            raise _error(
                "provider_scalar_mismatch",
                "screening provider scalars differ from verified provider receipt",
            )

        entry = G1CandidateEvidenceManifestEntryV1(
            lineage=lineage,
            worker_id=artifact.worker_id,
            worker_artifact_hash=artifact.content_hash,
            remediation_receipt_hash=remediation_receipt.receipt_id,
            provider_evaluation_receipt_hash=verified_provider.content_hash,
            provider_input_hash=verified_provider.input_hash,
            screening_evidence_hash=evidence.content_hash,
        )
        return evidence, entry

    def _provider_scalars(
        self, receipt: ProviderRegimeEvaluationReceipt
    ) -> tuple[int, int, float]:
        by_scope = {item.scope_id: item for item in receipt.segment_metrics}
        if set(by_scope) != {*STABILITY_SCOPES, BOUNDARY_SCOPE}:
            raise _error("provider_scope_mismatch", "provider receipt scopes differ")
        stability = [by_scope[scope] for scope in STABILITY_SCOPES]
        normalized_means: list[float] = []
        for metrics in stability:
            value = metrics.rank_ic_mean
            if value is None or not math.isfinite(value):
                raise _error(
                    "provider_scalar_unavailable",
                    "pre/post RankIC mean is unavailable",
                )
            normalized_means.append(0.0 if value == 0.0 else value)
        passed = sum(
            1
            for metrics, mean in zip(stability, normalized_means, strict=True)
            if metrics.valid_rank_ic_dates
            >= self._manifest_spec.minimum_stability_rank_ic_dates
            and mean >= self._screening_spec.min_worst_provider_regime_ic
        )
        return len(STABILITY_SCOPES), passed, min(normalized_means)

    def verify_and_replay(
        self,
        *,
        manifest_receipt_payload: bytes,
        expected_manifest_receipt_hash: str,
        screening_receipt_payload: bytes,
        expected_screening_receipt_hash: str,
        artifacts: Sequence[G1CandidateEvidenceWorkerArtifactV1],
        provider_observations: Mapping[str, pd.DataFrame],
    ) -> G1CandidateEvidenceAssemblyV1:
        observed_manifest = G1CandidateEvidenceManifestReceiptV1.from_wire_bytes(
            manifest_receipt_payload,
            expected_hash=expected_manifest_receipt_hash,
        )
        observed_screening = ResearchScreeningReceiptV1.from_wire_bytes(
            screening_receipt_payload,
            expected_hash=expected_screening_receipt_hash,
        )
        replayed = self.reduce(
            artifacts,
            provider_observations=provider_observations,
        )
        if replayed.manifest_receipt.to_wire_bytes() != observed_manifest.to_wire_bytes():
            raise _error("manifest_replay_mismatch", "G1 manifest replay differs")
        if replayed.screening_receipt.to_wire_bytes() != observed_screening.to_wire_bytes():
            raise _error("screening_replay_mismatch", "G1 screening replay differs")
        return replayed


__all__ = [
    "BOUNDARY_SCOPE",
    "G1CandidateEvidenceAssemblyV1",
    "G1CandidateEvidenceManifestEntryV1",
    "G1CandidateEvidenceManifestError",
    "G1CandidateEvidenceManifestReceiptV1",
    "G1CandidateEvidenceManifestSpecV1",
    "G1CandidateEvidenceReducerV1",
    "G1CandidateEvidenceWorkerArtifactV1",
    "G1CandidateLineageV1",
    "STABILITY_SCOPES",
    "expected_provider_evaluation_id",
    "expected_source_evidence_manifest_hash",
]
