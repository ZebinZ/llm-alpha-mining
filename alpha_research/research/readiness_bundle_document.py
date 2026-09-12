"""Content-addressed persistence and replay for research-readiness evidence.

This module is an output contract.  It deliberately does not duplicate or
grant the input authority implemented by :mod:`readiness_inputs`.  A document
binds the scalar identity of a ``ResearchReadinessBundle`` to seven immutable
evidence artifacts; replay reloads every artifact and asks the existing report
and bundle constructors to recompute their scientific decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from types import MappingProxyType
from typing import Final, Mapping, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_frame,
    hash_json,
    require_sha256,
)
from alpha_research.robustness.readiness import (
    DLReadinessVerdict,
    IncrementalBaselineReport,
    NegativeControlReport,
    ParameterStabilityReport,
    ResearchReadinessBundle,
    ResearchReadinessSpec,
)
from alpha_research.research.readiness_inputs import (
    AuthorityBoundResearchReadinessInputs,
    LoadedResearchReadinessInputs,
    ResearchReadinessAuthorityReceiptV1,
    ResearchReadinessFrameReferenceV1,
    ResearchReadinessInputError,
    ResearchReadinessInputManifestV1,
)
from factor_production.v5.artifacts import ArtifactRecord, ArtifactStore
from factor_production.v5.artifacts.manifest import ArtifactError


_DOCUMENT_SCHEMA: Final = "research-readiness-bundle-document/v3"
_EVIDENCE_REFERENCE_SCHEMA: Final = "research-readiness-evidence-reference/v1"
_INPUT_DOCUMENT_REFERENCE_SCHEMA: Final = (
    "research-readiness-input-document-reference/v1"
)
_PARQUET_MEDIA_TYPE: Final = "application/vnd.apache.parquet"
_DOCUMENT_MEDIA_TYPE: Final = (
    "application/vnd.alpha-research.research-readiness-bundle+json"
)
_DOCUMENT_ROLE: Final = "research_readiness_bundle_document"
_DOCUMENT_LOGICAL_NAME: Final = "research-readiness.bundle.document.v1"
_MAXIMUM_DOCUMENT_BYTES: Final = 4 * 1024 * 1024
_DEFAULT_MAXIMUM_EVIDENCE_BYTES: Final = 64 * 1024 * 1024
_DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES: Final = 256 * 1024 * 1024
_DEFAULT_MAXIMUM_TOTAL_ROWS: Final = 10_000_000
_DEFAULT_MAXIMUM_TOTAL_COLUMNS: Final = 1_000_000
_DEFAULT_MAXIMUM_LOADED_MEMORY_BYTES: Final = 512 * 1024 * 1024
_INPUT_DOCUMENTS: Final[Mapping[str, tuple[str, str, str]]] = MappingProxyType(
    {
        "input_manifest": (
            "research-readiness.input-manifest.v1",
            "application/vnd.alpha-research.research-readiness-input-manifest+json",
            "research_readiness_input_manifest",
        ),
        "authority_receipt": (
            "research-readiness.authority-receipt.v1",
            "application/vnd.alpha-research.research-readiness-authority-receipt+json",
            "research_readiness_authority_receipt",
        ),
    }
)
_EVIDENCE_ROLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "negative_observed_daily": "readiness_negative_observed_daily",
        "negative_pseudo_label_trials": "readiness_negative_pseudo_label_trials",
        "negative_lag_summary": "readiness_negative_lag_summary",
        "negative_lag_daily": "readiness_negative_lag_daily",
        "stability_scenario_summary": "readiness_stability_scenario_summary",
        "stability_daily": "readiness_stability_daily",
        "incremental_baseline_daily": "readiness_incremental_baseline_daily",
    }
)


class ResearchReadinessBundleDocumentError(ValueError):
    """Stable fail-closed error raised by readiness output persistence."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


def _error(code: str, detail: str) -> ResearchReadinessBundleDocumentError:
    return ResearchReadinessBundleDocumentError(code, detail)


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    fields = frozenset(value)
    if fields != expected:
        missing = ",".join(sorted(expected - fields)) or "-"
        extra = ",".join(sorted(fields - expected)) or "-"
        raise _error(
            "wire_fields_differ",
            f"{name} fields differ; missing={missing}; extra={extra}",
        )


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _error("invalid_object", f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _error("invalid_text", f"{name} must be non-empty stripped text")
    return value


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise _error("invalid_hash", f"{name} must be text")
    try:
        return cast(str, require_sha256(value, name=name))
    except ValueError as exc:
        raise _error("invalid_hash", str(exc)) from exc


def _integer(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error("invalid_integer", f"{name} must be an integer")
    if value < (1 if positive else 0):
        raise _error("invalid_integer", f"{name} lies outside its domain")
    return value


def _signed_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error("invalid_integer", f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _error("invalid_number", f"{name} must be numeric")
    result = float(value)
    if not pd.notna(result) or result in (float("inf"), float("-inf")):
        raise _error("invalid_number", f"{name} must be finite")
    return result


def _optional_number(value: object, *, name: str) -> float | None:
    return None if value is None else _number(value, name=name)


def _tristate(value: object, *, name: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise _error("invalid_tristate", f"{name} must be boolean or null")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise _error("invalid_boolean", f"{name} must be boolean")
    return value


def _parse_wire(payload: bytes) -> Mapping[str, object]:
    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > _MAXIMUM_DOCUMENT_BYTES
    ):
        raise _error("invalid_wire", "readiness bundle document wire size differs")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error("invalid_wire", "readiness bundle document is invalid JSON") from exc
    value = _object(decoded, name="readiness bundle document")
    if canonical_json_bytes(value) != payload:
        raise _error("noncanonical_wire", "readiness bundle document is not canonical")
    return value


def _parse_artifact_record(value: object) -> ArtifactRecord:
    raw = _object(value, name="evidence artifact")
    _exact_fields(
        raw,
        frozenset(
            {
                "logical_name",
                "location",
                "sha256",
                "size_bytes",
                "media_type",
                "role",
            }
        ),
        name="ArtifactRecord",
    )
    try:
        return ArtifactRecord(
            logical_name=_text(raw["logical_name"], name="artifact logical_name"),
            location=_text(raw["location"], name="artifact location"),
            sha256=_digest(raw["sha256"], name="artifact sha256"),
            size_bytes=_integer(
                raw["size_bytes"], name="artifact size_bytes", positive=True
            ),
            media_type=_text(raw["media_type"], name="artifact media_type"),
            role=_text(raw["role"], name="artifact role"),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ResearchReadinessBundleDocumentError):
            raise
        raise _error("invalid_artifact_reference", str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ResearchReadinessEvidenceReferenceV1:
    """Minimal output-evidence reference layered on the shared ArtifactRecord."""

    evidence_kind: str
    artifact: ArtifactRecord
    report_frame_hash: str
    parquet_uncompressed_bytes: int
    schema_version: str = _EVIDENCE_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _EVIDENCE_REFERENCE_SCHEMA:
            raise _error("unsupported_schema", "unsupported readiness evidence reference")
        kind = _text(self.evidence_kind, name="evidence_kind")
        if kind not in _EVIDENCE_ROLES:
            raise _error("invalid_evidence_kind", f"unsupported evidence kind:{kind}")
        if type(self.artifact) is not ArtifactRecord:
            raise _error("invalid_artifact_reference", "artifact record type differs")
        if self.artifact.media_type != _PARQUET_MEDIA_TYPE:
            raise _error("invalid_media_type", "readiness evidence must be parquet")
        if self.artifact.role != _EVIDENCE_ROLES[kind]:
            raise _error("artifact_role_mismatch", f"evidence role differs:{kind}")
        if self.artifact.size_bytes <= 0:
            raise _error("invalid_artifact_reference", "evidence payload must be non-empty")
        object.__setattr__(
            self,
            "report_frame_hash",
            _digest(self.report_frame_hash, name="report_frame_hash"),
        )
        object.__setattr__(
            self,
            "parquet_uncompressed_bytes",
            _integer(
                self.parquet_uncompressed_bytes,
                name="parquet_uncompressed_bytes",
            ),
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_kind": self.evidence_kind,
            "artifact": self.artifact.to_dict(),
            "report_frame_hash": self.report_frame_hash,
            "parquet_uncompressed_bytes": self.parquet_uncompressed_bytes,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessEvidenceReferenceV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "evidence_kind",
                    "artifact",
                    "report_frame_hash",
                    "parquet_uncompressed_bytes",
                }
            ),
            name="ResearchReadinessEvidenceReferenceV1",
        )
        return cls(
            schema_version=cast(str, value["schema_version"]),
            evidence_kind=cast(str, value["evidence_kind"]),
            artifact=_parse_artifact_record(value["artifact"]),
            report_frame_hash=cast(str, value["report_frame_hash"]),
            parquet_uncompressed_bytes=cast(
                int, value["parquet_uncompressed_bytes"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ResearchReadinessInputDocumentReferenceV1:
    """Exact content-addressed reference to one readiness authority document."""

    document_kind: str
    artifact: ArtifactRecord
    content_hash: str
    schema_version: str = _INPUT_DOCUMENT_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _INPUT_DOCUMENT_REFERENCE_SCHEMA:
            raise _error(
                "unsupported_schema", "unsupported readiness input document reference"
            )
        kind = _text(self.document_kind, name="input document kind")
        expected = _INPUT_DOCUMENTS.get(kind)
        if expected is None:
            raise _error(
                "input_document_reference_mismatch",
                "readiness input document kind differs",
            )
        if type(self.artifact) is not ArtifactRecord:
            raise _error(
                "input_document_reference_mismatch",
                "readiness input artifact type differs",
            )
        logical_name, media_type, role = expected
        if (
            self.artifact.logical_name != logical_name
            or self.artifact.media_type != media_type
            or self.artifact.role != role
            or self.artifact.size_bytes <= 0
        ):
            raise _error(
                "input_document_reference_mismatch",
                "readiness input artifact identity differs",
            )
        content_hash = _digest(self.content_hash, name="input document content_hash")
        if self.artifact.sha256 != content_hash:
            raise _error(
                "input_document_reference_mismatch",
                "readiness input document payload differs from its content identity",
            )
        object.__setattr__(self, "document_kind", kind)
        object.__setattr__(self, "content_hash", content_hash)

    @property
    def reference_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "document_kind": self.document_kind,
            "artifact": self.artifact.to_dict(),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessInputDocumentReferenceV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "document_kind",
                    "artifact",
                    "content_hash",
                }
            ),
            name="ResearchReadinessInputDocumentReferenceV1",
        )
        return cls(
            schema_version=cast(str, value["schema_version"]),
            document_kind=cast(str, value["document_kind"]),
            artifact=_parse_artifact_record(value["artifact"]),
            content_hash=cast(str, value["content_hash"]),
        )


def _spec_from_mapping(value: Mapping[str, object]) -> ResearchReadinessSpec:
    _exact_fields(
        value,
        frozenset(
            {
                "schema_version",
                "readiness_id",
                "version",
                "pseudo_label_trials",
                "random_seed",
                "signal_lag_sessions",
                "required_perturbation_names",
                "minimum_cross_sectional_observations",
                "minimum_valid_dates",
                "maximum_pseudo_label_pvalue",
                "minimum_observed_rank_ic",
                "minimum_lag_absolute_gap",
                "minimum_stability_retention",
                "minimum_incremental_rank_ic",
                "minimum_incremental_positive_date_fraction",
            }
        ),
        name="ResearchReadinessSpec",
    )
    raw_lags = value["signal_lag_sessions"]
    raw_perturbations = value["required_perturbation_names"]
    if not isinstance(raw_lags, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in raw_lags
    ):
        raise _error("invalid_spec", "signal_lag_sessions must be an integer array")
    if not isinstance(raw_perturbations, list) or not all(
        isinstance(item, str) for item in raw_perturbations
    ):
        raise _error("invalid_spec", "required perturbations must be a text array")
    try:
        return ResearchReadinessSpec(
            schema_version=cast(str, value["schema_version"]),
            readiness_id=_text(value["readiness_id"], name="readiness_id"),
            version=_text(value["version"], name="readiness version"),
            pseudo_label_trials=_integer(
                value["pseudo_label_trials"], name="pseudo_label_trials", positive=True
            ),
            random_seed=_signed_integer(value["random_seed"], name="random_seed"),
            signal_lag_sessions=tuple(raw_lags),
            required_perturbation_names=tuple(raw_perturbations),
            minimum_cross_sectional_observations=_integer(
                value["minimum_cross_sectional_observations"],
                name="minimum_cross_sectional_observations",
                positive=True,
            ),
            minimum_valid_dates=_integer(
                value["minimum_valid_dates"],
                name="minimum_valid_dates",
                positive=True,
            ),
            maximum_pseudo_label_pvalue=_number(
                value["maximum_pseudo_label_pvalue"],
                name="maximum_pseudo_label_pvalue",
            ),
            minimum_observed_rank_ic=_number(
                value["minimum_observed_rank_ic"], name="minimum_observed_rank_ic"
            ),
            minimum_lag_absolute_gap=_number(
                value["minimum_lag_absolute_gap"], name="minimum_lag_absolute_gap"
            ),
            minimum_stability_retention=_number(
                value["minimum_stability_retention"],
                name="minimum_stability_retention",
            ),
            minimum_incremental_rank_ic=_number(
                value["minimum_incremental_rank_ic"],
                name="minimum_incremental_rank_ic",
            ),
            minimum_incremental_positive_date_fraction=_number(
                value["minimum_incremental_positive_date_fraction"],
                name="minimum_incremental_positive_date_fraction",
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ResearchReadinessBundleDocumentError):
            raise
        raise _error("invalid_spec", str(exc)) from exc


_NEGATIVE_IDENTITY_FIELDS: Final = frozenset(
    {
        "schema_version",
        "readiness_spec_hash",
        "candidate_score_hash",
        "label_values_hash",
        "label_validity_hash",
        "scoring_eligibility_hash",
        "observed_rank_ic_mean",
        "observed_valid_dates",
        "pseudo_label_pvalue",
        "observed_results_hash",
        "null_distribution_hash",
        "lag_results_hash",
        "lag_daily_results_hash",
        "passed",
    }
)
_STABILITY_IDENTITY_FIELDS: Final = frozenset(
    {
        "schema_version",
        "readiness_spec_hash",
        "candidate_score_hash",
        "label_values_hash",
        "label_validity_hash",
        "scoring_eligibility_hash",
        "perturbation_score_hashes",
        "scenario_results_hash",
        "daily_results_hash",
        "passed",
    }
)
_INCREMENTAL_IDENTITY_FIELDS: Final = frozenset(
    {
        "schema_version",
        "readiness_spec_hash",
        "candidate_score_hash",
        "baseline_score_hash",
        "label_values_hash",
        "label_validity_hash",
        "scoring_eligibility_hash",
        "daily_results_hash",
        "candidate_rank_ic_mean",
        "baseline_rank_ic_mean",
        "incremental_rank_ic_mean",
        "positive_date_fraction",
        "valid_dates",
        "passed",
    }
)


def _normalize_negative_identity(value: object) -> Mapping[str, object]:
    raw = _object(value, name="negative control identity")
    _exact_fields(raw, _NEGATIVE_IDENTITY_FIELDS, name="negative control identity")
    if raw["schema_version"] != "negative-control-report/v2":
        raise _error("unsupported_schema", "negative control identity schema differs")
    result: dict[str, object] = {
        "schema_version": "negative-control-report/v2",
    }
    for name in (
        "readiness_spec_hash",
        "candidate_score_hash",
        "label_values_hash",
        "label_validity_hash",
        "scoring_eligibility_hash",
        "observed_results_hash",
        "null_distribution_hash",
        "lag_results_hash",
        "lag_daily_results_hash",
    ):
        result[name] = _digest(raw[name], name=f"negative control {name}")
    result["observed_rank_ic_mean"] = _optional_number(
        raw["observed_rank_ic_mean"], name="observed_rank_ic_mean"
    )
    result["observed_valid_dates"] = _integer(
        raw["observed_valid_dates"], name="observed_valid_dates"
    )
    pvalue = _optional_number(
        raw["pseudo_label_pvalue"], name="pseudo_label_pvalue"
    )
    if pvalue is not None and not 0.0 <= pvalue <= 1.0:
        raise _error("invalid_number", "pseudo_label_pvalue lies outside [0, 1]")
    result["pseudo_label_pvalue"] = pvalue
    result["passed"] = _tristate(raw["passed"], name="negative control passed")
    return MappingProxyType(result)


def _normalize_hash_mapping(
    value: object,
    *,
    name: str,
) -> Mapping[str, str]:
    raw = _object(value, name=name)
    result: dict[str, str] = {}
    for key, item in raw.items():
        clean = _text(key, name=f"{name} key")
        if clean in result:
            raise _error("duplicate_identity", f"{name} key repeats")
        result[clean] = _digest(item, name=f"{name}:{clean}")
    if not result:
        raise _error("invalid_identity", f"{name} must not be empty")
    return MappingProxyType(dict(sorted(result.items())))


def _normalize_stability_identity(value: object) -> Mapping[str, object]:
    raw = _object(value, name="parameter stability identity")
    _exact_fields(raw, _STABILITY_IDENTITY_FIELDS, name="parameter stability identity")
    if raw["schema_version"] != "parameter-stability-report/v2":
        raise _error("unsupported_schema", "parameter stability identity schema differs")
    result: dict[str, object] = {
        "schema_version": "parameter-stability-report/v2",
    }
    for name in (
        "readiness_spec_hash",
        "candidate_score_hash",
        "label_values_hash",
        "label_validity_hash",
        "scoring_eligibility_hash",
        "scenario_results_hash",
        "daily_results_hash",
    ):
        result[name] = _digest(raw[name], name=f"parameter stability {name}")
    result["perturbation_score_hashes"] = _normalize_hash_mapping(
        raw["perturbation_score_hashes"],
        name="perturbation_score_hashes",
    )
    result["passed"] = _tristate(raw["passed"], name="parameter stability passed")
    return MappingProxyType(result)


def _normalize_incremental_identity(value: object) -> Mapping[str, object]:
    raw = _object(value, name="incremental baseline identity")
    _exact_fields(raw, _INCREMENTAL_IDENTITY_FIELDS, name="incremental identity")
    if raw["schema_version"] != "incremental-baseline-report/v2":
        raise _error("unsupported_schema", "incremental baseline identity schema differs")
    result: dict[str, object] = {
        "schema_version": "incremental-baseline-report/v2",
    }
    for name in (
        "readiness_spec_hash",
        "candidate_score_hash",
        "baseline_score_hash",
        "label_values_hash",
        "label_validity_hash",
        "scoring_eligibility_hash",
        "daily_results_hash",
    ):
        result[name] = _digest(raw[name], name=f"incremental baseline {name}")
    for name in (
        "candidate_rank_ic_mean",
        "baseline_rank_ic_mean",
        "incremental_rank_ic_mean",
        "positive_date_fraction",
    ):
        result[name] = _optional_number(raw[name], name=name)
    fraction = result["positive_date_fraction"]
    if fraction is not None and not 0.0 <= cast(float, fraction) <= 1.0:
        raise _error("invalid_number", "positive_date_fraction lies outside [0, 1]")
    result["valid_dates"] = _integer(raw["valid_dates"], name="valid_dates")
    result["passed"] = _tristate(raw["passed"], name="incremental baseline passed")
    return MappingProxyType(result)


def _parameter_identity(report: ParameterStabilityReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "readiness_spec_hash": report.readiness_spec_hash,
        "candidate_score_hash": report.candidate_score_hash,
        "label_values_hash": report.label_values_hash,
        "label_validity_hash": report.label_validity_hash,
        "scoring_eligibility_hash": report.scoring_eligibility_hash,
        "perturbation_score_hashes": dict(report.perturbation_score_hashes),
        "scenario_results_hash": report.scenario_results_hash,
        "daily_results_hash": report.daily_results_hash,
        "passed": report.passed,
    }


def _incremental_identity(report: IncrementalBaselineReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "readiness_spec_hash": report.readiness_spec_hash,
        "candidate_score_hash": report.candidate_score_hash,
        "baseline_score_hash": report.baseline_score_hash,
        "label_values_hash": report.label_values_hash,
        "label_validity_hash": report.label_validity_hash,
        "scoring_eligibility_hash": report.scoring_eligibility_hash,
        "daily_results_hash": report.daily_results_hash,
        "candidate_rank_ic_mean": report.candidate_rank_ic_mean,
        "baseline_rank_ic_mean": report.baseline_rank_ic_mean,
        "incremental_rank_ic_mean": report.incremental_rank_ic_mean,
        "positive_date_fraction": report.positive_date_fraction,
        "valid_dates": report.valid_dates,
        "passed": report.passed,
    }


def _derive_decision(
    negative: bool | None,
    stability: bool | None,
    incremental: bool | None,
) -> tuple[DLReadinessVerdict, tuple[str, ...]]:
    states = (negative, stability, incremental)
    prefixes = ("negative_control", "parameter_stability", "incremental_baseline")
    reasons = [
        f"{prefix}_{'inconclusive' if state is None else 'failed'}"
        for prefix, state in zip(prefixes, states, strict=True)
        if state is not True
    ]
    if any(state is False for state in states):
        verdict = DLReadinessVerdict.NO_GO
    elif any(state is None for state in states):
        verdict = DLReadinessVerdict.INCONCLUSIVE
    elif all(state is True for state in states):
        verdict = DLReadinessVerdict.GO_RESEARCH_ONLY
        reasons.append("traditional_model_evidence_passed")
    else:  # pragma: no cover - identity normalizers enforce tri-state values.
        raise _error("invalid_decision", "readiness tri-state decision is invalid")
    return verdict, tuple(sorted(reasons))


@dataclass(frozen=True, slots=True)
class ResearchReadinessBundleDocumentV1:
    """Canonical scalar identity plus immutable references for one bundle."""

    readiness_spec: ResearchReadinessSpec
    readiness_spec_hash: str
    input_manifest_reference: ResearchReadinessInputDocumentReferenceV1
    input_manifest_hash: str
    authority_receipt_reference: ResearchReadinessInputDocumentReferenceV1
    authority_receipt_hash: str
    candidate_evidence_hash: str
    baseline_evidence_hash: str
    outer_validation_receipt_hash: str
    negative_control_identity: Mapping[str, object]
    negative_control_hash: str
    parameter_stability_identity: Mapping[str, object]
    parameter_stability_hash: str
    incremental_baseline_identity: Mapping[str, object]
    incremental_baseline_hash: str
    evidence_artifacts: Mapping[str, ResearchReadinessEvidenceReferenceV1]
    verdict: DLReadinessVerdict | str
    reason_codes: tuple[str, ...]
    bundle_content_hash: str
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _DOCUMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _DOCUMENT_SCHEMA:
            raise _error("unsupported_schema", "unsupported readiness bundle document")
        if type(self.readiness_spec) is not ResearchReadinessSpec:
            raise _error("invalid_spec", "readiness specification type differs")
        for name in (
            "readiness_spec_hash",
            "input_manifest_hash",
            "authority_receipt_hash",
            "candidate_evidence_hash",
            "baseline_evidence_hash",
            "outer_validation_receipt_hash",
            "negative_control_hash",
            "parameter_stability_hash",
            "incremental_baseline_hash",
            "bundle_content_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.readiness_spec.content_hash != self.readiness_spec_hash:
            raise _error("spec_hash_mismatch", "readiness specification hash differs")
        if (
            type(self.input_manifest_reference)
            is not ResearchReadinessInputDocumentReferenceV1
            or self.input_manifest_reference.document_kind != "input_manifest"
            or self.input_manifest_reference.content_hash != self.input_manifest_hash
            or type(self.authority_receipt_reference)
            is not ResearchReadinessInputDocumentReferenceV1
            or self.authority_receipt_reference.document_kind != "authority_receipt"
            or self.authority_receipt_reference.content_hash
            != self.authority_receipt_hash
        ):
            raise _error(
                "input_document_reference_mismatch",
                "readiness authority document references differ",
            )
        negative = _normalize_negative_identity(self.negative_control_identity)
        stability = _normalize_stability_identity(self.parameter_stability_identity)
        incremental = _normalize_incremental_identity(
            self.incremental_baseline_identity
        )
        object.__setattr__(self, "negative_control_identity", negative)
        object.__setattr__(self, "parameter_stability_identity", stability)
        object.__setattr__(self, "incremental_baseline_identity", incremental)
        expected_report_hashes = (
            ("negative_control_hash", hash_json(dict(negative))),
            ("parameter_stability_hash", hash_json(dict(stability))),
            ("incremental_baseline_hash", hash_json(dict(incremental))),
        )
        for name, expected in expected_report_hashes:
            if getattr(self, name) != expected:
                raise _error("report_hash_mismatch", f"{name} differs")
        self._verify_report_lineage(negative, stability, incremental)
        artifacts = self._normalize_artifacts(self.evidence_artifacts)
        object.__setattr__(self, "evidence_artifacts", artifacts)
        self._verify_frame_links(negative, stability, incremental, artifacts)
        try:
            verdict = DLReadinessVerdict(self.verdict)
        except (TypeError, ValueError) as exc:
            raise _error("invalid_decision", "readiness verdict is invalid") from exc
        reasons = tuple(self.reason_codes)
        if (
            not reasons
            or any(not isinstance(item, str) or not item for item in reasons)
            or reasons != tuple(sorted(set(reasons)))
        ):
            raise _error("invalid_decision", "readiness reason codes differ")
        expected_verdict, expected_reasons = _derive_decision(
            cast(bool | None, negative["passed"]),
            cast(bool | None, stability["passed"]),
            cast(bool | None, incremental["passed"]),
        )
        if verdict is not expected_verdict or reasons != expected_reasons:
            raise _error("decision_mismatch", "document decision differs from reports")
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "reason_codes", reasons)
        if _boolean(self.research_only, name="research_only") is not True:
            raise _error("invalid_release_flags", "bundle document is research-only")
        if _boolean(self.production_ready, name="production_ready") is not False:
            raise _error("invalid_release_flags", "bundle document is not production-ready")
        expected_bundle_hash = cast(
            str,
            hash_json(
                {
                    "schema_version": "research-readiness-bundle/v3",
                    "readiness_spec_hash": self.readiness_spec_hash,
                    "candidate_evidence_hash": self.candidate_evidence_hash,
                    "baseline_evidence_hash": self.baseline_evidence_hash,
                    "outer_validation_receipt_hash": (
                        self.outer_validation_receipt_hash
                    ),
                    "negative_control_hash": self.negative_control_hash,
                    "parameter_stability_hash": self.parameter_stability_hash,
                    "incremental_baseline_hash": self.incremental_baseline_hash,
                    "verdict": verdict.value,
                    "reason_codes": list(reasons),
                    "research_only": True,
                    "production_ready": False,
                }
            ),
        )
        if self.bundle_content_hash != expected_bundle_hash:
            raise _error("bundle_hash_mismatch", "bundle scalar identity differs")

    def _verify_report_lineage(
        self,
        negative: Mapping[str, object],
        stability: Mapping[str, object],
        incremental: Mapping[str, object],
    ) -> None:
        for identity in (negative, stability, incremental):
            if identity["readiness_spec_hash"] != self.readiness_spec_hash:
                raise _error("report_lineage_mismatch", "report specification differs")
        for identity in (stability, incremental):
            for name in (
                "candidate_score_hash",
                "label_values_hash",
                "label_validity_hash",
                "scoring_eligibility_hash",
            ):
                if identity[name] != negative[name]:
                    raise _error("report_lineage_mismatch", f"report {name} differs")
        perturbations = cast(
            Mapping[str, str], stability["perturbation_score_hashes"]
        )
        if tuple(perturbations) != self.readiness_spec.required_perturbation_names:
            raise _error(
                "perturbation_contract_mismatch",
                "report perturbation set differs from readiness specification",
            )

    @staticmethod
    def _normalize_artifacts(
        value: Mapping[str, ResearchReadinessEvidenceReferenceV1],
    ) -> Mapping[str, ResearchReadinessEvidenceReferenceV1]:
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str)
            and type(reference) is ResearchReadinessEvidenceReferenceV1
            for key, reference in value.items()
        ):
            raise _error("invalid_artifact_reference", "evidence references differ")
        normalized = dict(sorted(value.items()))
        if frozenset(normalized) != frozenset(_EVIDENCE_ROLES):
            raise _error("evidence_set_mismatch", "evidence artifact set differs")
        for kind, reference in normalized.items():
            if reference.evidence_kind != kind:
                raise _error("evidence_set_mismatch", f"evidence key differs:{kind}")
        records = tuple(reference.artifact for reference in normalized.values())
        if len({item.logical_name for item in records}) != len(records):
            raise _error("duplicate_artifact", "evidence logical names repeat")
        return MappingProxyType(normalized)

    @staticmethod
    def _verify_frame_links(
        negative: Mapping[str, object],
        stability: Mapping[str, object],
        incremental: Mapping[str, object],
        artifacts: Mapping[str, ResearchReadinessEvidenceReferenceV1],
    ) -> None:
        expected = {
            "negative_observed_daily": negative["observed_results_hash"],
            "negative_pseudo_label_trials": negative["null_distribution_hash"],
            "negative_lag_summary": negative["lag_results_hash"],
            "negative_lag_daily": negative["lag_daily_results_hash"],
            "stability_scenario_summary": stability["scenario_results_hash"],
            "stability_daily": stability["daily_results_hash"],
            "incremental_baseline_daily": incremental["daily_results_hash"],
        }
        for kind, digest in expected.items():
            if artifacts[kind].report_frame_hash != digest:
                raise _error("frame_link_mismatch", f"report frame differs:{kind}")

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        verdict = self.verdict
        if not isinstance(verdict, DLReadinessVerdict):  # pragma: no cover
            raise RuntimeError("readiness document verdict was not normalized")
        stability_identity = dict(self.parameter_stability_identity)
        stability_identity["perturbation_score_hashes"] = dict(
            cast(
                Mapping[str, str],
                self.parameter_stability_identity["perturbation_score_hashes"],
            )
        )
        return {
            "schema_version": self.schema_version,
            "readiness_spec": self.readiness_spec.to_dict(),
            "readiness_spec_hash": self.readiness_spec_hash,
            "input_manifest_reference": self.input_manifest_reference.to_dict(),
            "input_manifest_hash": self.input_manifest_hash,
            "authority_receipt_reference": (
                self.authority_receipt_reference.to_dict()
            ),
            "authority_receipt_hash": self.authority_receipt_hash,
            "candidate_evidence_hash": self.candidate_evidence_hash,
            "baseline_evidence_hash": self.baseline_evidence_hash,
            "outer_validation_receipt_hash": self.outer_validation_receipt_hash,
            "negative_control_identity": dict(self.negative_control_identity),
            "negative_control_hash": self.negative_control_hash,
            "parameter_stability_identity": stability_identity,
            "parameter_stability_hash": self.parameter_stability_hash,
            "incremental_baseline_identity": dict(
                self.incremental_baseline_identity
            ),
            "incremental_baseline_hash": self.incremental_baseline_hash,
            "evidence_artifacts": {
                name: reference.to_dict()
                for name, reference in self.evidence_artifacts.items()
            },
            "verdict": verdict.value,
            "reason_codes": list(self.reason_codes),
            "bundle_content_hash": self.bundle_content_hash,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessBundleDocumentV1":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "readiness_spec",
                    "readiness_spec_hash",
                    "input_manifest_reference",
                    "input_manifest_hash",
                    "authority_receipt_reference",
                    "authority_receipt_hash",
                    "candidate_evidence_hash",
                    "baseline_evidence_hash",
                    "outer_validation_receipt_hash",
                    "negative_control_identity",
                    "negative_control_hash",
                    "parameter_stability_identity",
                    "parameter_stability_hash",
                    "incremental_baseline_identity",
                    "incremental_baseline_hash",
                    "evidence_artifacts",
                    "verdict",
                    "reason_codes",
                    "bundle_content_hash",
                    "research_only",
                    "production_ready",
                }
            ),
            name="ResearchReadinessBundleDocumentV1",
        )
        raw_artifacts = _object(value["evidence_artifacts"], name="evidence artifacts")
        raw_reasons = value["reason_codes"]
        if not isinstance(raw_reasons, list) or not all(
            isinstance(item, str) for item in raw_reasons
        ):
            raise _error("invalid_decision", "reason_codes must be a text array")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            readiness_spec=_spec_from_mapping(
                _object(value["readiness_spec"], name="readiness_spec")
            ),
            readiness_spec_hash=cast(str, value["readiness_spec_hash"]),
            input_manifest_reference=(
                ResearchReadinessInputDocumentReferenceV1.from_mapping(
                    _object(
                        value["input_manifest_reference"],
                        name="input_manifest_reference",
                    )
                )
            ),
            input_manifest_hash=cast(str, value["input_manifest_hash"]),
            authority_receipt_reference=(
                ResearchReadinessInputDocumentReferenceV1.from_mapping(
                    _object(
                        value["authority_receipt_reference"],
                        name="authority_receipt_reference",
                    )
                )
            ),
            authority_receipt_hash=cast(str, value["authority_receipt_hash"]),
            candidate_evidence_hash=cast(str, value["candidate_evidence_hash"]),
            baseline_evidence_hash=cast(str, value["baseline_evidence_hash"]),
            outer_validation_receipt_hash=cast(
                str, value["outer_validation_receipt_hash"]
            ),
            negative_control_identity=_object(
                value["negative_control_identity"],
                name="negative_control_identity",
            ),
            negative_control_hash=cast(str, value["negative_control_hash"]),
            parameter_stability_identity=_object(
                value["parameter_stability_identity"],
                name="parameter_stability_identity",
            ),
            parameter_stability_hash=cast(str, value["parameter_stability_hash"]),
            incremental_baseline_identity=_object(
                value["incremental_baseline_identity"],
                name="incremental_baseline_identity",
            ),
            incremental_baseline_hash=cast(str, value["incremental_baseline_hash"]),
            evidence_artifacts={
                name: ResearchReadinessEvidenceReferenceV1.from_mapping(
                    _object(reference, name=f"evidence_artifacts:{name}")
                )
                for name, reference in raw_artifacts.items()
            },
            verdict=cast(str, value["verdict"]),
            reason_codes=tuple(raw_reasons),
            bundle_content_hash=cast(str, value["bundle_content_hash"]),
            research_only=cast(bool, value["research_only"]),
            production_ready=cast(bool, value["production_ready"]),
        )

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "ResearchReadinessBundleDocumentV1":
        result = cls.from_mapping(_parse_wire(payload))
        if result.to_wire_bytes() != payload:  # pragma: no cover - defensive.
            raise _error("wire_identity_mismatch", "document wire identity differs")
        return result


def _parquet_uncompressed_bytes(parquet: pq.ParquetFile) -> int:
    metadata = parquet.metadata
    if metadata is None:
        raise _error("invalid_parquet", "parquet metadata is missing")
    total = 0
    for row_group_offset in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_offset)
        for column_offset in range(row_group.num_columns):
            total += int(row_group.column(column_offset).total_uncompressed_size)
    return total


@dataclass(frozen=True, slots=True)
class _EncodedEvidence:
    payload: bytes
    uncompressed_bytes: int
    row_count: int
    column_count: int
    loaded_memory_bytes: int


@dataclass(frozen=True, slots=True)
class _EvidenceBudget:
    maximum_compressed_bytes: int
    maximum_uncompressed_bytes: int
    maximum_total_rows: int
    maximum_total_columns: int
    maximum_loaded_memory_bytes: int


def _encode_frame(frame: pd.DataFrame) -> _EncodedEvidence:
    value = pd.DataFrame(frame).copy(deep=True)
    buffer = BytesIO()
    try:
        value.to_parquet(
            buffer,
            engine="pyarrow",
            compression="zstd",
            index=None,
        )
        payload = buffer.getvalue()
        parquet = pq.ParquetFile(pa.BufferReader(payload))
        decoded = parquet.read().to_pandas()
        pd.testing.assert_frame_equal(
            value,
            decoded,
            check_exact=True,
            check_index_type=False,
        )
    except (AssertionError, OSError, TypeError, ValueError, pa.ArrowException) as exc:
        raise _error("parquet_roundtrip_failed", "evidence frame is not replayable") from exc
    if not payload or hash_frame(decoded) != hash_frame(value):
        raise _error("parquet_roundtrip_failed", "evidence frame identity differs")
    return _EncodedEvidence(
        payload=payload,
        uncompressed_bytes=_parquet_uncompressed_bytes(parquet),
        row_count=len(decoded.index),
        column_count=len(decoded.columns),
        loaded_memory_bytes=int(
            decoded.memory_usage(index=True, deep=True).sum()
        ),
    )


def _bundle_frames(bundle: ResearchReadinessBundle) -> Mapping[str, pd.DataFrame]:
    return {
        "negative_observed_daily": bundle.negative_control.observed_results,
        "negative_pseudo_label_trials": bundle.negative_control.null_distribution,
        "negative_lag_summary": bundle.negative_control.lag_results,
        "negative_lag_daily": bundle.negative_control.lag_daily_results,
        "stability_scenario_summary": (
            bundle.parameter_stability.scenario_results
        ),
        "stability_daily": bundle.parameter_stability.daily_results,
        "incremental_baseline_daily": bundle.incremental_baseline.daily_results,
    }


def _enforce_evidence_budget(
    encoded: Mapping[str, _EncodedEvidence], budget: _EvidenceBudget
) -> None:
    if frozenset(encoded) != frozenset(_EVIDENCE_ROLES):
        raise _error("evidence_set_mismatch", "evidence artifact set differs")
    totals = (
        (sum(len(item.payload) for item in encoded.values()), budget.maximum_compressed_bytes),
        (
            sum(item.uncompressed_bytes for item in encoded.values()),
            budget.maximum_uncompressed_bytes,
        ),
        (sum(item.row_count for item in encoded.values()), budget.maximum_total_rows),
        (
            sum(item.column_count for item in encoded.values()),
            budget.maximum_total_columns,
        ),
        (
            sum(item.loaded_memory_bytes for item in encoded.values()),
            budget.maximum_loaded_memory_bytes,
        ),
    )
    if any(observed > maximum for observed, maximum in totals):
        raise _error(
            "evidence_budget_exceeded",
            "aggregate readiness evidence exceeds its resource budget",
        )


def _readiness_frame_record(reference: ResearchReadinessFrameReferenceV1) -> ArtifactRecord:
    return ArtifactRecord(
        logical_name=reference.logical_name,
        location=reference.location,
        sha256=reference.payload_sha256,
        size_bytes=reference.size_bytes,
        media_type=reference.media_type,
        role=reference.semantic_role,
    )


def _publish_input_document(
    store: ArtifactStore,
    *,
    document_kind: str,
    payload: bytes,
    content_hash: str,
) -> ResearchReadinessInputDocumentReferenceV1:
    expected = _INPUT_DOCUMENTS[document_kind]
    logical_name, media_type, role = expected
    if len(payload) > _MAXIMUM_DOCUMENT_BYTES:
        raise _error("document_budget_exceeded", "readiness input document is too large")
    try:
        record = store.put_bytes(
            logical_name,
            payload,
            media_type=media_type,
            role=role,
        )
    except (ArtifactError, OSError, TypeError, ValueError) as exc:
        raise _error(
            "input_document_persistence_failed",
            "readiness input document could not be persisted",
        ) from exc
    if record.sha256 != content_hash:
        raise _error(
            "input_document_identity_mismatch",
            "readiness input document content hash differs",
        )
    try:
        restored_payload = store.read_bytes(record)
        if restored_payload != payload:
            raise _error(
                "input_document_identity_mismatch",
                "readiness input document replay bytes differ",
            )
        if document_kind == "input_manifest":
            restored_hash = ResearchReadinessInputManifestV1.from_wire_bytes(
                restored_payload
            ).content_hash
        elif document_kind == "authority_receipt":
            restored_hash = ResearchReadinessAuthorityReceiptV1.from_wire_bytes(
                restored_payload
            ).content_hash
        else:  # pragma: no cover - guarded by the private call sites.
            raise _error(
                "input_document_reference_mismatch",
                "readiness input document kind differs",
            )
    except ResearchReadinessBundleDocumentError:
        raise
    except (
        ArtifactError,
        OSError,
        ResearchReadinessInputError,
        TypeError,
        ValueError,
    ) as exc:
        raise _error(
            "input_document_persistence_failed",
            "readiness input document could not be replayed",
        ) from exc
    if restored_hash != content_hash:
        raise _error(
            "input_document_identity_mismatch",
            "readiness input document replay identity differs",
        )
    return ResearchReadinessInputDocumentReferenceV1(
        document_kind=document_kind,
        artifact=record,
        content_hash=content_hash,
    )


@dataclass(frozen=True, slots=True)
class _ReplayedReadinessAuthority:
    """Persisted input authority after independent close/reopen verification.

    This deliberately is *not* ``AuthorityBoundResearchReadinessInputs``.  The
    latter is a process-local one-shot handoff that only a live resolver may
    create.  A loader instead verifies the persisted manifest, receipt and
    every referenced frame without minting resolver authority.
    """

    loaded: LoadedResearchReadinessInputs
    authority_receipt: ResearchReadinessAuthorityReceiptV1


def _manifest_frame_references(
    manifest: ResearchReadinessInputManifestV1,
) -> tuple[ResearchReadinessFrameReferenceV1, ...]:
    return (
        manifest.score_artifacts.candidate_scores,
        manifest.score_artifacts.baseline_scores,
        *manifest.score_artifacts.perturbation_scores.values(),
        manifest.label_values,
        manifest.label_validity,
        manifest.scoring_eligibility,
    )


def _verify_replayed_authority_binding(
    loaded: LoadedResearchReadinessInputs,
    receipt: ResearchReadinessAuthorityReceiptV1,
) -> None:
    """Mirror the resolver handoff's content checks without forging its seal."""

    if type(loaded) is not LoadedResearchReadinessInputs or type(
        receipt
    ) is not ResearchReadinessAuthorityReceiptV1:
        raise _error("input_authority_mismatch", "readiness authority types differ")
    try:
        loaded.verify_content()
        manifest = loaded.manifest
        expected = {
            "manifest_hash": manifest.content_hash,
            "research_run_spec_hash": manifest.research_run_spec_hash,
            "experiment_spec_hash": manifest.experiment_spec_hash,
            "scientific_lineage_manifest_hash": (
                manifest.scientific_lineage_manifest_hash
            ),
            "phase_one_manifest_hash": manifest.phase_one_manifest_hash,
            "outer_evaluation_completion_hash": (
                manifest.outer_evaluation_completion_hash
            ),
            "outer_evaluation_result_hash": manifest.outer_evaluation_result_hash,
            "outer_execution_snapshot_hash": manifest.outer_execution_snapshot_hash,
        }
        if any(getattr(receipt, name) != digest for name, digest in expected.items()):
            raise _error(
                "input_authority_mismatch",
                "readiness authority receipt differs from its manifest",
            )
        if dict(receipt.parent_payload_hashes) != dict(
            manifest.parent_artifact_hashes
        ):
            raise _error(
                "input_authority_mismatch",
                "readiness parent artifacts differ",
            )
        references = _manifest_frame_references(manifest)
        if (
            receipt.total_compressed_bytes
            != sum(reference.size_bytes for reference in references)
            or receipt.total_parquet_uncompressed_bytes
            != sum(
                reference.parquet_uncompressed_bytes for reference in references
            )
            or receipt.total_loaded_memory_bytes != loaded.loaded_memory_bytes
            or manifest.research_only is not True
            or manifest.production_ready is not False
            or receipt.research_only is not True
            or receipt.production_ready is not False
        ):
            raise _error(
                "input_authority_mismatch",
                "readiness authority resource or release binding differs",
            )
    except ResearchReadinessBundleDocumentError:
        raise
    except (ResearchReadinessInputError, RuntimeError, TypeError, ValueError) as exc:
        raise _error(
            "input_authority_mismatch", "readiness input authority differs"
        ) from exc


def _verify_bundle_input_lineage(
    bundle: ResearchReadinessBundle,
    loaded: LoadedResearchReadinessInputs,
    receipt: ResearchReadinessAuthorityReceiptV1,
) -> None:
    _verify_replayed_authority_binding(loaded, receipt)
    manifest = loaded.manifest
    if (
        bundle.readiness_spec_hash != manifest.readiness_plan.readiness_spec_hash
        or bundle.candidate_evidence_hash
        != manifest.score_artifacts.candidate_scores.frame_hash
        or bundle.baseline_evidence_hash
        != manifest.score_artifacts.baseline_scores.frame_hash
        or bundle.outer_validation_receipt_hash
        != manifest.outer_validation_receipt_hash
    ):
        raise _error(
            "input_lineage_mismatch",
            "readiness bundle authority hashes differ from its input manifest",
        )
    expected_common = {
        "candidate_score_hash": hash_frame(loaded.candidate_scores),
        "label_values_hash": hash_frame(loaded.labels),
        "label_validity_hash": hash_frame(loaded.label_validity),
        "scoring_eligibility_hash": hash_frame(loaded.scoring_eligibility),
    }
    reports = (
        bundle.negative_control,
        bundle.parameter_stability,
        bundle.incremental_baseline,
    )
    if any(
        any(getattr(report, name) != digest for name, digest in expected_common.items())
        for report in reports
    ):
        raise _error(
            "input_lineage_mismatch",
            "readiness report inputs differ from the authority manifest",
        )
    if (
        bundle.incremental_baseline.baseline_score_hash
        != hash_frame(loaded.baseline_scores)
        or dict(bundle.parameter_stability.perturbation_score_hashes)
        != {
            name: hash_frame(frame)
            for name, frame in loaded.perturbation_scores.items()
        }
    ):
        raise _error(
            "input_lineage_mismatch",
            "readiness baseline or perturbation inputs differ",
        )


class ResearchReadinessBundleProducerV1:
    """Publish evidence frames and one canonical bundle document."""

    __slots__ = ("_artifact_store", "_budget")

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        maximum_evidence_bytes: int = _DEFAULT_MAXIMUM_EVIDENCE_BYTES,
        maximum_uncompressed_bytes: int = _DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES,
        maximum_total_rows: int = _DEFAULT_MAXIMUM_TOTAL_ROWS,
        maximum_total_columns: int = _DEFAULT_MAXIMUM_TOTAL_COLUMNS,
        maximum_loaded_memory_bytes: int = _DEFAULT_MAXIMUM_LOADED_MEMORY_BYTES,
    ) -> None:
        if type(artifact_store) is not ArtifactStore:
            raise TypeError("readiness bundle producer requires an exact ArtifactStore")
        self._artifact_store = artifact_store
        self._budget = _EvidenceBudget(
            maximum_compressed_bytes=_integer(
                maximum_evidence_bytes,
                name="maximum_evidence_bytes",
                positive=True,
            ),
            maximum_uncompressed_bytes=_integer(
                maximum_uncompressed_bytes,
                name="maximum_uncompressed_bytes",
                positive=True,
            ),
            maximum_total_rows=_integer(
                maximum_total_rows,
                name="maximum_total_rows",
                positive=True,
            ),
            maximum_total_columns=_integer(
                maximum_total_columns,
                name="maximum_total_columns",
                positive=True,
            ),
            maximum_loaded_memory_bytes=_integer(
                maximum_loaded_memory_bytes,
                name="maximum_loaded_memory_bytes",
                positive=True,
            ),
        )

    def publish(
        self,
        bundle: ResearchReadinessBundle,
        *,
        authority_bound: AuthorityBoundResearchReadinessInputs,
    ) -> tuple[ResearchReadinessBundleDocumentV1, ArtifactRecord]:
        if type(bundle) is not ResearchReadinessBundle:
            raise TypeError("readiness bundle producer requires an exact bundle")
        if type(authority_bound) is not AuthorityBoundResearchReadinessInputs:
            raise TypeError("readiness bundle producer requires exact input authority")
        try:
            bundle_hash = bundle.content_hash
        except (RuntimeError, TypeError, ValueError) as exc:
            raise _error("invalid_bundle", "readiness bundle verification failed") from exc
        try:
            authority_bound._verify_binding()
        except (ResearchReadinessInputError, RuntimeError, TypeError, ValueError) as exc:
            raise _error(
                "input_authority_mismatch", "readiness input authority differs"
            ) from exc
        _verify_bundle_input_lineage(
            bundle,
            authority_bound.loaded,
            authority_bound.authority_receipt,
        )
        encoded = {
            kind: _encode_frame(frame)
            for kind, frame in _bundle_frames(bundle).items()
        }
        _enforce_evidence_budget(encoded, self._budget)
        manifest = authority_bound.manifest
        receipt = authority_bound.authority_receipt
        manifest_reference = _publish_input_document(
            self._artifact_store,
            document_kind="input_manifest",
            payload=manifest.to_wire_bytes(),
            content_hash=manifest.content_hash,
        )
        receipt_reference = _publish_input_document(
            self._artifact_store,
            document_kind="authority_receipt",
            payload=receipt.to_wire_bytes(),
            content_hash=receipt.content_hash,
        )
        references: dict[str, ResearchReadinessEvidenceReferenceV1] = {}
        for kind, frame in _bundle_frames(bundle).items():
            value = encoded[kind]
            record = self._artifact_store.put_bytes(
                f"research-readiness.{kind}.parquet",
                value.payload,
                media_type=_PARQUET_MEDIA_TYPE,
                role=_EVIDENCE_ROLES[kind],
            )
            references[kind] = ResearchReadinessEvidenceReferenceV1(
                evidence_kind=kind,
                artifact=record,
                report_frame_hash=hash_frame(frame),
                parquet_uncompressed_bytes=value.uncompressed_bytes,
            )
        negative_identity = bundle.negative_control.identity_payload()
        stability_identity = _parameter_identity(bundle.parameter_stability)
        incremental_identity = _incremental_identity(bundle.incremental_baseline)
        verdict = bundle.verdict
        if not isinstance(verdict, DLReadinessVerdict):  # pragma: no cover
            raise _error("invalid_bundle", "bundle verdict was not normalized")
        document = ResearchReadinessBundleDocumentV1(
            readiness_spec=bundle.negative_control.readiness_spec,
            readiness_spec_hash=bundle.readiness_spec_hash,
            input_manifest_reference=manifest_reference,
            input_manifest_hash=manifest.content_hash,
            authority_receipt_reference=receipt_reference,
            authority_receipt_hash=receipt.content_hash,
            candidate_evidence_hash=bundle.candidate_evidence_hash,
            baseline_evidence_hash=bundle.baseline_evidence_hash,
            outer_validation_receipt_hash=(
                bundle.outer_validation_receipt_hash
            ),
            negative_control_identity=negative_identity,
            negative_control_hash=bundle.negative_control.content_hash,
            parameter_stability_identity=stability_identity,
            parameter_stability_hash=bundle.parameter_stability.content_hash,
            incremental_baseline_identity=incremental_identity,
            incremental_baseline_hash=bundle.incremental_baseline.content_hash,
            evidence_artifacts=references,
            verdict=verdict,
            reason_codes=bundle.reason_codes,
            bundle_content_hash=bundle_hash,
        )
        document_record = self._artifact_store.put_bytes(
            _DOCUMENT_LOGICAL_NAME,
            document.to_wire_bytes(),
            media_type=_DOCUMENT_MEDIA_TYPE,
            role=_DOCUMENT_ROLE,
        )
        return document, document_record


class ResearchReadinessBundleLoaderV1:
    """Reload immutable evidence and reconstruct a fully verified bundle."""

    __slots__ = (
        "_artifact_store",
        "_budget",
    )

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        maximum_evidence_bytes: int = _DEFAULT_MAXIMUM_EVIDENCE_BYTES,
        maximum_uncompressed_bytes: int = _DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES,
        maximum_total_rows: int = _DEFAULT_MAXIMUM_TOTAL_ROWS,
        maximum_total_columns: int = _DEFAULT_MAXIMUM_TOTAL_COLUMNS,
        maximum_loaded_memory_bytes: int = _DEFAULT_MAXIMUM_LOADED_MEMORY_BYTES,
    ) -> None:
        if type(artifact_store) is not ArtifactStore:
            raise TypeError("readiness bundle loader requires an exact ArtifactStore")
        self._artifact_store = artifact_store
        self._budget = _EvidenceBudget(
            maximum_compressed_bytes=_integer(
                maximum_evidence_bytes,
                name="maximum_evidence_bytes",
                positive=True,
            ),
            maximum_uncompressed_bytes=_integer(
                maximum_uncompressed_bytes,
                name="maximum_uncompressed_bytes",
                positive=True,
            ),
            maximum_total_rows=_integer(
                maximum_total_rows,
                name="maximum_total_rows",
                positive=True,
            ),
            maximum_total_columns=_integer(
                maximum_total_columns,
                name="maximum_total_columns",
                positive=True,
            ),
            maximum_loaded_memory_bytes=_integer(
                maximum_loaded_memory_bytes,
                name="maximum_loaded_memory_bytes",
                positive=True,
            ),
        )

    def load_document(
        self, record: ArtifactRecord
    ) -> ResearchReadinessBundleDocumentV1:
        if type(record) is not ArtifactRecord:
            raise TypeError("readiness document reload requires an exact ArtifactRecord")
        if (
            record.logical_name != _DOCUMENT_LOGICAL_NAME
            or record.media_type != _DOCUMENT_MEDIA_TYPE
            or record.role != _DOCUMENT_ROLE
        ):
            raise _error("document_reference_mismatch", "document artifact role differs")
        if record.size_bytes > _MAXIMUM_DOCUMENT_BYTES:
            raise _error("document_budget_exceeded", "document payload is too large")
        try:
            payload = self._artifact_store.read_bytes(record)
        except (ArtifactError, OSError, TypeError, ValueError) as exc:
            raise _error("document_reload_failed", "document artifact differs") from exc
        return ResearchReadinessBundleDocumentV1.from_wire_bytes(payload)

    def load_bundle(self, record: ArtifactRecord) -> ResearchReadinessBundle:
        return self.rebuild(self.load_document(record))

    def rebuild(
        self, document: ResearchReadinessBundleDocumentV1
    ) -> ResearchReadinessBundle:
        if type(document) is not ResearchReadinessBundleDocumentV1:
            raise TypeError("readiness replay requires an exact bundle document")
        replayed_authority = self._load_input_authority(document)
        frames = self._load_evidence_set(document.evidence_artifacts)
        negative_identity = document.negative_control_identity
        stability_identity = document.parameter_stability_identity
        incremental_identity = document.incremental_baseline_identity
        spec = document.readiness_spec
        try:
            negative = NegativeControlReport(
                readiness_spec=spec,
                readiness_spec_hash=cast(
                    str, negative_identity["readiness_spec_hash"]
                ),
                candidate_score_hash=cast(
                    str, negative_identity["candidate_score_hash"]
                ),
                label_values_hash=cast(str, negative_identity["label_values_hash"]),
                label_validity_hash=cast(
                    str, negative_identity["label_validity_hash"]
                ),
                scoring_eligibility_hash=cast(
                    str, negative_identity["scoring_eligibility_hash"]
                ),
                observed_rank_ic_mean=cast(
                    float | None, negative_identity["observed_rank_ic_mean"]
                ),
                observed_valid_dates=cast(
                    int, negative_identity["observed_valid_dates"]
                ),
                pseudo_label_pvalue=cast(
                    float | None, negative_identity["pseudo_label_pvalue"]
                ),
                observed_results_hash=cast(
                    str, negative_identity["observed_results_hash"]
                ),
                null_distribution_hash=cast(
                    str, negative_identity["null_distribution_hash"]
                ),
                lag_results_hash=cast(str, negative_identity["lag_results_hash"]),
                lag_daily_results_hash=cast(
                    str, negative_identity["lag_daily_results_hash"]
                ),
                passed=cast(bool | None, negative_identity["passed"]),
                observed_results=frames["negative_observed_daily"],
                null_distribution=frames["negative_pseudo_label_trials"],
                lag_results=frames["negative_lag_summary"],
                lag_daily_results=frames["negative_lag_daily"],
            )
            stability = ParameterStabilityReport(
                readiness_spec=spec,
                readiness_spec_hash=cast(
                    str, stability_identity["readiness_spec_hash"]
                ),
                candidate_score_hash=cast(
                    str, stability_identity["candidate_score_hash"]
                ),
                label_values_hash=cast(str, stability_identity["label_values_hash"]),
                label_validity_hash=cast(
                    str, stability_identity["label_validity_hash"]
                ),
                scoring_eligibility_hash=cast(
                    str, stability_identity["scoring_eligibility_hash"]
                ),
                perturbation_score_hashes=cast(
                    Mapping[str, str],
                    stability_identity["perturbation_score_hashes"],
                ),
                scenario_results_hash=cast(
                    str, stability_identity["scenario_results_hash"]
                ),
                daily_results_hash=cast(
                    str, stability_identity["daily_results_hash"]
                ),
                passed=cast(bool | None, stability_identity["passed"]),
                scenario_results=frames["stability_scenario_summary"],
                daily_results=frames["stability_daily"],
            )
            incremental = IncrementalBaselineReport(
                readiness_spec=spec,
                readiness_spec_hash=cast(
                    str, incremental_identity["readiness_spec_hash"]
                ),
                candidate_score_hash=cast(
                    str, incremental_identity["candidate_score_hash"]
                ),
                baseline_score_hash=cast(
                    str, incremental_identity["baseline_score_hash"]
                ),
                label_values_hash=cast(
                    str, incremental_identity["label_values_hash"]
                ),
                label_validity_hash=cast(
                    str, incremental_identity["label_validity_hash"]
                ),
                scoring_eligibility_hash=cast(
                    str, incremental_identity["scoring_eligibility_hash"]
                ),
                daily_results_hash=cast(
                    str, incremental_identity["daily_results_hash"]
                ),
                candidate_rank_ic_mean=cast(
                    float | None, incremental_identity["candidate_rank_ic_mean"]
                ),
                baseline_rank_ic_mean=cast(
                    float | None, incremental_identity["baseline_rank_ic_mean"]
                ),
                incremental_rank_ic_mean=cast(
                    float | None, incremental_identity["incremental_rank_ic_mean"]
                ),
                positive_date_fraction=cast(
                    float | None, incremental_identity["positive_date_fraction"]
                ),
                valid_dates=cast(int, incremental_identity["valid_dates"]),
                passed=cast(bool | None, incremental_identity["passed"]),
                daily_results=frames["incremental_baseline_daily"],
            )
            bundle = ResearchReadinessBundle(
                readiness_spec_hash=document.readiness_spec_hash,
                candidate_evidence_hash=document.candidate_evidence_hash,
                baseline_evidence_hash=document.baseline_evidence_hash,
                outer_validation_receipt_hash=(
                    document.outer_validation_receipt_hash
                ),
                negative_control=negative,
                parameter_stability=stability,
                incremental_baseline=incremental,
                verdict=document.verdict,
                reason_codes=document.reason_codes,
            )
        except (TypeError, ValueError) as exc:
            raise _error(
                "evidence_replay_failed",
                "readiness report evidence is inconsistent",
            ) from exc
        if (
            negative.content_hash != document.negative_control_hash
            or stability.content_hash != document.parameter_stability_hash
            or incremental.content_hash != document.incremental_baseline_hash
            or bundle.content_hash != document.bundle_content_hash
        ):
            raise _error("evidence_replay_failed", "replayed bundle identity differs")
        _verify_bundle_input_lineage(
            bundle,
            replayed_authority.loaded,
            replayed_authority.authority_receipt,
        )
        return bundle

    def _load_input_authority(
        self, document: ResearchReadinessBundleDocumentV1
    ) -> _ReplayedReadinessAuthority:
        manifest_payload = self._load_input_document(
            document.input_manifest_reference
        )
        receipt_payload = self._load_input_document(
            document.authority_receipt_reference
        )
        try:
            manifest = ResearchReadinessInputManifestV1.from_wire_bytes(
                manifest_payload
            )
            receipt = ResearchReadinessAuthorityReceiptV1.from_wire_bytes(
                receipt_payload
            )
        except (ResearchReadinessInputError, TypeError, ValueError) as exc:
            raise _error(
                "input_document_reload_failed",
                "readiness input document cannot be reconstructed",
            ) from exc
        if (
            manifest.content_hash != document.input_manifest_hash
            or receipt.content_hash != document.authority_receipt_hash
            or receipt.manifest_hash != manifest.content_hash
        ):
            raise _error(
                "input_authority_mismatch",
                "readiness manifest and authority receipt differ",
            )
        loaded = self._load_manifest_frames(manifest, receipt=receipt)
        _verify_replayed_authority_binding(loaded, receipt)
        return _ReplayedReadinessAuthority(
            loaded=loaded,
            authority_receipt=receipt,
        )

    def _load_input_document(
        self, reference: ResearchReadinessInputDocumentReferenceV1
    ) -> bytes:
        if type(reference) is not ResearchReadinessInputDocumentReferenceV1:
            raise _error(
                "input_document_reference_mismatch",
                "readiness input document reference differs",
            )
        if reference.artifact.size_bytes > _MAXIMUM_DOCUMENT_BYTES:
            raise _error(
                "document_budget_exceeded", "readiness input document is too large"
            )
        try:
            payload = self._artifact_store.read_bytes(reference.artifact)
        except (ArtifactError, OSError, TypeError, ValueError) as exc:
            raise _error(
                "input_document_reload_failed",
                "readiness input document artifact differs",
            ) from exc
        return cast(bytes, payload)

    def _load_manifest_frames(
        self,
        manifest: ResearchReadinessInputManifestV1,
        *,
        receipt: ResearchReadinessAuthorityReceiptV1,
    ) -> LoadedResearchReadinessInputs:
        references: dict[str, ResearchReadinessFrameReferenceV1] = {
            "candidate_scores": manifest.score_artifacts.candidate_scores,
            "baseline_scores": manifest.score_artifacts.baseline_scores,
            **{
                f"perturbation:{name}": reference
                for name, reference in manifest.score_artifacts.perturbation_scores.items()
            },
            "labels": manifest.label_values,
            "label_validity": manifest.label_validity,
            "scoring_eligibility": manifest.scoring_eligibility,
        }
        if (
            sum(item.size_bytes for item in references.values())
            != receipt.total_compressed_bytes
            or sum(item.parquet_uncompressed_bytes for item in references.values())
            != receipt.total_parquet_uncompressed_bytes
            or receipt.total_compressed_bytes > _DEFAULT_MAXIMUM_EVIDENCE_BYTES
            or receipt.total_parquet_uncompressed_bytes
            > _DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES
            or sum(item.row_count for item in references.values())
            > _DEFAULT_MAXIMUM_TOTAL_ROWS
            or sum(item.column_count for item in references.values())
            > _DEFAULT_MAXIMUM_TOTAL_COLUMNS
        ):
            raise _error(
                "input_budget_exceeded",
                "aggregate readiness inputs exceed the replay budget",
            )
        frames: dict[str, pd.DataFrame] = {}
        loaded_memory = 0
        for name, reference in references.items():
            frame = self._load_input_frame(reference)
            loaded_memory += int(frame.memory_usage(index=True, deep=True).sum())
            if loaded_memory > _DEFAULT_MAXIMUM_LOADED_MEMORY_BYTES:
                raise _error(
                    "input_budget_exceeded",
                    "aggregate readiness input memory exceeds the replay budget",
                )
            frames[name] = frame
        if loaded_memory != receipt.total_loaded_memory_bytes:
            raise _error(
                "input_authority_mismatch",
                "aggregate readiness input memory differs from the receipt",
            )
        try:
            return LoadedResearchReadinessInputs(
                manifest=manifest,
                candidate_scores=frames["candidate_scores"],
                baseline_scores=frames["baseline_scores"],
                perturbation_scores={
                    name: frames[f"perturbation:{name}"]
                    for name in manifest.score_artifacts.perturbation_scores
                },
                labels=frames["labels"],
                label_validity=frames["label_validity"],
                scoring_eligibility=frames["scoring_eligibility"],
            )
        except (ResearchReadinessInputError, TypeError, ValueError) as exc:
            raise _error(
                "input_authority_mismatch", "decoded readiness inputs differ"
            ) from exc

    def _load_input_frame(
        self, reference: ResearchReadinessFrameReferenceV1
    ) -> pd.DataFrame:
        try:
            payload = self._artifact_store.read_bytes(
                _readiness_frame_record(reference)
            )
            parquet = pq.ParquetFile(pa.BufferReader(payload))
            if (
                _parquet_uncompressed_bytes(parquet)
                != reference.parquet_uncompressed_bytes
            ):
                raise _error(
                    "input_authority_mismatch",
                    "readiness input parquet size differs",
                )
            frame = parquet.read().to_pandas()
            reference.verify_frame(frame)
            return frame
        except ResearchReadinessBundleDocumentError:
            raise
        except (
            ArtifactError,
            MemoryError,
            OSError,
            ResearchReadinessInputError,
            TypeError,
            ValueError,
            pa.ArrowException,
        ) as exc:
            raise _error(
                "input_document_reload_failed",
                "readiness input frame artifact differs",
            ) from exc

    def _load_evidence_set(
        self,
        references: Mapping[str, ResearchReadinessEvidenceReferenceV1],
    ) -> Mapping[str, pd.DataFrame]:
        if (
            frozenset(references) != frozenset(_EVIDENCE_ROLES)
            or any(
                type(reference) is not ResearchReadinessEvidenceReferenceV1
                for reference in references.values()
            )
        ):
            raise _error("evidence_set_mismatch", "evidence artifact set differs")
        if (
            sum(reference.artifact.size_bytes for reference in references.values())
            > self._budget.maximum_compressed_bytes
            or sum(
                reference.parquet_uncompressed_bytes
                for reference in references.values()
            )
            > self._budget.maximum_uncompressed_bytes
        ):
            raise _error(
                "evidence_budget_exceeded",
                "aggregate readiness evidence exceeds its byte budget",
            )
        parquet_files: dict[str, pq.ParquetFile] = {}
        total_rows = 0
        total_columns = 0
        for kind, reference in references.items():
            try:
                payload = self._artifact_store.read_bytes(reference.artifact)
                parquet = pq.ParquetFile(pa.BufferReader(payload))
                metadata = parquet.metadata
            except (
                ArtifactError,
                OSError,
                TypeError,
                ValueError,
                pa.ArrowException,
            ) as exc:
                raise _error(
                    "evidence_reload_failed", "evidence artifact differs"
                ) from exc
            if (
                metadata is None
                or _parquet_uncompressed_bytes(parquet)
                != reference.parquet_uncompressed_bytes
            ):
                raise _error(
                    "evidence_reload_failed", "parquet size identity differs"
                )
            total_rows += metadata.num_rows
            total_columns += metadata.num_columns
            parquet_files[kind] = parquet
        if (
            total_rows > self._budget.maximum_total_rows
            or total_columns > self._budget.maximum_total_columns
        ):
            raise _error(
                "evidence_budget_exceeded",
                "aggregate readiness evidence exceeds its shape budget",
            )
        frames: dict[str, pd.DataFrame] = {}
        loaded_memory = 0
        for kind, parquet in parquet_files.items():
            try:
                frame = parquet.read().to_pandas()
            except (MemoryError, OSError, ValueError, pa.ArrowException) as exc:
                raise _error(
                    "evidence_reload_failed",
                    "evidence parquet cannot be decoded",
                ) from exc
            loaded_memory += int(frame.memory_usage(index=True, deep=True).sum())
            if loaded_memory > self._budget.maximum_loaded_memory_bytes:
                raise _error(
                    "evidence_budget_exceeded",
                    "aggregate readiness evidence exceeds its memory budget",
                )
            if hash_frame(frame) != references[kind].report_frame_hash:
                raise _error(
                    "frame_hash_mismatch", "decoded readiness evidence differs"
                )
            frames[kind] = frame
        return MappingProxyType(frames)


__all__ = [
    "ResearchReadinessBundleDocumentError",
    "ResearchReadinessBundleDocumentV1",
    "ResearchReadinessBundleLoaderV1",
    "ResearchReadinessBundleProducerV1",
    "ResearchReadinessEvidenceReferenceV1",
    "ResearchReadinessInputDocumentReferenceV1",
]
