"""Fail-closed local governance for one-shot nested outer evaluation.

This store prevents accidental or concurrent reuse of one phase-one manifest
inside a trusted local research workflow.  It is deliberately *not* an
institutional hidden-OOS security boundary: a user who controls the filesystem
also controls this store.  Every artifact therefore remains research-only and
must never claim production readiness.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)
from alpha_research.core.immutable_json import (
    ImmutableJsonError,
    load_immutable_json_document,
    write_immutable_json_document,
)
from alpha_research.models.nested_selection_manifest import (
    NestedSelectionPhaseOneManifest,
)
from alpha_research.models.model_execution_governance import (
    ModelExecutionContext,
    ModelFitAttemptOutcome,
    ModelFitAttemptSnapshot,
)
from alpha_research.models.selection_result import NestedSelectionResult


_CLAIM_SCHEMA: Final[str] = "nested-outer-evaluation-claim/v2"
_COMPLETION_SCHEMA: Final[str] = "nested-outer-evaluation-completion/v2"
_CLAIM_SUFFIX: Final[str] = ".nested-outer-evaluation.claim.json"
_RESULT_SUFFIX: Final[str] = ".nested-selection-result.json"
_EXECUTION_SNAPSHOT_SUFFIX: Final[str] = ".model-fit-attempt-snapshot.json"
_COMPLETION_SUFFIX: Final[str] = ".nested-outer-evaluation-completion.json"
_DIRECTORY_MODE: Final[int] = 0o750
_DOCUMENT_MODE: Final[int] = 0o440
_MAXIMUM_CLAIM_BYTES: Final[int] = 64 * 1024
_MAXIMUM_COMPLETION_BYTES: Final[int] = 64 * 1024
MAXIMUM_OUTER_EVALUATION_RESULT_BYTES: Final[int] = 256 * 1024 * 1024
MAXIMUM_OUTER_EVALUATION_EXECUTION_SNAPSHOT_BYTES: Final[int] = 16 * 1024 * 1024
_MAXIMUM_RESULT_BYTES: Final[int] = MAXIMUM_OUTER_EVALUATION_RESULT_BYTES
_MAXIMUM_EXECUTION_SNAPSHOT_BYTES: Final[int] = (
    MAXIMUM_OUTER_EVALUATION_EXECUTION_SNAPSHOT_BYTES
)


class NestedOuterEvaluationStoreError(RuntimeError):
    """Stable fail-closed error emitted by the local governance store."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.detail}")


def outer_evaluation_execution_context(
    claim: "OuterEvaluationClaim",
) -> ModelExecutionContext:
    """Derive the only execution context valid for one durable outer claim."""

    if not isinstance(claim, OuterEvaluationClaim):
        raise TypeError("outer evaluation execution claim type differs")
    scope_hash = cast(
        str,
        hash_json(
            {
                "schema_version": "nested-outer-evaluation-execution-scope/v1",
                "claim_hash": claim.content_hash,
                "manifest_hash": claim.manifest_hash,
                "evaluation_partition_hash": claim.evaluation_partition_hash,
                "evaluator_protocol_hash": claim.evaluator_protocol_hash,
                "runtime_fingerprint_hash": claim.runtime_fingerprint_hash,
            }
        ),
    )
    return ModelExecutionContext(
        execution_id=f"nested-outer:{claim.manifest_hash}",
        execution_scope_hash=scope_hash,
    )


@dataclass(frozen=True, slots=True)
class OuterEvaluationClaim:
    evaluation_partition_hash: str
    manifest_hash: str
    evaluator_protocol_hash: str
    runtime_fingerprint_hash: str
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _CLAIM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _CLAIM_SCHEMA:
            raise ValueError("unsupported nested outer evaluation claim schema")
        for name in (
            "evaluation_partition_hash",
            "manifest_hash",
            "evaluator_protocol_hash",
            "runtime_fingerprint_hash",
        ):
            _digest(getattr(self, name), name=f"outer evaluation claim {name}")
        _require_research_boundary(
            self.research_only,
            self.production_ready,
            name="outer evaluation claim",
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    @property
    def filename(self) -> str:
        return f"{self.evaluation_partition_hash}{_CLAIM_SUFFIX}"

    @property
    def document_sha256(self) -> str:
        return hashlib.sha256(_payload(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_partition_hash": self.evaluation_partition_hash,
            "manifest_hash": self.manifest_hash,
            "evaluator_protocol_hash": self.evaluator_protocol_hash,
            "runtime_fingerprint_hash": self.runtime_fingerprint_hash,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "OuterEvaluationClaim":
        expected = {
            "schema_version",
            "evaluation_partition_hash",
            "manifest_hash",
            "evaluator_protocol_hash",
            "runtime_fingerprint_hash",
            "research_only",
            "production_ready",
        }
        if set(value) != expected:
            raise ValueError("OuterEvaluationClaim wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            evaluation_partition_hash=_text(
                value["evaluation_partition_hash"], "evaluation_partition_hash"
            ),
            manifest_hash=_text(value["manifest_hash"], "manifest_hash"),
            evaluator_protocol_hash=_text(
                value["evaluator_protocol_hash"], "evaluator_protocol_hash"
            ),
            runtime_fingerprint_hash=_text(
                value["runtime_fingerprint_hash"], "runtime_fingerprint_hash"
            ),
            research_only=_boolean(value["research_only"], "research_only"),
            production_ready=_boolean(value["production_ready"], "production_ready"),
        )


@dataclass(frozen=True, slots=True)
class OuterEvaluationCompletion:
    evaluation_partition_hash: str
    manifest_hash: str
    claim_hash: str
    evaluator_protocol_hash: str
    runtime_fingerprint_hash: str
    result_hash: str
    result_document_sha256: str
    result_filename: str
    result_size_bytes: int
    execution_snapshot_hash: str
    execution_snapshot_document_sha256: str
    execution_snapshot_filename: str
    execution_snapshot_size_bytes: int
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _COMPLETION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _COMPLETION_SCHEMA:
            raise ValueError("unsupported nested outer evaluation completion schema")
        for name in (
            "evaluation_partition_hash",
            "manifest_hash",
            "claim_hash",
            "evaluator_protocol_hash",
            "runtime_fingerprint_hash",
            "result_hash",
            "result_document_sha256",
            "execution_snapshot_hash",
            "execution_snapshot_document_sha256",
        ):
            _digest(getattr(self, name), name=f"outer evaluation completion {name}")
        if self.result_filename != f"{self.result_hash}{_RESULT_SUFFIX}":
            raise ValueError("outer evaluation result filename differs")
        if self.execution_snapshot_filename != (
            f"{self.execution_snapshot_hash}{_EXECUTION_SNAPSHOT_SUFFIX}"
        ):
            raise ValueError("outer evaluation execution snapshot filename differs")
        if (
            not isinstance(self.result_size_bytes, int)
            or isinstance(self.result_size_bytes, bool)
            or self.result_size_bytes <= 0
            or self.result_size_bytes > _MAXIMUM_RESULT_BYTES
        ):
            raise ValueError("outer evaluation result size is invalid")
        if (
            not isinstance(self.execution_snapshot_size_bytes, int)
            or isinstance(self.execution_snapshot_size_bytes, bool)
            or self.execution_snapshot_size_bytes <= 0
            or self.execution_snapshot_size_bytes > _MAXIMUM_EXECUTION_SNAPSHOT_BYTES
        ):
            raise ValueError("outer evaluation execution snapshot size is invalid")
        _require_research_boundary(
            self.research_only,
            self.production_ready,
            name="outer evaluation completion",
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    @property
    def document_sha256(self) -> str:
        return hashlib.sha256(_payload(self.to_dict())).hexdigest()

    @property
    def filename(self) -> str:
        return (
            f"{self.evaluation_partition_hash}."
            f"{self.document_sha256}{_COMPLETION_SUFFIX}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_partition_hash": self.evaluation_partition_hash,
            "manifest_hash": self.manifest_hash,
            "claim_hash": self.claim_hash,
            "evaluator_protocol_hash": self.evaluator_protocol_hash,
            "runtime_fingerprint_hash": self.runtime_fingerprint_hash,
            "result_hash": self.result_hash,
            "result_document_sha256": self.result_document_sha256,
            "result_filename": self.result_filename,
            "result_size_bytes": self.result_size_bytes,
            "execution_snapshot_hash": self.execution_snapshot_hash,
            "execution_snapshot_document_sha256": (
                self.execution_snapshot_document_sha256
            ),
            "execution_snapshot_filename": self.execution_snapshot_filename,
            "execution_snapshot_size_bytes": self.execution_snapshot_size_bytes,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "OuterEvaluationCompletion":
        expected = {
            "schema_version",
            "evaluation_partition_hash",
            "manifest_hash",
            "claim_hash",
            "evaluator_protocol_hash",
            "runtime_fingerprint_hash",
            "result_hash",
            "result_document_sha256",
            "result_filename",
            "result_size_bytes",
            "execution_snapshot_hash",
            "execution_snapshot_document_sha256",
            "execution_snapshot_filename",
            "execution_snapshot_size_bytes",
            "research_only",
            "production_ready",
        }
        if set(value) != expected:
            raise ValueError("OuterEvaluationCompletion wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            evaluation_partition_hash=_text(
                value["evaluation_partition_hash"], "evaluation_partition_hash"
            ),
            manifest_hash=_text(value["manifest_hash"], "manifest_hash"),
            claim_hash=_text(value["claim_hash"], "claim_hash"),
            evaluator_protocol_hash=_text(
                value["evaluator_protocol_hash"], "evaluator_protocol_hash"
            ),
            runtime_fingerprint_hash=_text(
                value["runtime_fingerprint_hash"], "runtime_fingerprint_hash"
            ),
            result_hash=_text(value["result_hash"], "result_hash"),
            result_document_sha256=_text(
                value["result_document_sha256"], "result_document_sha256"
            ),
            result_filename=_text(value["result_filename"], "result_filename"),
            result_size_bytes=_integer(value["result_size_bytes"], "result_size_bytes"),
            execution_snapshot_hash=_text(
                value["execution_snapshot_hash"], "execution_snapshot_hash"
            ),
            execution_snapshot_document_sha256=_text(
                value["execution_snapshot_document_sha256"],
                "execution_snapshot_document_sha256",
            ),
            execution_snapshot_filename=_text(
                value["execution_snapshot_filename"], "execution_snapshot_filename"
            ),
            execution_snapshot_size_bytes=_integer(
                value["execution_snapshot_size_bytes"],
                "execution_snapshot_size_bytes",
            ),
            research_only=_boolean(value["research_only"], "research_only"),
            production_ready=_boolean(value["production_ready"], "production_ready"),
        )


@dataclass(frozen=True, slots=True)
class CompletedOuterEvaluation:
    completion: OuterEvaluationCompletion
    result: NestedSelectionResult
    execution_snapshot: ModelFitAttemptSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.completion, OuterEvaluationCompletion):
            raise TypeError("completed outer evaluation completion type differs")
        if not isinstance(self.result, NestedSelectionResult):
            raise TypeError("completed outer evaluation result type differs")
        if not isinstance(self.execution_snapshot, ModelFitAttemptSnapshot):
            raise TypeError(
                "completed outer evaluation execution snapshot type differs"
            )
        if self.completion.result_hash != self.result.content_hash:
            raise ValueError("completed outer evaluation result hash differs")
        if (
            self.completion.execution_snapshot_hash
            != self.execution_snapshot.content_hash
        ):
            raise ValueError(
                "completed outer evaluation execution snapshot hash differs"
            )
        _require_research_boundary(
            self.result.research_only,
            self.result.production_ready,
            name="completed outer evaluation result",
        )


class NestedOuterEvaluationStore:
    """Local one-shot claim and immutable completion store.

    A newly returned :class:`OuterEvaluationClaim` authorizes one local caller
    to start outer evaluation.  When the same manifest already has a fully
    verified completion, :class:`CompletedOuterEvaluation` is returned without
    authorizing another execution.  A claim without a completion is terminally
    uncertain and never auto-retried.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = _absolute(root)
        self._claims = self._root / "claims"
        self._results = self._root / "results"
        self._execution_snapshots = self._root / "execution_snapshots"
        self._completions = self._root / "completions"
        _ensure_store_directory(self._root)
        _ensure_store_directory(self._claims)
        _ensure_store_directory(self._results)
        _ensure_store_directory(self._execution_snapshots)
        _ensure_store_directory(self._completions)

    @property
    def root(self) -> Path:
        return self._root

    def claim_or_load(
        self,
        manifest: NestedSelectionPhaseOneManifest,
        *,
        evaluator_protocol_hash: str,
        runtime_fingerprint_hash: str,
        maximum_result_bytes: int = _MAXIMUM_RESULT_BYTES,
        maximum_execution_snapshot_bytes: int = (
            _MAXIMUM_EXECUTION_SNAPSHOT_BYTES
        ),
    ) -> OuterEvaluationClaim | CompletedOuterEvaluation:
        result_limit, execution_snapshot_limit = _validated_read_budgets(
            maximum_result_bytes=maximum_result_bytes,
            maximum_execution_snapshot_bytes=maximum_execution_snapshot_bytes,
        )
        _require_manifest(manifest)
        expected = OuterEvaluationClaim(
            evaluation_partition_hash=manifest.outer_validation_receipt_hash,
            manifest_hash=manifest.content_hash,
            evaluator_protocol_hash=evaluator_protocol_hash,
            runtime_fingerprint_hash=runtime_fingerprint_hash,
        )
        claim_path = self._claims / expected.filename

        # A completion without its authoritative claim is inconsistent.  Check
        # before creating anything so a damaged store cannot heal itself.
        with _directory_lock(self._completions, exclusive=False):
            existing_completion = _find_completion_path(
                self._completions,
                evaluation_partition_hash=manifest.outer_validation_receipt_hash,
            )
        if existing_completion is not None and not claim_path.exists():
            raise NestedOuterEvaluationStoreError(
                "outer_completion_without_claim", manifest.content_hash
            )

        try:
            _write_claim_exclusive(claim_path, expected)
        except FileExistsError:
            observed = self._load_expected_claim(claim_path, expected)
            with _directory_lock(self._completions, exclusive=False):
                completion_path = _find_completion_path(
                    self._completions,
                    evaluation_partition_hash=manifest.outer_validation_receipt_hash,
                )
                if completion_path is None:
                    raise NestedOuterEvaluationStoreError(
                        "outer_evaluation_uncertain", manifest.content_hash
                    ) from None
                return self._load_completed(
                    manifest=manifest,
                    claim=observed,
                    completion_path=completion_path,
                    maximum_result_bytes=result_limit,
                    maximum_execution_snapshot_bytes=execution_snapshot_limit,
                )

        # The exclusive create proves this caller is the first claimant.  A
        # pre-existing completion at this point would indicate store damage or
        # an unauthorized writer and must not be treated as a cache hit.
        with _directory_lock(self._completions, exclusive=False):
            completion_path = _find_completion_path(
                self._completions,
                evaluation_partition_hash=manifest.outer_validation_receipt_hash,
            )
        if completion_path is not None:
            raise NestedOuterEvaluationStoreError(
                "outer_completion_precedes_claim", manifest.content_hash
            )
        return expected

    def load_completed(
        self,
        manifest: NestedSelectionPhaseOneManifest,
        *,
        evaluator_protocol_hash: str,
        runtime_fingerprint_hash: str,
        maximum_result_bytes: int = _MAXIMUM_RESULT_BYTES,
        maximum_execution_snapshot_bytes: int = (
            _MAXIMUM_EXECUTION_SNAPSHOT_BYTES
        ),
    ) -> CompletedOuterEvaluation:
        """Load a completion without creating or changing a claim.

        This audit-only path fails closed when either side of the durable
        claim/completion pair is absent.  Reading can therefore never consume
        an outer partition or turn interrupted state into a retry.
        """

        result_limit, execution_snapshot_limit = _validated_read_budgets(
            maximum_result_bytes=maximum_result_bytes,
            maximum_execution_snapshot_bytes=maximum_execution_snapshot_bytes,
        )
        _require_manifest(manifest)
        expected = OuterEvaluationClaim(
            evaluation_partition_hash=manifest.outer_validation_receipt_hash,
            manifest_hash=manifest.content_hash,
            evaluator_protocol_hash=evaluator_protocol_hash,
            runtime_fingerprint_hash=runtime_fingerprint_hash,
        )
        claim_path = self._claims / expected.filename
        if not claim_path.exists():
            raise NestedOuterEvaluationStoreError(
                "outer_claim_missing", manifest.content_hash
            )
        observed = self._load_expected_claim(claim_path, expected)
        with _directory_lock(self._completions, exclusive=False):
            completion_path = _find_completion_path(
                self._completions,
                evaluation_partition_hash=manifest.outer_validation_receipt_hash,
            )
            if completion_path is None:
                raise NestedOuterEvaluationStoreError(
                    "outer_completion_missing", manifest.content_hash
                )
            return self._load_completed(
                manifest=manifest,
                claim=observed,
                completion_path=completion_path,
                maximum_result_bytes=result_limit,
                maximum_execution_snapshot_bytes=execution_snapshot_limit,
            )

    def publish_completed(
        self,
        claim: OuterEvaluationClaim,
        *,
        manifest: NestedSelectionPhaseOneManifest,
        result: NestedSelectionResult,
        execution_snapshot: ModelFitAttemptSnapshot,
    ) -> CompletedOuterEvaluation:
        if not isinstance(claim, OuterEvaluationClaim):
            raise TypeError("outer evaluation claim type differs")
        _require_manifest(manifest)
        if not isinstance(result, NestedSelectionResult):
            raise TypeError("nested selection result type differs")
        if (
            claim.manifest_hash != manifest.content_hash
            or claim.evaluation_partition_hash != manifest.outer_validation_receipt_hash
        ):
            raise ValueError("outer evaluation claim manifest or partition differs")
        _require_research_boundary(
            result.research_only,
            result.production_ready,
            name="nested selection result",
        )
        _verify_result_manifest_lineage(result, manifest=manifest)
        _verify_execution_snapshot(
            execution_snapshot,
            manifest=manifest,
            claim=claim,
            result=result,
        )
        self._load_expected_claim(self._claims / claim.filename, claim)

        result_wire = result.to_dict()
        result_payload = _payload(result_wire)
        if len(result_payload) > _MAXIMUM_RESULT_BYTES:
            raise ValueError("nested selection result exceeds size limit")
        result_document_sha256 = hashlib.sha256(result_payload).hexdigest()
        result_filename = f"{result.content_hash}{_RESULT_SUFFIX}"
        execution_snapshot_wire = execution_snapshot.to_dict()
        execution_snapshot_payload = _payload(execution_snapshot_wire)
        if len(execution_snapshot_payload) > _MAXIMUM_EXECUTION_SNAPSHOT_BYTES:
            raise ValueError("model fit attempt snapshot exceeds size limit")
        execution_snapshot_document_sha256 = hashlib.sha256(
            execution_snapshot_payload
        ).hexdigest()
        execution_snapshot_filename = (
            f"{execution_snapshot.content_hash}{_EXECUTION_SNAPSHOT_SUFFIX}"
        )
        completion = OuterEvaluationCompletion(
            evaluation_partition_hash=claim.evaluation_partition_hash,
            manifest_hash=manifest.content_hash,
            claim_hash=claim.content_hash,
            evaluator_protocol_hash=claim.evaluator_protocol_hash,
            runtime_fingerprint_hash=claim.runtime_fingerprint_hash,
            result_hash=result.content_hash,
            result_document_sha256=result_document_sha256,
            result_filename=result_filename,
            result_size_bytes=len(result_payload),
            execution_snapshot_hash=execution_snapshot.content_hash,
            execution_snapshot_document_sha256=execution_snapshot_document_sha256,
            execution_snapshot_filename=execution_snapshot_filename,
            execution_snapshot_size_bytes=len(execution_snapshot_payload),
        )

        with _directory_lock(self._completions, exclusive=True):
            existing_path = _find_completion_path(
                self._completions,
                evaluation_partition_hash=claim.evaluation_partition_hash,
            )
            if existing_path is not None:
                loaded = self._load_completed(
                    manifest=manifest,
                    claim=claim,
                    completion_path=existing_path,
                )
                if loaded.completion != completion:
                    raise NestedOuterEvaluationStoreError(
                        "outer_completion_conflict", manifest.content_hash
                    )
                return loaded

            try:
                write_immutable_json_document(
                    self._results,
                    filename=result_filename,
                    value=result_wire,
                    mode=_DOCUMENT_MODE,
                )
                write_immutable_json_document(
                    self._execution_snapshots,
                    filename=execution_snapshot_filename,
                    value=execution_snapshot_wire,
                    mode=_DOCUMENT_MODE,
                )
                write_immutable_json_document(
                    self._completions,
                    filename=completion.filename,
                    value=completion.to_dict(),
                    mode=_DOCUMENT_MODE,
                )
            except (ImmutableJsonError, OSError, ValueError) as error:
                raise NestedOuterEvaluationStoreError(
                    "outer_completion_publish_failed", manifest.content_hash
                ) from error
            return self._load_completed(
                manifest=manifest,
                claim=claim,
                completion_path=self._completions / completion.filename,
            )

    def _load_expected_claim(
        self, path: Path, expected: OuterEvaluationClaim
    ) -> OuterEvaluationClaim:
        try:
            value = load_immutable_json_document(
                path,
                expected_filename=expected.filename,
                expected_document_sha256=expected.document_sha256,
                mode=_DOCUMENT_MODE,
                maximum_bytes=_MAXIMUM_CLAIM_BYTES,
            )
            observed = OuterEvaluationClaim.from_mapping(value)
        except (ImmutableJsonError, OSError, TypeError, ValueError) as error:
            raise NestedOuterEvaluationStoreError(
                "outer_claim_invalid", expected.manifest_hash
            ) from error
        if observed != expected or observed.content_hash != expected.content_hash:
            raise NestedOuterEvaluationStoreError(
                "outer_claim_drift", expected.manifest_hash
            )
        return observed

    def _load_completed(
        self,
        *,
        manifest: NestedSelectionPhaseOneManifest,
        claim: OuterEvaluationClaim,
        completion_path: Path,
        maximum_result_bytes: int = _MAXIMUM_RESULT_BYTES,
        maximum_execution_snapshot_bytes: int = (
            _MAXIMUM_EXECUTION_SNAPSHOT_BYTES
        ),
    ) -> CompletedOuterEvaluation:
        result_limit, execution_snapshot_limit = _validated_read_budgets(
            maximum_result_bytes=maximum_result_bytes,
            maximum_execution_snapshot_bytes=maximum_execution_snapshot_bytes,
        )
        document_sha = _completion_document_sha_from_filename(
            completion_path.name,
            evaluation_partition_hash=manifest.outer_validation_receipt_hash,
        )
        try:
            completion_wire = load_immutable_json_document(
                completion_path,
                expected_filename=completion_path.name,
                expected_document_sha256=document_sha,
                mode=_DOCUMENT_MODE,
                maximum_bytes=_MAXIMUM_COMPLETION_BYTES,
            )
            completion = OuterEvaluationCompletion.from_mapping(completion_wire)
        except (ImmutableJsonError, OSError, TypeError, ValueError) as error:
            raise NestedOuterEvaluationStoreError(
                "outer_completion_invalid", manifest.content_hash
            ) from error
        if (
            completion.filename != completion_path.name
            or completion.evaluation_partition_hash
            != manifest.outer_validation_receipt_hash
            or completion.manifest_hash != manifest.content_hash
            or completion.claim_hash != claim.content_hash
            or completion.evaluator_protocol_hash != claim.evaluator_protocol_hash
            or completion.runtime_fingerprint_hash != claim.runtime_fingerprint_hash
        ):
            raise NestedOuterEvaluationStoreError(
                "outer_completion_lineage_drift", manifest.content_hash
            )
        # Completion metadata is deliberately small and loaded first.  Its
        # declared sizes are the fail-closed admission check for the two much
        # larger documents: an over-budget document must never be opened or
        # decoded merely to discover that it exceeds the caller's budget.
        if completion.result_size_bytes > result_limit:
            raise NestedOuterEvaluationStoreError(
                "outer_result_budget_exceeded", manifest.content_hash
            )
        if completion.execution_snapshot_size_bytes > execution_snapshot_limit:
            raise NestedOuterEvaluationStoreError(
                "outer_execution_snapshot_budget_exceeded", manifest.content_hash
            )
        result_path = self._results / completion.result_filename
        try:
            result_wire = load_immutable_json_document(
                result_path,
                expected_filename=completion.result_filename,
                expected_document_sha256=completion.result_document_sha256,
                mode=_DOCUMENT_MODE,
                maximum_bytes=min(result_limit, _MAXIMUM_RESULT_BYTES),
            )
            if len(_payload(result_wire)) != completion.result_size_bytes:
                raise ValueError("nested selection result size differs")
            result = NestedSelectionResult.from_mapping(result_wire)
        except (ImmutableJsonError, OSError, TypeError, ValueError) as error:
            raise NestedOuterEvaluationStoreError(
                "outer_result_invalid", manifest.content_hash
            ) from error
        if result.content_hash != completion.result_hash:
            raise NestedOuterEvaluationStoreError(
                "outer_result_hash_drift", manifest.content_hash
            )
        try:
            _verify_result_manifest_lineage(result, manifest=manifest)
        except ValueError as error:
            raise NestedOuterEvaluationStoreError(
                "outer_result_lineage_drift", manifest.content_hash
            ) from error
        execution_snapshot_path = (
            self._execution_snapshots / completion.execution_snapshot_filename
        )
        try:
            execution_snapshot_wire = load_immutable_json_document(
                execution_snapshot_path,
                expected_filename=completion.execution_snapshot_filename,
                expected_document_sha256=(
                    completion.execution_snapshot_document_sha256
                ),
                mode=_DOCUMENT_MODE,
                maximum_bytes=min(
                    execution_snapshot_limit,
                    _MAXIMUM_EXECUTION_SNAPSHOT_BYTES,
                ),
            )
            if (
                len(_payload(execution_snapshot_wire))
                != completion.execution_snapshot_size_bytes
            ):
                raise ValueError("model fit attempt snapshot size differs")
            execution_snapshot = ModelFitAttemptSnapshot.from_mapping(
                execution_snapshot_wire
            )
        except (ImmutableJsonError, OSError, TypeError, ValueError) as error:
            raise NestedOuterEvaluationStoreError(
                "outer_execution_snapshot_invalid", manifest.content_hash
            ) from error
        if execution_snapshot.content_hash != completion.execution_snapshot_hash:
            raise NestedOuterEvaluationStoreError(
                "outer_execution_snapshot_hash_drift", manifest.content_hash
            )
        try:
            _verify_execution_snapshot(
                execution_snapshot,
                manifest=manifest,
                claim=claim,
                result=result,
            )
        except ValueError as error:
            raise NestedOuterEvaluationStoreError(
                "outer_execution_snapshot_lineage_drift", manifest.content_hash
            ) from error
        return CompletedOuterEvaluation(
            completion=completion,
            result=result,
            execution_snapshot=execution_snapshot,
        )


def _require_manifest(manifest: object) -> NestedSelectionPhaseOneManifest:
    if not isinstance(manifest, NestedSelectionPhaseOneManifest):
        raise TypeError("phase-one manifest type differs")
    _require_research_boundary(
        manifest.research_only,
        manifest.production_ready,
        name="phase-one manifest",
    )
    return manifest


def _verify_result_manifest_lineage(
    result: NestedSelectionResult,
    *,
    manifest: NestedSelectionPhaseOneManifest,
) -> None:
    if (
        result.selection_spec_hash != manifest.selection_spec_hash
        or result.selection_spec != manifest.selection_spec
        or result.outer_validation_spec_hash != manifest.outer_validation_spec_hash
        or result.outer_validation_spec != manifest.outer_validation_spec
        or result.outer_validation_receipt_hash
        != manifest.outer_validation_receipt_hash
        or result.outer_validation_receipt != manifest.outer_validation_receipt
        or dict(result.source_feature_hashes) != dict(manifest.source_feature_hashes)
        or result.source_label_values_hash != manifest.source_label_values_hash
        or result.source_label_validity_hash != manifest.source_label_validity_hash
        or result.source_scoring_eligibility_hash
        != manifest.source_scoring_eligibility_hash
        or result.selections != manifest.selections
    ):
        raise ValueError("nested selection result differs from phase-one manifest")


def _verify_execution_snapshot(
    snapshot: ModelFitAttemptSnapshot,
    *,
    manifest: NestedSelectionPhaseOneManifest,
    claim: OuterEvaluationClaim,
    result: NestedSelectionResult,
) -> None:
    if not isinstance(snapshot, ModelFitAttemptSnapshot):
        raise TypeError("outer evaluation execution snapshot type differs")
    expected_context = outer_evaluation_execution_context(claim)
    if (
        snapshot.execution_context != expected_context
        or snapshot.execution_context_hash != expected_context.content_hash
    ):
        raise ValueError("outer evaluation execution context differs")
    expected_attempts = manifest.planned_outer_fold_evaluations
    evaluations = tuple(result.outer_evaluations)
    if len(evaluations) != expected_attempts:
        raise ValueError("outer evaluation result fold count differs")
    if (
        snapshot.maximum_attempts != expected_attempts
        or snapshot.consumed_attempt_count != expected_attempts
        or snapshot.terminal_attempt_count != expected_attempts
        or snapshot.open_attempt_count != 0
        or snapshot.remaining_attempt_count != 0
    ):
        raise ValueError("outer evaluation execution budget evidence differs")
    if any(
        terminal.outcome is not ModelFitAttemptOutcome.SUCCEEDED
        or terminal.failure_type is not None
        for terminal in snapshot.terminals
    ):
        raise ValueError("outer evaluation execution contains failed attempts")
    for attempt, evaluation in zip(snapshot.attempts, evaluations, strict=True):
        if (
            attempt.fold_attempt_ordinal != 1
            or attempt.fold_id != evaluation.outer_fold_id
            or attempt.model_spec_hash != evaluation.materialized_model_spec_hash
            or attempt.validation_receipt_hash
            != evaluation.projected_outer_validation_receipt_hash
        ):
            raise ValueError("outer evaluation execution attempt lineage differs")


def _find_completion_path(root: Path, *, evaluation_partition_hash: str) -> Path | None:
    _digest(
        evaluation_partition_hash,
        name="completion lookup evaluation_partition_hash",
    )
    prefix = f"{evaluation_partition_hash}."
    matches = tuple(
        sorted(
            (
                entry
                for entry in root.iterdir()
                if entry.name.startswith(prefix)
                and entry.name.endswith(_COMPLETION_SUFFIX)
            ),
            key=lambda item: item.name,
        )
    )
    if len(matches) > 1:
        raise NestedOuterEvaluationStoreError(
            "multiple_outer_completions", evaluation_partition_hash
        )
    return None if not matches else matches[0]


def _completion_document_sha_from_filename(
    filename: str, *, evaluation_partition_hash: str
) -> str:
    prefix = f"{evaluation_partition_hash}."
    if not filename.startswith(prefix) or not filename.endswith(_COMPLETION_SUFFIX):
        raise NestedOuterEvaluationStoreError(
            "outer_completion_filename_invalid", evaluation_partition_hash
        )
    digest = filename[len(prefix) : -len(_COMPLETION_SUFFIX)]
    try:
        return _digest(digest, name="completion filename document sha256")
    except (TypeError, ValueError) as error:
        raise NestedOuterEvaluationStoreError(
            "outer_completion_filename_invalid", evaluation_partition_hash
        ) from error


def _write_claim_exclusive(path: Path, claim: OuterEvaluationClaim) -> None:
    payload = _payload(claim.to_dict())
    if len(payload) > _MAXIMUM_CLAIM_BYTES:
        raise ValueError("outer evaluation claim exceeds size limit")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, _DOCUMENT_MODE)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short outer evaluation claim write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, _DOCUMENT_MODE)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


@contextmanager
def _directory_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    _validate_store_directory(path)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _ensure_store_directory(path: Path) -> None:
    _reject_symlink_ancestors(path)
    try:
        path.mkdir(mode=_DIRECTORY_MODE)
    except FileExistsError:
        pass
    except FileNotFoundError:
        path.mkdir(parents=True, mode=_DIRECTORY_MODE)
    _validate_store_directory(path)


def _validate_store_directory(path: Path) -> None:
    _reject_symlink_ancestors(path)
    try:
        observed = path.lstat()
    except OSError as error:
        raise NestedOuterEvaluationStoreError(
            "outer_store_directory_invalid", str(path)
        ) from error
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != _DIRECTORY_MODE
    ):
        raise NestedOuterEvaluationStoreError(
            "outer_store_directory_invalid", str(path)
        )


def _reject_symlink_ancestors(path: Path) -> None:
    current = path
    while True:
        try:
            observed = current.lstat()
        except FileNotFoundError:
            observed = None
        except OSError as error:
            raise NestedOuterEvaluationStoreError(
                "outer_store_path_invalid", str(current)
            ) from error
        if observed is not None and stat.S_ISLNK(observed.st_mode):
            raise NestedOuterEvaluationStoreError(
                "outer_store_symlink_forbidden", str(current)
            )
        if current.parent == current:
            return
        current = current.parent


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _payload(value: Mapping[str, object]) -> bytes:
    return cast(bytes, canonical_json_bytes(dict(value)) + b"\n")


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return cast(str, require_sha256(value, name=name))


def _require_research_boundary(
    research_only: object, production_ready: object, *, name: str
) -> None:
    if (
        not isinstance(research_only, bool)
        or not research_only
        or not isinstance(production_ready, bool)
        or production_ready
    ):
        raise ValueError(f"{name} assurance boundary differs")


def _validated_read_budgets(
    *,
    maximum_result_bytes: object,
    maximum_execution_snapshot_bytes: object,
) -> tuple[int, int]:
    return (
        _bounded_positive_integer(
            maximum_result_bytes,
            name="maximum_result_bytes",
            global_maximum=_MAXIMUM_RESULT_BYTES,
        ),
        _bounded_positive_integer(
            maximum_execution_snapshot_bytes,
            name="maximum_execution_snapshot_bytes",
            global_maximum=_MAXIMUM_EXECUTION_SNAPSHOT_BYTES,
        ),
    )


def _bounded_positive_integer(
    value: object, *, name: str, global_maximum: int
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > global_maximum:
        raise ValueError(f"{name} exceeds global maximum")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


__all__ = [
    "CompletedOuterEvaluation",
    "MAXIMUM_OUTER_EVALUATION_EXECUTION_SNAPSHOT_BYTES",
    "MAXIMUM_OUTER_EVALUATION_RESULT_BYTES",
    "NestedOuterEvaluationStore",
    "NestedOuterEvaluationStoreError",
    "OuterEvaluationClaim",
    "OuterEvaluationCompletion",
    "outer_evaluation_execution_context",
]
