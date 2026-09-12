"""Authority-separated control surfaces for nested model selection.

The scientific selector intentionally keeps manual convenience APIs for local
research.  Those APIs must not be injected into an automated Agent.  This
module provides two narrow facets created by one trusted composition root:

* the Agent facet can only execute and publish phase-one (inner) selection;
* the human/audit facet owns the fixed global outer store and is the only
  surface that can execute or read outer evaluation.

This is trustworthy local workflow governance, not an institutional security
boundary.  A filesystem owner can still create another authority root, so all
outputs remain ``research_only=true`` and ``production_ready=false``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast, final

import pandas as pd

from alpha_research.core.frequency import TradingCalendar
from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.labels import LabelResult
from alpha_research.models.nested_outer_evaluation_store import (
    CompletedOuterEvaluation,
    MAXIMUM_OUTER_EVALUATION_EXECUTION_SNAPSHOT_BYTES,
    MAXIMUM_OUTER_EVALUATION_RESULT_BYTES,
    NestedOuterEvaluationStore,
    OuterEvaluationCompletion,
)
from alpha_research.models.nested_selection import (
    NestedPurgedModelSelector,
    verify_nested_selection_phase_one_manifest_inputs,
)
from alpha_research.models.nested_selection_manifest import (
    NestedSelectionPhaseOneManifest,
    NestedSelectionPhaseOneManifestReference,
    load_nested_selection_phase_one_manifest,
    publish_nested_selection_phase_one_manifest,
)
from alpha_research.models.selection_spec import NestedPurgedSelectionSpec
from alpha_research.validation import ValidationReceipt, ValidationSpec


@dataclass(frozen=True, slots=True)
class VerifiedNestedSelectionPhaseOne:
    """Reloaded phase-one document bound to one operator authority."""

    authority_hash: str
    reference: NestedSelectionPhaseOneManifestReference
    manifest: NestedSelectionPhaseOneManifest

    def __post_init__(self) -> None:
        require_sha256(self.authority_hash, name="nested selection authority hash")
        if not isinstance(self.reference, NestedSelectionPhaseOneManifestReference):
            raise TypeError("phase-one manifest reference type differs")
        if not isinstance(self.manifest, NestedSelectionPhaseOneManifest):
            raise TypeError("phase-one manifest type differs")
        if self.manifest.content_hash != self.reference.manifest_id:
            raise ValueError("phase-one verified manifest identity differs")

    @property
    def content_hash(self) -> str:
        return cast(
            str,
            hash_json(
                {
                    "schema_version": "verified-nested-selection-phase-one/v1",
                    "authority_hash": self.authority_hash,
                    "reference": self.reference.to_dict(),
                    "manifest_hash": self.manifest.content_hash,
                }
            ),
        )


class AgentNestedSelectionController:
    """Agent-facing facet: phase-one selection is its only public operation."""

    _authority_hash: str
    _manifest_root: Path
    _selector: NestedPurgedModelSelector

    __slots__ = ("_authority_hash", "_manifest_root", "_selector")

    def __init__(
        self,
        *,
        authority_hash: str,
        manifest_root: Path,
        selector: NestedPurgedModelSelector,
    ) -> None:
        if type(selector) is not NestedPurgedModelSelector:
            raise TypeError("Agent selector must be an exact NestedPurgedModelSelector")
        object.__setattr__(
            self,
            "_authority_hash",
            cast(
                str,
                require_sha256(authority_hash, name="nested selection authority hash"),
            ),
        )
        object.__setattr__(self, "_manifest_root", manifest_root)
        object.__setattr__(self, "_selector", selector)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AgentNestedSelectionController is immutable")

    @property
    def authority_hash(self) -> str:
        """Opaque binding to the trusted composition root."""

        return self._authority_hash

    def select_inner(
        self,
        spec: NestedPurgedSelectionSpec,
        *,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        outer_validation_spec: ValidationSpec,
        outer_validation_receipt: ValidationReceipt,
        calendar: TradingCalendar,
        scoring_eligibility: pd.DataFrame | None = None,
    ) -> NestedSelectionPhaseOneManifestReference:
        """Execute inner selection and return only its immutable reference."""

        manifest = NestedPurgedModelSelector.select_inner_phase(
            self._selector,
            spec,
            feature_signals=feature_signals,
            labels=labels,
            outer_validation_spec=outer_validation_spec,
            outer_validation_receipt=outer_validation_receipt,
            calendar=calendar,
            scoring_eligibility=scoring_eligibility,
        )
        return publish_nested_selection_phase_one_manifest(
            manifest,
            self._manifest_root,
        )

    def _select_inner_verified(
        self,
        spec: NestedPurgedSelectionSpec,
        *,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        outer_validation_spec: ValidationSpec,
        outer_validation_receipt: ValidationReceipt,
        calendar: TradingCalendar,
        scoring_eligibility: pd.DataFrame | None = None,
    ) -> VerifiedNestedSelectionPhaseOne:
        """Internal composition hook: execute, reload and verify phase one."""

        reference = AgentNestedSelectionController.select_inner(
            self,
            spec,
            feature_signals=feature_signals,
            labels=labels,
            outer_validation_spec=outer_validation_spec,
            outer_validation_receipt=outer_validation_receipt,
            calendar=calendar,
            scoring_eligibility=scoring_eligibility,
        )
        manifest = AgentNestedSelectionController._reload_phase_one_manifest(
            self,
            reference,
        )
        verify_nested_selection_phase_one_manifest_inputs(
            manifest,
            spec=spec,
            feature_signals=feature_signals,
            labels=labels,
            outer_validation_spec=outer_validation_spec,
            outer_validation_receipt=outer_validation_receipt,
            calendar=calendar,
            scoring_eligibility=scoring_eligibility,
        )
        return VerifiedNestedSelectionPhaseOne(
            authority_hash=self._authority_hash,
            reference=reference,
            manifest=manifest,
        )

    def _reload_phase_one_manifest(
        self,
        reference: NestedSelectionPhaseOneManifestReference,
    ) -> NestedSelectionPhaseOneManifest:
        """Internal immutable reload bound to this controller's fixed root."""

        return load_nested_selection_phase_one_manifest(
            self._manifest_root / reference.filename,
            reference,
        )


@final
class NestedOuterEvaluationAuditReader:
    """Read-only facet for an already-completed outer evaluation.

    The reader is intentionally separate from the human execution controller:
    consumers that only need audited outer evidence receive no capability to
    claim an outer partition, execute an evaluator, or publish a completion.
    Its fixed authority inputs are snapshotted at construction and the object
    cannot be rebound through ordinary attribute assignment.
    """

    _authority_hash: str
    _evaluator_protocol_hash: str
    _manifest_root: Path
    _outer_store: NestedOuterEvaluationStore
    _runtime_fingerprint_hash: str

    __slots__ = (
        "_authority_hash",
        "_evaluator_protocol_hash",
        "_manifest_root",
        "_outer_store",
        "_runtime_fingerprint_hash",
    )

    def __init__(
        self,
        *,
        authority_hash: str,
        manifest_root: Path,
        outer_store: NestedOuterEvaluationStore,
        evaluator_protocol_hash: str,
        runtime_fingerprint_hash: str,
    ) -> None:
        if type(outer_store) is not NestedOuterEvaluationStore:
            raise TypeError(
                "audit outer store must be an exact NestedOuterEvaluationStore"
            )
        object.__setattr__(
            self,
            "_authority_hash",
            cast(
                str,
                require_sha256(authority_hash, name="nested selection authority hash"),
            ),
        )
        object.__setattr__(self, "_manifest_root", manifest_root)
        object.__setattr__(self, "_outer_store", outer_store)
        object.__setattr__(
            self,
            "_evaluator_protocol_hash",
            cast(
                str,
                require_sha256(
                    evaluator_protocol_hash,
                    name="nested selection evaluator protocol hash",
                ),
            ),
        )
        object.__setattr__(
            self,
            "_runtime_fingerprint_hash",
            cast(
                str,
                require_sha256(
                    runtime_fingerprint_hash,
                    name="nested selection runtime fingerprint hash",
                ),
            ),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("NestedOuterEvaluationAuditReader is immutable")

    @property
    def authority_hash(self) -> str:
        """Opaque binding shared with both nested-selection controllers."""

        return self._authority_hash

    def read_completed(
        self,
        manifest_reference: NestedSelectionPhaseOneManifestReference,
        *,
        maximum_result_bytes: int = MAXIMUM_OUTER_EVALUATION_RESULT_BYTES,
        maximum_execution_snapshot_bytes: int = (
            MAXIMUM_OUTER_EVALUATION_EXECUTION_SNAPSHOT_BYTES
        ),
    ) -> CompletedOuterEvaluation:
        """Load and fully verify existing evidence without mutating the store."""

        if type(manifest_reference) is not NestedSelectionPhaseOneManifestReference:
            raise TypeError("phase-one manifest reference type differs")
        manifest = load_nested_selection_phase_one_manifest(
            self._manifest_root / manifest_reference.filename,
            manifest_reference,
        )
        return NestedOuterEvaluationStore.load_completed(
            self._outer_store,
            manifest,
            evaluator_protocol_hash=self._evaluator_protocol_hash,
            runtime_fingerprint_hash=self._runtime_fingerprint_hash,
            maximum_result_bytes=maximum_result_bytes,
            maximum_execution_snapshot_bytes=maximum_execution_snapshot_bytes,
        )


class HumanNestedOuterEvaluationController:
    """Human/audit facet with fixed manifest and global outer-store authority."""

    _authority_hash: str
    _evaluator_protocol_hash: str
    _manifest_root: Path
    _outer_store: NestedOuterEvaluationStore
    _runtime_fingerprint_hash: str
    _selector: NestedPurgedModelSelector

    __slots__ = (
        "_authority_hash",
        "_evaluator_protocol_hash",
        "_manifest_root",
        "_outer_store",
        "_runtime_fingerprint_hash",
        "_selector",
    )

    def __init__(
        self,
        *,
        authority_hash: str,
        manifest_root: Path,
        outer_store: NestedOuterEvaluationStore,
        evaluator_protocol_hash: str,
        runtime_fingerprint_hash: str,
        selector: NestedPurgedModelSelector,
    ) -> None:
        self._authority_hash = cast(
            str,
            require_sha256(authority_hash, name="nested selection authority hash"),
        )
        self._manifest_root = manifest_root
        self._outer_store = outer_store
        self._evaluator_protocol_hash = require_sha256(
            evaluator_protocol_hash,
            name="nested selection evaluator protocol hash",
        )
        self._runtime_fingerprint_hash = require_sha256(
            runtime_fingerprint_hash,
            name="nested selection runtime fingerprint hash",
        )
        self._selector = selector

    @property
    def authority_hash(self) -> str:
        """Opaque binding shared with the Agent facet."""

        return self._authority_hash

    def evaluate_once(
        self,
        manifest_reference: NestedSelectionPhaseOneManifestReference,
        *,
        feature_signals: Mapping[str, pd.DataFrame],
        labels: LabelResult,
        calendar: TradingCalendar,
        scoring_eligibility: pd.DataFrame | None = None,
    ) -> OuterEvaluationCompletion:
        """Execute/load outer evaluation and return metadata, never metrics."""

        manifest = self._load_manifest(manifest_reference)
        completed = self._selector.evaluate_outer_phase_managed_record(
            manifest,
            store=self._outer_store,
            evaluator_protocol_hash=self._evaluator_protocol_hash,
            runtime_fingerprint_hash=self._runtime_fingerprint_hash,
            feature_signals=feature_signals,
            labels=labels,
            calendar=calendar,
            scoring_eligibility=scoring_eligibility,
        )
        return cast(OuterEvaluationCompletion, completed.completion)

    def read_completed(
        self,
        manifest_reference: NestedSelectionPhaseOneManifestReference,
    ) -> CompletedOuterEvaluation:
        """Explicit audit read of a pre-existing completion and full metrics."""

        manifest = self._load_manifest(manifest_reference)
        return self._outer_store.load_completed(
            manifest,
            evaluator_protocol_hash=self._evaluator_protocol_hash,
            runtime_fingerprint_hash=self._runtime_fingerprint_hash,
        )

    def _load_manifest(
        self,
        reference: NestedSelectionPhaseOneManifestReference,
    ) -> NestedSelectionPhaseOneManifest:
        if not isinstance(reference, NestedSelectionPhaseOneManifestReference):
            raise TypeError("phase-one manifest reference type differs")
        return load_nested_selection_phase_one_manifest(
            self._manifest_root / reference.filename,
            reference,
        )


class NestedSelectionResearchAuthority:
    """Trusted factory that fixes one global authority root for both facets."""

    _authority_hash: str
    _evaluator_protocol_hash: str
    _manifest_root: Path
    _outer_store: NestedOuterEvaluationStore
    _runtime_fingerprint_hash: str

    __slots__ = (
        "_authority_hash",
        "_evaluator_protocol_hash",
        "_manifest_root",
        "_outer_store",
        "_runtime_fingerprint_hash",
    )

    def __init__(
        self,
        authority_root: str | Path,
        *,
        evaluator_protocol_hash: str,
        runtime_fingerprint_hash: str,
    ) -> None:
        root = Path(os.path.abspath(os.fspath(Path(authority_root).expanduser())))
        protocol_hash = require_sha256(
            evaluator_protocol_hash,
            name="nested selection evaluator protocol hash",
        )
        runtime_hash = require_sha256(
            runtime_fingerprint_hash,
            name="nested selection runtime fingerprint hash",
        )
        object.__setattr__(self, "_manifest_root", root / "manifests")
        object.__setattr__(
            self,
            "_outer_store",
            NestedOuterEvaluationStore(root / "outer_store"),
        )
        object.__setattr__(self, "_evaluator_protocol_hash", protocol_hash)
        object.__setattr__(self, "_runtime_fingerprint_hash", runtime_hash)
        object.__setattr__(
            self,
            "_authority_hash",
            cast(
                str,
                hash_json(
                    {
                        "schema_version": "nested-selection-research-authority/v1",
                        "authority_root": str(root),
                        "evaluator_protocol_hash": protocol_hash,
                        "runtime_fingerprint_hash": runtime_hash,
                    }
                ),
            ),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("NestedSelectionResearchAuthority is immutable")

    @property
    def authority_hash(self) -> str:
        """Opaque identity; the filesystem root itself is not delegated."""

        return self._authority_hash

    def agent_controller(
        self,
        *,
        selector: NestedPurgedModelSelector | None = None,
    ) -> AgentNestedSelectionController:
        return AgentNestedSelectionController(
            authority_hash=self._authority_hash,
            manifest_root=self._manifest_root,
            selector=selector or NestedPurgedModelSelector(),
        )

    def human_controller(
        self,
        *,
        selector: NestedPurgedModelSelector | None = None,
    ) -> HumanNestedOuterEvaluationController:
        return HumanNestedOuterEvaluationController(
            authority_hash=self._authority_hash,
            manifest_root=self._manifest_root,
            outer_store=self._outer_store,
            evaluator_protocol_hash=self._evaluator_protocol_hash,
            runtime_fingerprint_hash=self._runtime_fingerprint_hash,
            selector=selector or NestedPurgedModelSelector(),
        )

    def audit_reader(self) -> NestedOuterEvaluationAuditReader:
        """Return an immutable read-only view of completed outer evidence."""

        return NestedOuterEvaluationAuditReader(
            authority_hash=self._authority_hash,
            manifest_root=self._manifest_root,
            outer_store=self._outer_store,
            evaluator_protocol_hash=self._evaluator_protocol_hash,
            runtime_fingerprint_hash=self._runtime_fingerprint_hash,
        )


__all__ = [
    "AgentNestedSelectionController",
    "HumanNestedOuterEvaluationController",
    "NestedOuterEvaluationAuditReader",
    "NestedSelectionResearchAuthority",
    "VerifiedNestedSelectionPhaseOne",
]
