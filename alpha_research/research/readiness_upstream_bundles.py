"""Controlled, replayable score recipes for ML-F readiness research.

This module is a deliberately closed authority boundary.  Its public producer
accepts only a preregistered readiness plan and one exact sealed outer
evaluation.  Candidate ranks are rebuilt from the sealed prediction-rank
evidence; the lagged baseline and EMA parameter-stability panels are then
computed internally.  Callers cannot inject score frames, labels, arbitrary
operators, or post-hoc output identities.

The resulting artifacts remain research-only.  They are not a production
release and do not grant access to hidden out-of-sample data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import re
from types import MappingProxyType
from typing import Final, TypeAlias, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)
from alpha_research.models.nested_outer_evaluation_store import (
    CompletedOuterEvaluation,
)
from alpha_research.research.model_training_inputs import strong_hash_frame
from alpha_research.research.readiness_inputs import (
    ResearchReadinessFrameReferenceV1,
    ResearchReadinessInputError,
    ResearchReadinessPlanV1,
    ResearchReadinessScoreArtifactsV1,
)
from factor_production.v5.artifacts import (
    ArtifactRecord,
    ArtifactStore,
)
from factor_production.v5.artifacts.manifest import ArtifactError


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
RecipeParameter: TypeAlias = str | int | bool

_DEFINITION_SCHEMA: Final = "research-readiness-score-recipe-definition/v2"
_RECIPE_SET_SCHEMA: Final = "research-readiness-score-recipe-set/v2"
_RECEIPT_SCHEMA: Final = "research-readiness-score-recipe-execution-receipt/v2"
_PUBLICATION_SCHEMA: Final = "research-readiness-score-publication-manifest/v2"
_RESULT_SCHEMA: Final = "research-readiness-score-construction-result/v2"
_OUTER_BINDING_SCHEMA: Final = "research-readiness-outer-score-binding/v2"
_ALGORITHM_VERSION: Final = "controlled-score-recipes/2.0.0"
_REPRESENTATION: Final = "cross_sectional_rank"
_PARQUET_MEDIA_TYPE: Final = "application/vnd.apache.parquet"
_JSON_MEDIA_TYPE: Final = "application/json"

_CANDIDATE_NAME: Final = "outer_selected_candidate_rank"
_CANDIDATE_RECIPE: Final = "outer_selected_candidate_rank_v1"
_CANDIDATE_SOURCE: Final = "sealed_outer_prediction_rank_evidence"
_BASELINE_NAME: Final = "one_session_lagged_candidate_rank"
_BASELINE_RECIPE: Final = "one_session_lagged_candidate_rank_v1"
_BASELINE_SOURCE: Final = "candidate_score_frame"
_EMA_RECIPE: Final = "candidate_ema_rank_v1"
_EMA_SOURCE: Final = "candidate_score_frame"
_EMA_NAME = re.compile(r"ema_span_([2-9]|1[0-9]|20)")

_DEFINITION_ROLE: Final = "readiness_score_definition"
_RECEIPT_ROLE: Final = "readiness_score_execution_receipt"
_PUBLICATION_ROLE: Final = "readiness_score_publication_manifest"


def _error(code: str, detail: str) -> ResearchReadinessInputError:
    return ResearchReadinessInputError(code, detail)


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise _error("invalid_hash", f"{name} must be a hash")
    try:
        return cast(str, require_sha256(value, name=name))
    except ValueError as exc:
        raise _error("invalid_hash", str(exc)) from exc


def _positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _error("invalid_resource_budget", f"{name} must be a positive integer")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _error("invalid_publication_manifest", f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _exact_fields(
    value: Mapping[str, object], expected: set[str], *, name: str
) -> None:
    if set(value) != expected:
        raise _error("wire_fields_differ", f"{name} fields differ")


def _canonical_json(payload: bytes, *, name: str) -> Mapping[str, object]:
    if not isinstance(payload, bytes) or not payload:
        raise _error("invalid_wire", f"{name} payload differs")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error("invalid_wire", f"{name} is not JSON") from exc
    value = _mapping(decoded, name=name)
    if canonical_json_bytes(value) != payload:
        raise _error("noncanonical_wire", f"{name} is not canonical JSON")
    return value


@dataclass(frozen=True, slots=True)
class ResearchReadinessScoreRecipeDefinitionV2:
    """One closed recipe definition whose hash is knowable before evaluation."""

    definition_name: str
    semantic_role: str
    recipe: str
    source_kind: str
    nested_selection_spec_hash: str
    parameters: Mapping[str, RecipeParameter]
    score_representation: str = _REPRESENTATION
    algorithm_version: str = _ALGORITHM_VERSION
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _DEFINITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _DEFINITION_SCHEMA:
            raise _error("invalid_score_recipe", "definition schema differs")
        if self.algorithm_version != _ALGORITHM_VERSION:
            raise _error("invalid_score_recipe", "algorithm version differs")
        if self.score_representation != _REPRESENTATION:
            raise _error("invalid_score_recipe", "score representation differs")
        if self.research_only is not True or self.production_ready is not False:
            raise _error("invalid_release_flags", "score recipes must be research-only")
        nested_hash = _digest(
            self.nested_selection_spec_hash,
            name="nested_selection_spec_hash",
        )
        raw_parameters = self.parameters
        if not isinstance(raw_parameters, Mapping) or not all(
            isinstance(key, str)
            and isinstance(value, (str, int, bool))
            and not isinstance(value, float)
            for key, value in raw_parameters.items()
        ):
            raise _error("invalid_score_recipe", "recipe parameters differ")
        parameters = dict(sorted(raw_parameters.items()))

        if self.semantic_role == "candidate":
            valid = (
                self.definition_name == _CANDIDATE_NAME
                and self.recipe == _CANDIDATE_RECIPE
                and self.source_kind == _CANDIDATE_SOURCE
                and not parameters
            )
        elif self.semantic_role == "baseline":
            valid = (
                self.definition_name == _BASELINE_NAME
                and self.recipe == _BASELINE_RECIPE
                and self.source_kind == _BASELINE_SOURCE
                and parameters
                == {
                    "lag_sessions": 1,
                    "missing_history_policy": "current_day_midrank_then_rerank",
                }
            )
        elif self.semantic_role == "perturbation":
            match = _EMA_NAME.fullmatch(self.definition_name)
            span = parameters.get("span")
            valid = (
                match is not None
                and self.recipe == _EMA_RECIPE
                and self.source_kind == _EMA_SOURCE
                and isinstance(span, int)
                and not isinstance(span, bool)
                and 2 <= span <= 20
                and self.definition_name == f"ema_span_{span}"
                and parameters
                == {
                    "adjust": False,
                    "ignore_missing": True,
                    "minimum_periods": 1,
                    "span": span,
                }
            )
        else:
            valid = False
        if not valid:
            raise _error("invalid_score_recipe", "recipe is outside the closed set")
        object.__setattr__(self, "nested_selection_spec_hash", nested_hash)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "definition_name": self.definition_name,
            "semantic_role": self.semantic_role,
            "recipe": self.recipe,
            "source_kind": self.source_kind,
            "nested_selection_spec_hash": self.nested_selection_spec_hash,
            "parameters": dict(self.parameters),
            "score_representation": self.score_representation,
            "algorithm_version": self.algorithm_version,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def candidate(
        cls, *, nested_selection_spec_hash: str
    ) -> "ResearchReadinessScoreRecipeDefinitionV2":
        return cls(
            definition_name=_CANDIDATE_NAME,
            semantic_role="candidate",
            recipe=_CANDIDATE_RECIPE,
            source_kind=_CANDIDATE_SOURCE,
            nested_selection_spec_hash=nested_selection_spec_hash,
            parameters={},
        )

    @classmethod
    def baseline(
        cls, *, nested_selection_spec_hash: str
    ) -> "ResearchReadinessScoreRecipeDefinitionV2":
        return cls(
            definition_name=_BASELINE_NAME,
            semantic_role="baseline",
            recipe=_BASELINE_RECIPE,
            source_kind=_BASELINE_SOURCE,
            nested_selection_spec_hash=nested_selection_spec_hash,
            parameters={
                "lag_sessions": 1,
                "missing_history_policy": "current_day_midrank_then_rerank",
            },
        )

    @classmethod
    def ema(
        cls, *, nested_selection_spec_hash: str, span: int
    ) -> "ResearchReadinessScoreRecipeDefinitionV2":
        if not isinstance(span, int) or isinstance(span, bool) or not 2 <= span <= 20:
            raise _error("invalid_score_recipe", "EMA span must be an integer in [2,20]")
        return cls(
            definition_name=f"ema_span_{span}",
            semantic_role="perturbation",
            recipe=_EMA_RECIPE,
            source_kind=_EMA_SOURCE,
            nested_selection_spec_hash=nested_selection_spec_hash,
            parameters={
                "adjust": False,
                "ignore_missing": True,
                "minimum_periods": 1,
                "span": span,
            },
        )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessScoreRecipeDefinitionV2":
        _exact_fields(
            value,
            {
                "schema_version",
                "definition_name",
                "semantic_role",
                "recipe",
                "source_kind",
                "nested_selection_spec_hash",
                "parameters",
                "score_representation",
                "algorithm_version",
                "research_only",
                "production_ready",
            },
            name="score recipe definition",
        )
        parameters = _mapping(value["parameters"], name="recipe parameters")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            definition_name=cast(str, value["definition_name"]),
            semantic_role=cast(str, value["semantic_role"]),
            recipe=cast(str, value["recipe"]),
            source_kind=cast(str, value["source_kind"]),
            nested_selection_spec_hash=cast(
                str, value["nested_selection_spec_hash"]
            ),
            parameters=cast(Mapping[str, RecipeParameter], parameters),
            score_representation=cast(str, value["score_representation"]),
            algorithm_version=cast(str, value["algorithm_version"]),
            research_only=cast(bool, value["research_only"]),
            production_ready=cast(bool, value["production_ready"]),
        )

    @classmethod
    def from_wire_bytes(
        cls, payload: bytes
    ) -> "ResearchReadinessScoreRecipeDefinitionV2":
        return cls.from_mapping(_canonical_json(payload, name="score recipe definition"))


@dataclass(frozen=True, slots=True)
class ResearchReadinessScoreRecipeSetV2:
    """Candidate, fixed lagged baseline, and at least two distinct EMA recipes."""

    candidate: ResearchReadinessScoreRecipeDefinitionV2
    baseline: ResearchReadinessScoreRecipeDefinitionV2
    perturbations: tuple[ResearchReadinessScoreRecipeDefinitionV2, ...]

    def __post_init__(self) -> None:
        if (
            type(self.candidate) is not ResearchReadinessScoreRecipeDefinitionV2
            or self.candidate.semantic_role != "candidate"
            or type(self.baseline) is not ResearchReadinessScoreRecipeDefinitionV2
            or self.baseline.semantic_role != "baseline"
        ):
            raise _error("invalid_score_recipe", "candidate or baseline recipe differs")
        perturbations = tuple(self.perturbations)
        if len(perturbations) < 2 or not all(
            type(item) is ResearchReadinessScoreRecipeDefinitionV2
            and item.semantic_role == "perturbation"
            for item in perturbations
        ):
            raise _error(
                "invalid_score_recipe",
                "at least two exact EMA perturbation recipes are required",
            )
        perturbations = tuple(
            sorted(perturbations, key=lambda item: cast(int, item.parameters["span"]))
        )
        names = tuple(item.definition_name for item in perturbations)
        spans = tuple(item.parameters["span"] for item in perturbations)
        hashes = (
            self.candidate.content_hash,
            self.baseline.content_hash,
            *(item.content_hash for item in perturbations),
        )
        nested_hashes = {
            self.candidate.nested_selection_spec_hash,
            self.baseline.nested_selection_spec_hash,
            *(item.nested_selection_spec_hash for item in perturbations),
        }
        if (
            len(names) != len(set(names))
            or len(spans) != len(set(spans))
            or len(hashes) != len(set(hashes))
            or len(nested_hashes) != 1
        ):
            raise _error("invalid_score_recipe", "recipe identities repeat or drift")
        object.__setattr__(self, "perturbations", perturbations)

    @property
    def definitions(self) -> Mapping[str, ResearchReadinessScoreRecipeDefinitionV2]:
        return MappingProxyType(
            {
                item.definition_name: item
                for item in (self.candidate, self.baseline, *self.perturbations)
            }
        )

    @property
    def nested_selection_spec_hash(self) -> str:
        """Return the one nested-selection identity shared by every recipe."""

        return self.candidate.nested_selection_spec_hash

    def to_descriptor(self) -> Mapping[str, object]:
        """Canonical preregistration descriptor suitable for ``score_spec_hash``.

        The descriptor contains definitions rather than evaluated frame hashes,
        so its identity is knowable before any outer evaluation is opened.
        """

        return MappingProxyType(
            {
                "schema_version": _RECIPE_SET_SCHEMA,
                "algorithm_version": _ALGORITHM_VERSION,
                "score_representation": _REPRESENTATION,
                "nested_selection_spec_hash": self.nested_selection_spec_hash,
                "definitions": {
                    name: definition.to_dict()
                    for name, definition in self.definitions.items()
                },
                "research_only": True,
                "production_ready": False,
            }
        )

    @property
    def content_hash(self) -> str:
        """Hash of the immutable, pre-evaluation recipe-set descriptor."""

        return cast(str, hash_json(dict(self.to_descriptor())))

    @classmethod
    def preregister(
        cls,
        *,
        nested_selection_spec_hash: str,
        ema_spans: Sequence[int],
    ) -> "ResearchReadinessScoreRecipeSetV2":
        if isinstance(ema_spans, (str, bytes)):
            raise _error("invalid_score_recipe", "EMA spans must be an integer sequence")
        spans = tuple(ema_spans)
        if len(spans) < 2 or len(spans) != len(set(spans)):
            raise _error(
                "invalid_score_recipe", "at least two distinct EMA spans are required"
            )
        if any(
            not isinstance(span, int)
            or isinstance(span, bool)
            or not 2 <= span <= 20
            for span in spans
        ):
            raise _error("invalid_score_recipe", "EMA spans must be integers in [2,20]")
        nested_hash = _digest(
            nested_selection_spec_hash,
            name="nested_selection_spec_hash",
        )
        return cls(
            candidate=ResearchReadinessScoreRecipeDefinitionV2.candidate(
                nested_selection_spec_hash=nested_hash
            ),
            baseline=ResearchReadinessScoreRecipeDefinitionV2.baseline(
                nested_selection_spec_hash=nested_hash
            ),
            perturbations=tuple(
                ResearchReadinessScoreRecipeDefinitionV2.ema(
                    nested_selection_spec_hash=nested_hash,
                    span=span,
                )
                for span in sorted(spans)
            ),
        )

    @classmethod
    def from_plan(
        cls, plan: ResearchReadinessPlanV1
    ) -> "ResearchReadinessScoreRecipeSetV2":
        if type(plan) is not ResearchReadinessPlanV1:
            raise _error("score_definition_mismatch", "readiness plan type differs")
        spans: list[int] = []
        for name in plan.perturbation_definition_hashes:
            match = _EMA_NAME.fullmatch(name)
            if match is None:
                raise _error(
                    "invalid_score_recipe", f"unsupported perturbation recipe:{name}"
                )
            spans.append(int(match.group(1)))
        recipes = cls.preregister(
            nested_selection_spec_hash=plan.nested_selection_spec_hash,
            ema_spans=tuple(spans),
        )
        if (
            plan.candidate_definition_hash != recipes.candidate.content_hash
            or plan.baseline_definition_hash != recipes.baseline.content_hash
            or dict(plan.perturbation_definition_hashes)
            != {
                item.definition_name: item.content_hash
                for item in recipes.perturbations
            }
        ):
            raise _error(
                "score_definition_mismatch",
                "plan recipe hashes do not match the closed definitions",
            )
        return recipes


@dataclass(frozen=True, slots=True)
class ResearchReadinessScoreExecutionReceiptV2:
    """Immutable execution identity kept separate from preregistered definition."""

    recipe_name: str
    semantic_role: str
    plan_hash: str
    outer_evaluation_completion_hash: str
    outer_evaluation_result_hash: str
    outer_execution_snapshot_hash: str
    candidate_source_frame_hash: str
    definition_hash: str
    definition_artifact_payload_sha256: str
    output_frame_hash: str
    output_frame_reference_hash: str
    output_payload_sha256: str
    algorithm_version: str = _ALGORITHM_VERSION
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _RECEIPT_SCHEMA:
            raise _error("invalid_execution_receipt", "receipt schema differs")
        if self.algorithm_version != _ALGORITHM_VERSION:
            raise _error("invalid_execution_receipt", "algorithm version differs")
        if self.research_only is not True or self.production_ready is not False:
            raise _error("invalid_release_flags", "receipt must be research-only")
        if self.semantic_role not in {"candidate", "baseline", "perturbation"}:
            raise _error("invalid_execution_receipt", "receipt role differs")
        if (
            self.recipe_name not in {_CANDIDATE_NAME, _BASELINE_NAME}
            and _EMA_NAME.fullmatch(self.recipe_name) is None
        ):
            raise _error("invalid_execution_receipt", "receipt recipe differs")
        for name in (
            "plan_hash",
            "outer_evaluation_completion_hash",
            "outer_evaluation_result_hash",
            "outer_execution_snapshot_hash",
            "candidate_source_frame_hash",
            "definition_hash",
            "definition_artifact_payload_sha256",
            "output_frame_hash",
            "output_frame_reference_hash",
            "output_payload_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if self.definition_hash != self.definition_artifact_payload_sha256:
            raise _error(
                "invalid_execution_receipt",
                "canonical definition content and artifact hashes differ",
            )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "recipe_name": self.recipe_name,
            "semantic_role": self.semantic_role,
            "plan_hash": self.plan_hash,
            "outer_evaluation_completion_hash": (
                self.outer_evaluation_completion_hash
            ),
            "outer_evaluation_result_hash": self.outer_evaluation_result_hash,
            "outer_execution_snapshot_hash": self.outer_execution_snapshot_hash,
            "candidate_source_frame_hash": self.candidate_source_frame_hash,
            "definition_hash": self.definition_hash,
            "definition_artifact_payload_sha256": (
                self.definition_artifact_payload_sha256
            ),
            "output_frame_hash": self.output_frame_hash,
            "output_frame_reference_hash": self.output_frame_reference_hash,
            "output_payload_sha256": self.output_payload_sha256,
            "algorithm_version": self.algorithm_version,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessScoreExecutionReceiptV2":
        _exact_fields(
            value,
            {
                "schema_version",
                "recipe_name",
                "semantic_role",
                "plan_hash",
                "outer_evaluation_completion_hash",
                "outer_evaluation_result_hash",
                "outer_execution_snapshot_hash",
                "candidate_source_frame_hash",
                "definition_hash",
                "definition_artifact_payload_sha256",
                "output_frame_hash",
                "output_frame_reference_hash",
                "output_payload_sha256",
                "algorithm_version",
                "research_only",
                "production_ready",
            },
            name="score recipe execution receipt",
        )
        return cls(**cast(dict[str, object], dict(value)))  # type: ignore[arg-type]

    @classmethod
    def from_wire_bytes(
        cls, payload: bytes
    ) -> "ResearchReadinessScoreExecutionReceiptV2":
        return cls.from_mapping(_canonical_json(payload, name="execution receipt"))


def _artifact_mapping(
    value: object, *, name: str
) -> Mapping[str, ArtifactRecord]:
    raw = _mapping(value, name=name)
    if all(type(item) is ArtifactRecord for item in raw.values()):
        return MappingProxyType(
            dict(sorted(cast(Mapping[str, ArtifactRecord], raw).items()))
        )
    result: dict[str, ArtifactRecord] = {}
    expected_record_fields = {
        "logical_name",
        "location",
        "sha256",
        "size_bytes",
        "media_type",
        "role",
    }
    for key, item in raw.items():
        item_mapping = _mapping(item, name=f"{name}:{key}")
        _exact_fields(
            item_mapping,
            expected_record_fields,
            name=f"{name}:{key}",
        )
        try:
            result[key] = ArtifactRecord.from_dict(item_mapping)
        except (TypeError, ValueError) as exc:
            raise _error("invalid_publication_manifest", f"{name}:{key} differs") from exc
    return MappingProxyType(dict(sorted(result.items())))


def _hash_mapping(value: object, *, name: str) -> Mapping[str, str]:
    raw = _mapping(value, name=name)
    result = {
        key: _digest(item, name=f"{name}:{key}") for key, item in raw.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True, slots=True)
class ResearchReadinessScorePublicationManifestV2:
    """Closure of frame, definition, receipt, plan, and outer identities."""

    plan_hash: str
    nested_selection_spec_hash: str
    score_artifacts: ResearchReadinessScoreArtifactsV1
    score_artifacts_hash: str
    definition_artifacts: Mapping[str, ArtifactRecord]
    definition_hashes: Mapping[str, str]
    execution_receipt_artifacts: Mapping[str, ArtifactRecord]
    execution_receipt_hashes: Mapping[str, str]
    outer_evaluation_binding: Mapping[str, str]
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _PUBLICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _PUBLICATION_SCHEMA:
            raise _error("invalid_publication_manifest", "manifest schema differs")
        if type(self.score_artifacts) is not ResearchReadinessScoreArtifactsV1:
            raise _error("invalid_publication_manifest", "score artifacts type differs")
        if self.research_only is not True or self.production_ready is not False:
            raise _error("invalid_release_flags", "publication must be research-only")
        plan_hash = _digest(self.plan_hash, name="plan_hash")
        nested_hash = _digest(
            self.nested_selection_spec_hash,
            name="nested_selection_spec_hash",
        )
        score_hash = _digest(self.score_artifacts_hash, name="score_artifacts_hash")
        if score_hash != self.score_artifacts.content_hash:
            raise _error("invalid_publication_manifest", "score artifact hash differs")

        definitions = _artifact_mapping(
            self.definition_artifacts,
            name="definition_artifacts",
        )
        definition_hashes = _hash_mapping(
            self.definition_hashes,
            name="definition_hashes",
        )
        receipts = _artifact_mapping(
            self.execution_receipt_artifacts,
            name="execution_receipt_artifacts",
        )
        receipt_hashes = _hash_mapping(
            self.execution_receipt_hashes,
            name="execution_receipt_hashes",
        )
        expected_names = {
            _CANDIDATE_NAME,
            _BASELINE_NAME,
            *self.score_artifacts.perturbation_scores,
        }
        if not (
            set(definitions)
            == set(definition_hashes)
            == set(receipts)
            == set(receipt_hashes)
            == expected_names
        ):
            raise _error("invalid_publication_manifest", "recipe artifact sets differ")
        if any(
            record.sha256 != definition_hashes[name]
            or record.media_type != _JSON_MEDIA_TYPE
            or record.role != _DEFINITION_ROLE
            or record.logical_name != f"readiness.score.definition.{name}"
            for name, record in definitions.items()
        ):
            raise _error("invalid_publication_manifest", "definition references differ")
        if any(
            record.sha256 != receipt_hashes[name]
            or record.media_type != _JSON_MEDIA_TYPE
            or record.role != _RECEIPT_ROLE
            or record.logical_name != f"readiness.score.execution_receipt.{name}"
            for name, record in receipts.items()
        ):
            raise _error("invalid_publication_manifest", "receipt references differ")
        expected_definition_hashes = {
            _CANDIDATE_NAME: self.score_artifacts.candidate_definition_hash,
            _BASELINE_NAME: self.score_artifacts.baseline_definition_hash,
            **dict(self.score_artifacts.perturbation_definition_hashes),
        }
        if dict(definition_hashes) != expected_definition_hashes:
            raise _error("invalid_publication_manifest", "definition hashes differ")
        if any(
            reference.logical_name != f"readiness.score.{name}"
            for name, reference in _score_references(self.score_artifacts).items()
        ):
            raise _error("invalid_publication_manifest", "score logical names differ")

        outer = _hash_free_outer_binding(self.outer_evaluation_binding)
        object.__setattr__(self, "plan_hash", plan_hash)
        object.__setattr__(self, "nested_selection_spec_hash", nested_hash)
        object.__setattr__(self, "score_artifacts_hash", score_hash)
        object.__setattr__(self, "definition_artifacts", definitions)
        object.__setattr__(self, "definition_hashes", definition_hashes)
        object.__setattr__(self, "execution_receipt_artifacts", receipts)
        object.__setattr__(self, "execution_receipt_hashes", receipt_hashes)
        object.__setattr__(self, "outer_evaluation_binding", outer)

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "nested_selection_spec_hash": self.nested_selection_spec_hash,
            "score_artifacts": self.score_artifacts.to_dict(),
            "score_artifacts_hash": self.score_artifacts_hash,
            "definition_artifacts": {
                name: record.to_dict()
                for name, record in self.definition_artifacts.items()
            },
            "definition_hashes": dict(self.definition_hashes),
            "execution_receipt_artifacts": {
                name: record.to_dict()
                for name, record in self.execution_receipt_artifacts.items()
            },
            "execution_receipt_hashes": dict(self.execution_receipt_hashes),
            "outer_evaluation_binding": dict(self.outer_evaluation_binding),
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchReadinessScorePublicationManifestV2":
        _exact_fields(
            value,
            {
                "schema_version",
                "plan_hash",
                "nested_selection_spec_hash",
                "score_artifacts",
                "score_artifacts_hash",
                "definition_artifacts",
                "definition_hashes",
                "execution_receipt_artifacts",
                "execution_receipt_hashes",
                "outer_evaluation_binding",
                "research_only",
                "production_ready",
            },
            name="score publication manifest",
        )
        return cls(
            schema_version=cast(str, value["schema_version"]),
            plan_hash=cast(str, value["plan_hash"]),
            nested_selection_spec_hash=cast(
                str, value["nested_selection_spec_hash"]
            ),
            score_artifacts=ResearchReadinessScoreArtifactsV1.from_mapping(
                _mapping(value["score_artifacts"], name="score_artifacts")
            ),
            score_artifacts_hash=cast(str, value["score_artifacts_hash"]),
            definition_artifacts=_artifact_mapping(
                value["definition_artifacts"],
                name="definition_artifacts",
            ),
            definition_hashes=_hash_mapping(
                value["definition_hashes"],
                name="definition_hashes",
            ),
            execution_receipt_artifacts=_artifact_mapping(
                value["execution_receipt_artifacts"],
                name="execution_receipt_artifacts",
            ),
            execution_receipt_hashes=_hash_mapping(
                value["execution_receipt_hashes"],
                name="execution_receipt_hashes",
            ),
            outer_evaluation_binding=cast(
                Mapping[str, str],
                _mapping(
                    value["outer_evaluation_binding"],
                    name="outer_evaluation_binding",
                ),
            ),
            research_only=cast(bool, value["research_only"]),
            production_ready=cast(bool, value["production_ready"]),
        )

    @classmethod
    def from_wire_bytes(
        cls, payload: bytes
    ) -> "ResearchReadinessScorePublicationManifestV2":
        return cls.from_mapping(_canonical_json(payload, name="publication manifest"))


@dataclass(frozen=True, slots=True)
class ResearchReadinessScoreConstructionResultV2:
    """Published controlled score bundle with independently reloadable audit trail."""

    score_artifacts: ResearchReadinessScoreArtifactsV1
    recipe_definitions: Mapping[str, ResearchReadinessScoreRecipeDefinitionV2]
    execution_receipts: Mapping[str, ResearchReadinessScoreExecutionReceiptV2]
    publication_manifest: ResearchReadinessScorePublicationManifestV2
    publication_manifest_reference: ArtifactRecord
    research_only: bool = True
    production_ready: bool = False

    def __post_init__(self) -> None:
        if type(self.score_artifacts) is not ResearchReadinessScoreArtifactsV1:
            raise _error("invalid_score_bundle", "score artifacts type differs")
        if type(self.publication_manifest) is not ResearchReadinessScorePublicationManifestV2:
            raise _error("invalid_publication_manifest", "manifest type differs")
        if type(self.publication_manifest_reference) is not ArtifactRecord:
            raise _error("invalid_publication_manifest", "manifest reference differs")
        if self.research_only is not True or self.production_ready is not False:
            raise _error("invalid_release_flags", "score result must be research-only")
        definitions = dict(sorted(self.recipe_definitions.items()))
        receipts = dict(sorted(self.execution_receipts.items()))
        if not all(
            isinstance(name, str)
            and type(definition) is ResearchReadinessScoreRecipeDefinitionV2
            for name, definition in definitions.items()
        ) or not all(
            isinstance(name, str)
            and type(receipt) is ResearchReadinessScoreExecutionReceiptV2
            for name, receipt in receipts.items()
        ):
            raise _error("invalid_publication_manifest", "recipe audit types differ")
        manifest = self.publication_manifest
        if (
            manifest.score_artifacts != self.score_artifacts
            or set(definitions) != set(manifest.definition_hashes)
            or set(receipts) != set(manifest.execution_receipt_hashes)
            or any(
                definitions[name].content_hash != manifest.definition_hashes[name]
                for name in definitions
            )
            or any(
                receipts[name].content_hash
                != manifest.execution_receipt_hashes[name]
                for name in receipts
            )
            or self.publication_manifest_reference.sha256 != manifest.content_hash
            or self.publication_manifest_reference.media_type != _JSON_MEDIA_TYPE
            or self.publication_manifest_reference.role != _PUBLICATION_ROLE
            or self.publication_manifest_reference.logical_name
            != "readiness.score.publication_manifest"
            or any(
                definition.nested_selection_spec_hash
                != manifest.nested_selection_spec_hash
                for definition in definitions.values()
            )
        ):
            raise _error("invalid_publication_manifest", "result audit identity differs")
        _verify_receipt_bindings(
            score_artifacts=self.score_artifacts,
            definitions=definitions,
            receipts=receipts,
            manifest=manifest,
        )
        object.__setattr__(
            self, "recipe_definitions", MappingProxyType(definitions)
        )
        object.__setattr__(
            self, "execution_receipts", MappingProxyType(receipts)
        )

    @property
    def result_payload(self) -> Mapping[str, JsonValue]:
        """V2 persisted result; the manifest reference closes definitions/receipts."""

        return cast(
            Mapping[str, JsonValue],
            {
                "schema_version": _RESULT_SCHEMA,
                "research_readiness_score_artifacts": self.score_artifacts.to_dict(),
                "research_readiness_score_artifacts_hash": (
                    self.score_artifacts.content_hash
                ),
                "score_recipe_publication_manifest_reference": (
                    self.publication_manifest_reference.to_dict()
                ),
                "score_recipe_publication_manifest_hash": (
                    self.publication_manifest.content_hash
                ),
                "outer_evaluation_binding": dict(
                    self.publication_manifest.outer_evaluation_binding
                ),
                "research_only": True,
                "production_ready": False,
            },
        )

    @property
    def evidence_hashes(self) -> tuple[str, ...]:
        return (
            self.publication_manifest.plan_hash,
            self.score_artifacts.content_hash,
            *(item.content_hash for item in self.recipe_definitions.values()),
            *(item.content_hash for item in self.execution_receipts.values()),
            self.publication_manifest.content_hash,
        )


@dataclass(frozen=True, slots=True)
class _ResourceBudget:
    maximum_compressed_bytes: int
    maximum_uncompressed_bytes: int
    maximum_loaded_memory_bytes: int
    maximum_rows: int
    maximum_columns: int

    def __post_init__(self) -> None:
        for name in (
            "maximum_compressed_bytes",
            "maximum_uncompressed_bytes",
            "maximum_loaded_memory_bytes",
            "maximum_rows",
            "maximum_columns",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), name=name),
            )

    def require_axes(self, *, rows: int, columns: int, frames: int) -> None:
        if rows > self.maximum_rows:
            raise _error("resource_budget_exceeded", "score rows exceed the stage budget")
        if columns > self.maximum_columns:
            raise _error(
                "resource_budget_exceeded", "score columns exceed the stage budget"
            )
        estimated_numeric_bytes = rows * columns * frames * 8
        if estimated_numeric_bytes > self.maximum_loaded_memory_bytes:
            raise _error(
                "resource_budget_exceeded",
                "estimated score memory exceeds the stage budget",
            )
        if estimated_numeric_bytes > self.maximum_uncompressed_bytes:
            raise _error(
                "resource_budget_exceeded",
                "estimated uncompressed scores exceed the stage budget",
            )

    def require_totals(
        self,
        *,
        compressed_bytes: int,
        uncompressed_bytes: int,
        loaded_memory_bytes: int,
    ) -> None:
        if compressed_bytes > self.maximum_compressed_bytes:
            raise _error(
                "resource_budget_exceeded",
                "total compressed publication exceeds the stage budget",
            )
        if uncompressed_bytes > self.maximum_uncompressed_bytes:
            raise _error(
                "resource_budget_exceeded",
                "total uncompressed score panels exceed the stage budget",
            )
        if loaded_memory_bytes > self.maximum_loaded_memory_bytes:
            raise _error(
                "resource_budget_exceeded",
                "total loaded score panels exceed the stage budget",
            )


@dataclass(frozen=True, slots=True)
class _PreparedFrame:
    frame: pd.DataFrame
    payload: bytes
    reference: ResearchReadinessFrameReferenceV1
    record: ArtifactRecord


@dataclass(frozen=True, slots=True)
class ResearchReadinessScoreUpstreamProducer:
    """Trusted publisher used by the governed SCORE_CONSTRUCTION stage."""

    artifact_store: ArtifactStore

    def __post_init__(self) -> None:
        if type(self.artifact_store) is not ArtifactStore:
            raise TypeError("score upstream producer requires an exact ArtifactStore")

    def publish(
        self,
        *,
        readiness_plan: ResearchReadinessPlanV1,
        completed_outer_evaluation: CompletedOuterEvaluation,
        maximum_compressed_bytes: int,
        maximum_uncompressed_bytes: int,
        maximum_loaded_memory_bytes: int,
        maximum_rows: int,
        maximum_columns: int,
    ) -> ResearchReadinessScoreConstructionResultV2:
        """Rebuild and publish only the recipes preregistered in ``readiness_plan``."""

        budget = _ResourceBudget(
            maximum_compressed_bytes=maximum_compressed_bytes,
            maximum_uncompressed_bytes=maximum_uncompressed_bytes,
            maximum_loaded_memory_bytes=maximum_loaded_memory_bytes,
            maximum_rows=maximum_rows,
            maximum_columns=maximum_columns,
        )
        if type(readiness_plan) is not ResearchReadinessPlanV1:
            raise _error("score_definition_mismatch", "readiness plan type differs")
        if type(completed_outer_evaluation) is not CompletedOuterEvaluation:
            raise _error("outer_score_mismatch", "outer completion type differs")
        recipes = ResearchReadinessScoreRecipeSetV2.from_plan(readiness_plan)
        _verify_outer_lineage(
            readiness_plan=readiness_plan,
            completed=completed_outer_evaluation,
        )

        vector_count, member_count = _outer_axes(completed_outer_evaluation)
        frame_count = 2 + len(recipes.perturbations)
        budget.require_axes(
            rows=vector_count,
            columns=member_count,
            frames=frame_count,
        )
        candidate = _rebuild_outer_candidate(completed_outer_evaluation)
        _verify_plan_window(candidate, plan=readiness_plan)
        baseline = _one_session_lagged_rank(candidate)
        perturbations = {
            definition.definition_name: _candidate_ema_rank(
                candidate,
                span=cast(int, definition.parameters["span"]),
            )
            for definition in recipes.perturbations
        }
        frames = {
            _CANDIDATE_NAME: candidate,
            _BASELINE_NAME: baseline,
            **perturbations,
        }
        _require_same_axes_membership_and_distinct(frames)

        prepared = {
            name: self._prepare_frame(name=name, frame=frame)
            for name, frame in frames.items()
        }
        definition_payloads = {
            name: definition.to_wire_bytes()
            for name, definition in recipes.definitions.items()
        }
        definition_records = {
            name: self._predict_record(
                logical_name=f"readiness.score.definition.{name}",
                payload=payload,
                media_type=_JSON_MEDIA_TYPE,
                role=_DEFINITION_ROLE,
            )
            for name, payload in definition_payloads.items()
        }

        outer_binding = _outer_binding(
            completed_outer_evaluation,
            readiness_plan=readiness_plan,
        )
        candidate_hash = strong_hash_frame(candidate)
        receipts = {
            name: ResearchReadinessScoreExecutionReceiptV2(
                recipe_name=name,
                semantic_role=definition.semantic_role,
                plan_hash=readiness_plan.content_hash,
                outer_evaluation_completion_hash=(
                    completed_outer_evaluation.completion.content_hash
                ),
                outer_evaluation_result_hash=(
                    completed_outer_evaluation.result.content_hash
                ),
                outer_execution_snapshot_hash=(
                    completed_outer_evaluation.execution_snapshot.content_hash
                ),
                candidate_source_frame_hash=candidate_hash,
                definition_hash=definition.content_hash,
                definition_artifact_payload_sha256=definition_records[name].sha256,
                output_frame_hash=prepared[name].reference.frame_hash,
                output_frame_reference_hash=prepared[name].reference.content_hash,
                output_payload_sha256=prepared[name].record.sha256,
            )
            for name, definition in recipes.definitions.items()
        }
        receipt_payloads = {
            name: receipt.to_wire_bytes() for name, receipt in receipts.items()
        }
        receipt_records = {
            name: self._predict_record(
                logical_name=f"readiness.score.execution_receipt.{name}",
                payload=payload,
                media_type=_JSON_MEDIA_TYPE,
                role=_RECEIPT_ROLE,
            )
            for name, payload in receipt_payloads.items()
        }

        score_artifacts = ResearchReadinessScoreArtifactsV1(
            candidate_definition_hash=recipes.candidate.content_hash,
            candidate_scores=prepared[_CANDIDATE_NAME].reference,
            baseline_definition_hash=recipes.baseline.content_hash,
            baseline_scores=prepared[_BASELINE_NAME].reference,
            perturbation_definition_hashes={
                definition.definition_name: definition.content_hash
                for definition in recipes.perturbations
            },
            perturbation_scores={
                definition.definition_name: prepared[definition.definition_name].reference
                for definition in recipes.perturbations
            },
        )
        manifest = ResearchReadinessScorePublicationManifestV2(
            plan_hash=readiness_plan.content_hash,
            nested_selection_spec_hash=readiness_plan.nested_selection_spec_hash,
            score_artifacts=score_artifacts,
            score_artifacts_hash=score_artifacts.content_hash,
            definition_artifacts=definition_records,
            definition_hashes={
                name: definition.content_hash
                for name, definition in recipes.definitions.items()
            },
            execution_receipt_artifacts=receipt_records,
            execution_receipt_hashes={
                name: receipt.content_hash for name, receipt in receipts.items()
            },
            outer_evaluation_binding=outer_binding,
        )
        manifest_payload = manifest.to_wire_bytes()
        manifest_record = self._predict_record(
            logical_name="readiness.score.publication_manifest",
            payload=manifest_payload,
            media_type=_JSON_MEDIA_TYPE,
            role=_PUBLICATION_ROLE,
        )

        all_payloads = (
            *(item.payload for item in prepared.values()),
            *definition_payloads.values(),
            *receipt_payloads.values(),
            manifest_payload,
        )
        budget.require_totals(
            compressed_bytes=sum(len(item) for item in all_payloads),
            uncompressed_bytes=sum(
                item.reference.parquet_uncompressed_bytes
                for item in prepared.values()
            ),
            loaded_memory_bytes=sum(
                int(item.frame.memory_usage(index=True, deep=True).sum())
                for item in prepared.values()
            ),
        )

        # Nothing enters the immutable store until every scientific and resource
        # check above has passed.
        for name, item in prepared.items():
            self._put_expected(item.record, item.payload, context=f"frame:{name}")
        for name, payload in definition_payloads.items():
            self._put_expected(
                definition_records[name], payload, context=f"definition:{name}"
            )
        for name, payload in receipt_payloads.items():
            self._put_expected(receipt_records[name], payload, context=f"receipt:{name}")
        self._put_expected(manifest_record, manifest_payload, context="manifest")

        return ResearchReadinessScoreConstructionResultV2(
            score_artifacts=score_artifacts,
            recipe_definitions=recipes.definitions,
            execution_receipts=receipts,
            publication_manifest=manifest,
            publication_manifest_reference=manifest_record,
        )

    def reopen(
        self,
        *,
        publication_manifest_reference: ArtifactRecord,
        maximum_compressed_bytes: int,
        maximum_uncompressed_bytes: int,
        maximum_loaded_memory_bytes: int,
        maximum_rows: int,
        maximum_columns: int,
    ) -> ResearchReadinessScoreConstructionResultV2:
        """Reload every content-addressed document and frame, failing closed."""

        budget = _ResourceBudget(
            maximum_compressed_bytes=maximum_compressed_bytes,
            maximum_uncompressed_bytes=maximum_uncompressed_bytes,
            maximum_loaded_memory_bytes=maximum_loaded_memory_bytes,
            maximum_rows=maximum_rows,
            maximum_columns=maximum_columns,
        )
        if type(publication_manifest_reference) is not ArtifactRecord:
            raise _error("invalid_publication_manifest", "manifest reference differs")
        if (
            publication_manifest_reference.media_type != _JSON_MEDIA_TYPE
            or publication_manifest_reference.role != _PUBLICATION_ROLE
            or publication_manifest_reference.logical_name
            != "readiness.score.publication_manifest"
        ):
            raise _error("invalid_publication_manifest", "manifest artifact role differs")
        if publication_manifest_reference.size_bytes > budget.maximum_compressed_bytes:
            raise _error(
                "resource_budget_exceeded",
                "publication manifest exceeds the stage budget",
            )
        manifest_payload = self._read(publication_manifest_reference, context="manifest")
        manifest = ResearchReadinessScorePublicationManifestV2.from_wire_bytes(
            manifest_payload
        )
        if manifest.content_hash != publication_manifest_reference.sha256:
            raise _error("invalid_publication_manifest", "manifest hash differs")

        frame_references = _score_references(manifest.score_artifacts)
        budget.require_axes(
            rows=manifest.score_artifacts.candidate_scores.row_count,
            columns=manifest.score_artifacts.candidate_scores.column_count,
            frames=len(frame_references),
        )
        all_records = (
            *(_record_from_frame(reference) for reference in frame_references.values()),
            *manifest.definition_artifacts.values(),
            *manifest.execution_receipt_artifacts.values(),
            publication_manifest_reference,
        )
        budget.require_totals(
            compressed_bytes=sum(record.size_bytes for record in all_records),
            uncompressed_bytes=sum(
                reference.parquet_uncompressed_bytes
                for reference in frame_references.values()
            ),
            loaded_memory_bytes=sum(
                reference.row_count * reference.column_count * 8
                for reference in frame_references.values()
            ),
        )

        loaded_frames: dict[str, pd.DataFrame] = {}
        for name, reference in frame_references.items():
            payload = self._read(_record_from_frame(reference), context=f"frame:{name}")
            frame = _decode_parquet(payload, name=f"frame:{name}")
            reference.verify_frame(frame)
            loaded_frames[name] = frame
        budget.require_totals(
            compressed_bytes=sum(record.size_bytes for record in all_records),
            uncompressed_bytes=sum(
                reference.parquet_uncompressed_bytes
                for reference in frame_references.values()
            ),
            loaded_memory_bytes=sum(
                int(frame.memory_usage(index=True, deep=True).sum())
                for frame in loaded_frames.values()
            ),
        )
        _require_same_axes_membership_and_distinct(loaded_frames)

        definitions = {
            name: ResearchReadinessScoreRecipeDefinitionV2.from_wire_bytes(
                self._read(record, context=f"definition:{name}")
            )
            for name, record in manifest.definition_artifacts.items()
        }
        receipts = {
            name: ResearchReadinessScoreExecutionReceiptV2.from_wire_bytes(
                self._read(record, context=f"receipt:{name}")
            )
            for name, record in manifest.execution_receipt_artifacts.items()
        }
        if any(
            definition.content_hash != manifest.definition_hashes[name]
            for name, definition in definitions.items()
        ) or any(
            receipt.content_hash != manifest.execution_receipt_hashes[name]
            for name, receipt in receipts.items()
        ):
            raise _error("invalid_publication_manifest", "audit document hashes differ")

        return ResearchReadinessScoreConstructionResultV2(
            score_artifacts=manifest.score_artifacts,
            recipe_definitions=definitions,
            execution_receipts=receipts,
            publication_manifest=manifest,
            publication_manifest_reference=publication_manifest_reference,
        )

    def _prepare_frame(self, *, name: str, frame: pd.DataFrame) -> _PreparedFrame:
        role = (
            "candidate_score"
            if name == _CANDIDATE_NAME
            else "baseline_score"
            if name == _BASELINE_NAME
            else "perturbation_score"
        )
        logical_name = f"readiness.score.{name}"
        payload = _parquet_bytes(frame, name=logical_name)
        decoded, uncompressed_bytes = _verified_parquet_roundtrip(
            payload,
            expected=frame,
            name=logical_name,
        )
        record = self._predict_record(
            logical_name=logical_name,
            payload=payload,
            media_type=_PARQUET_MEDIA_TYPE,
            role=role,
        )
        reference = ResearchReadinessFrameReferenceV1.bind_verified_frame(
            record=record,
            frame=decoded,
            semantic_role=role,
            parquet_uncompressed_bytes=uncompressed_bytes,
        )
        reference.verify_frame(frame)
        return _PreparedFrame(
            frame=frame,
            payload=payload,
            reference=reference,
            record=record,
        )

    def _predict_record(
        self,
        *,
        logical_name: str,
        payload: bytes,
        media_type: str,
        role: str,
    ) -> ArtifactRecord:
        digest = hashlib.sha256(payload).hexdigest()
        destination = self.artifact_store.object_root / digest[:2] / digest
        try:
            location = destination.relative_to(
                self.artifact_store.workspace_root
            ).as_posix()
        except ValueError as exc:  # pragma: no cover - ArtifactStore constructor guards it
            raise _error("unsafe_artifact_reference", "artifact root escapes workspace") from exc
        return ArtifactRecord(
            logical_name=logical_name,
            location=location,
            sha256=digest,
            size_bytes=len(payload),
            media_type=media_type,
            role=role,
        )

    def _put_expected(
        self, expected: ArtifactRecord, payload: bytes, *, context: str
    ) -> None:
        observed = self.artifact_store.put_bytes(
            expected.logical_name,
            payload,
            media_type=expected.media_type,
            role=expected.role,
        )
        if observed != expected:
            raise _error("artifact_payload_mismatch", f"{context} record differs")
        if self._read(observed, context=context) != payload:
            raise _error("artifact_payload_mismatch", f"{context} reload differs")

    def _read(self, record: ArtifactRecord, *, context: str) -> bytes:
        try:
            return cast(bytes, self.artifact_store.read_bytes(record))
        except (ArtifactError, OSError, ValueError) as exc:
            raise _error("artifact_payload_mismatch", f"{context} reload failed") from exc


def _outer_axes(completed: CompletedOuterEvaluation) -> tuple[int, int]:
    seen: set[pd.Timestamp] = set()
    members: set[str] = set()
    for evaluation in completed.result.outer_evaluations:
        for vector in evaluation.outer_score.prediction_rank_evidence.rank_vectors:
            timestamp = pd.Timestamp(vector.signal_timestamp)
            if timestamp.tzinfo is None or timestamp in seen:
                raise _error("outer_score_mismatch", "outer rank timestamps differ")
            seen.add(timestamp)
            members.update(vector.security_members)
    if not seen or not members:
        raise _error("outer_score_mismatch", "outer rank evidence is empty")
    return len(seen), len(members)


def _rebuild_outer_candidate(completed: CompletedOuterEvaluation) -> pd.DataFrame:
    vectors = tuple(
        vector
        for evaluation in completed.result.outer_evaluations
        for vector in evaluation.outer_score.prediction_rank_evidence.rank_vectors
    )
    timestamps = pd.DatetimeIndex(
        sorted(pd.Timestamp(vector.signal_timestamp).tz_convert("UTC") for vector in vectors)
    )
    columns = pd.Index(
        sorted({member for vector in vectors for member in vector.security_members})
    )
    candidate = pd.DataFrame(np.nan, index=timestamps, columns=columns, dtype=float)
    seen: set[pd.Timestamp] = set()
    for vector in vectors:
        timestamp = pd.Timestamp(vector.signal_timestamp).tz_convert("UTC")
        if (
            timestamp in seen
            or tuple(sorted(vector.security_members)) != vector.security_members
            or len(vector.security_members) != len(set(vector.security_members))
        ):
            raise _error("outer_score_mismatch", "outer rank axes differ")
        candidate.loc[timestamp, list(vector.security_members)] = tuple(
            vector.prediction_ranks
        )
        seen.add(timestamp)
    canonical = _canonical_cross_sectional_rank(candidate)
    if not canonical.equals(candidate):
        raise _error(
            "score_representation_mismatch",
            "sealed outer candidate is not canonical rank evidence",
        )
    return candidate


def _one_session_lagged_rank(candidate: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        np.nan,
        index=candidate.index.copy(),
        columns=candidate.columns.copy(),
        dtype=float,
    )
    membership = candidate.notna()
    for offset, timestamp in enumerate(candidate.index):
        active = membership.iloc[offset]
        active_count = int(active.sum())
        midrank = (active_count + 1.0) / 2.0
        source = pd.Series(midrank, index=candidate.columns, dtype=float)
        if offset > 0:
            previous = candidate.iloc[offset - 1]
            shared = active & previous.notna()
            source.loc[shared] = previous.loc[shared]
        result.loc[timestamp, active] = source.loc[active].rank(method="average")
    return result


def _candidate_ema_rank(candidate: pd.DataFrame, *, span: int) -> pd.DataFrame:
    smoothed = candidate.ewm(
        span=span,
        adjust=False,
        ignore_na=True,
        min_periods=1,
    ).mean()
    return _canonical_cross_sectional_rank(smoothed.where(candidate.notna()))


def _canonical_cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", na_option="keep").astype(float)


def _require_same_axes_membership_and_distinct(
    frames: Mapping[str, pd.DataFrame],
) -> None:
    if set(frames) < {_CANDIDATE_NAME, _BASELINE_NAME}:
        raise _error("invalid_score_bundle", "required score panels are missing")
    candidate = frames[_CANDIDATE_NAME]
    membership = candidate.notna()
    identities: list[str] = []
    for name, frame in frames.items():
        if type(frame) is not pd.DataFrame:
            raise _error("invalid_score_frame", f"{name} frame type differs")
        if (
            not frame.index.equals(candidate.index)
            or not frame.columns.equals(candidate.columns)
        ):
            raise _error("panel_axis_mismatch", "score frame axes differ")
        if not frame.notna().equals(membership):
            raise _error("member_set_mismatch", "score memberships differ")
        if not _canonical_cross_sectional_rank(frame).equals(frame):
            raise _error(
                "score_representation_mismatch",
                f"{name} is not canonical cross-sectional rank",
            )
        identities.append(strong_hash_frame(frame))
    if len(identities) != len(set(identities)):
        raise _error("duplicate_score", "controlled score recipe outputs repeat")


def _verify_outer_lineage(
    *,
    readiness_plan: ResearchReadinessPlanV1,
    completed: CompletedOuterEvaluation,
) -> None:
    result = completed.result
    completion = completed.completion
    if (
        readiness_plan.nested_selection_spec_hash != result.selection_spec_hash
        or readiness_plan.validation_spec_hash != result.outer_validation_spec_hash
        or readiness_plan.outer_validation_receipt_hash
        != completion.evaluation_partition_hash
        or readiness_plan.outer_validation_receipt_hash
        != result.outer_validation_receipt_hash
        or readiness_plan.candidate_score_representation != _REPRESENTATION
        or completion.result_hash != result.content_hash
        or completion.execution_snapshot_hash
        != completed.execution_snapshot.content_hash
    ):
        raise _error("outer_score_mismatch", "outer plan lineage differs")


def _verify_plan_window(candidate: pd.DataFrame, *, plan: ResearchReadinessPlanV1) -> None:
    start = pd.Timestamp(plan.validation_window_start).tz_convert("UTC")
    end = pd.Timestamp(plan.validation_window_end).tz_convert("UTC")
    if candidate.index[0] < start or candidate.index[-1] > end:
        raise _error("outer_score_mismatch", "outer candidate falls outside plan window")


def _outer_binding(
    completed: CompletedOuterEvaluation,
    *,
    readiness_plan: ResearchReadinessPlanV1,
) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "schema_version": _OUTER_BINDING_SCHEMA,
            "candidate_score_representation": _REPRESENTATION,
            "validation_partition_hash": readiness_plan.validation_partition_hash,
            "outer_validation_receipt_hash": (
                completed.completion.evaluation_partition_hash
            ),
            "phase_one_manifest_hash": completed.completion.manifest_hash,
            "outer_evaluation_completion_hash": completed.completion.content_hash,
            "outer_evaluation_result_hash": completed.result.content_hash,
            "outer_execution_snapshot_hash": completed.execution_snapshot.content_hash,
        }
    )


def _hash_free_outer_binding(value: object) -> Mapping[str, str]:
    raw = _mapping(value, name="outer_evaluation_binding")
    expected = {
        "schema_version",
        "candidate_score_representation",
        "validation_partition_hash",
        "outer_validation_receipt_hash",
        "phase_one_manifest_hash",
        "outer_evaluation_completion_hash",
        "outer_evaluation_result_hash",
        "outer_execution_snapshot_hash",
    }
    _exact_fields(raw, expected, name="outer_evaluation_binding")
    if (
        raw["schema_version"] != _OUTER_BINDING_SCHEMA
        or raw["candidate_score_representation"] != _REPRESENTATION
    ):
        raise _error("invalid_publication_manifest", "outer binding contract differs")
    result = {
        "schema_version": cast(str, raw["schema_version"]),
        "candidate_score_representation": cast(
            str, raw["candidate_score_representation"]
        ),
    }
    for name in expected - {"schema_version", "candidate_score_representation"}:
        result[name] = _digest(raw[name], name=name)
    return MappingProxyType(dict(sorted(result.items())))


def _score_references(
    artifacts: ResearchReadinessScoreArtifactsV1,
) -> Mapping[str, ResearchReadinessFrameReferenceV1]:
    return MappingProxyType(
        {
            _CANDIDATE_NAME: artifacts.candidate_scores,
            _BASELINE_NAME: artifacts.baseline_scores,
            **dict(artifacts.perturbation_scores),
        }
    )


def _record_from_frame(reference: ResearchReadinessFrameReferenceV1) -> ArtifactRecord:
    return ArtifactRecord(
        logical_name=reference.logical_name,
        location=reference.location,
        sha256=reference.payload_sha256,
        size_bytes=reference.size_bytes,
        media_type=reference.media_type,
        role=reference.semantic_role,
    )


def _verify_receipt_bindings(
    *,
    score_artifacts: ResearchReadinessScoreArtifactsV1,
    definitions: Mapping[str, ResearchReadinessScoreRecipeDefinitionV2],
    receipts: Mapping[str, ResearchReadinessScoreExecutionReceiptV2],
    manifest: ResearchReadinessScorePublicationManifestV2,
) -> None:
    references = _score_references(score_artifacts)
    candidate_hash = score_artifacts.candidate_scores.frame_hash
    outer = manifest.outer_evaluation_binding
    for name, definition in definitions.items():
        receipt = receipts[name]
        reference = references[name]
        if (
            receipt.recipe_name != name
            or receipt.semantic_role != definition.semantic_role
            or receipt.plan_hash != manifest.plan_hash
            or receipt.outer_evaluation_completion_hash
            != outer["outer_evaluation_completion_hash"]
            or receipt.outer_evaluation_result_hash
            != outer["outer_evaluation_result_hash"]
            or receipt.outer_execution_snapshot_hash
            != outer["outer_execution_snapshot_hash"]
            or receipt.candidate_source_frame_hash != candidate_hash
            or receipt.definition_hash != definition.content_hash
            or receipt.definition_artifact_payload_sha256
            != manifest.definition_artifacts[name].sha256
            or receipt.output_frame_hash != reference.frame_hash
            or receipt.output_frame_reference_hash != reference.content_hash
            or receipt.output_payload_sha256 != reference.payload_sha256
        ):
            raise _error("invalid_execution_receipt", f"receipt binding differs:{name}")


def _parquet_bytes(frame: pd.DataFrame, *, name: str) -> bytes:
    buffer = io.BytesIO()
    try:
        frame.to_parquet(buffer, engine="pyarrow", compression="zstd", index=True)
    except (OSError, TypeError, ValueError) as exc:
        raise _error("parquet_serialization_failed", f"{name} cannot be serialized") from exc
    payload = buffer.getvalue()
    if not payload:
        raise _error("parquet_serialization_failed", f"{name} payload is empty")
    return payload


def _decode_parquet(payload: bytes, *, name: str) -> pd.DataFrame:
    try:
        value = pq.read_table(pa.BufferReader(payload)).to_pandas()
    except (pa.ArrowException, OSError, TypeError, ValueError) as exc:
        raise _error("parquet_serialization_failed", f"{name} cannot be decoded") from exc
    if type(value) is not pd.DataFrame:
        raise _error("parquet_roundtrip_mismatch", f"{name} is not a DataFrame")
    return value


def _verified_parquet_roundtrip(
    payload: bytes,
    *,
    expected: pd.DataFrame,
    name: str,
) -> tuple[pd.DataFrame, int]:
    try:
        parquet = pq.ParquetFile(pa.BufferReader(payload))
        decoded = parquet.read().to_pandas()
    except (pa.ArrowException, OSError, TypeError, ValueError) as exc:
        raise _error("parquet_serialization_failed", f"{name} cannot be verified") from exc
    if type(decoded) is not pd.DataFrame or not decoded.equals(expected):
        raise _error("parquet_roundtrip_mismatch", f"{name} roundtrip differs")
    metadata = parquet.metadata
    if metadata is None:
        raise _error("invalid_parquet", f"{name} metadata is missing")
    uncompressed = sum(
        int(metadata.row_group(row).column(column).total_uncompressed_size)
        for row in range(metadata.num_row_groups)
        for column in range(metadata.row_group(row).num_columns)
    )
    if uncompressed <= 0:
        raise _error("invalid_parquet", f"{name} uncompressed size differs")
    return decoded, uncompressed


__all__ = [
    "ResearchReadinessScoreConstructionResultV2",
    "ResearchReadinessScoreExecutionReceiptV2",
    "ResearchReadinessScorePublicationManifestV2",
    "ResearchReadinessScoreRecipeDefinitionV2",
    "ResearchReadinessScoreRecipeSetV2",
    "ResearchReadinessScoreUpstreamProducer",
]
