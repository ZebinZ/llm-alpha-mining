from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)
from alpha_research.research.spec import ResearchRunSpec, ResearchStage


_ENVELOPE_FIELDS = {
    "schema_version",
    "research_run_spec_hash",
    "stage",
    "component_bindings",
    "parent_artifact_hashes",
    "payload",
    "payload_hash",
    "binding_hash",
}

_PROTECTED_PARTITION_BINDINGS = frozenset(
    {"data:partition:test", "data:partition:holdout"}
)
_EARLY_DATA_STAGES = frozenset(
    {
        ResearchStage.DATA_QUALITY,
        ResearchStage.FACTOR_GENERATION,
        ResearchStage.LABEL_BUILDING,
        ResearchStage.VALIDATION_SPLIT,
        ResearchStage.MODEL_TRAINING,
        ResearchStage.FACTOR_EVALUATION,
        ResearchStage.SCORE_CONSTRUCTION,
        ResearchStage.PORTFOLIO_CONSTRUCTION,
        ResearchStage.BACKTEST,
        ResearchStage.ROBUSTNESS,
        ResearchStage.REPORT,
    }
)


@dataclass(frozen=True, slots=True)
class StageContract:
    research_run_spec_hash: str
    stage: ResearchStage | str
    component_bindings: Mapping[str, str]
    required_parent_stages: tuple[ResearchStage | str, ...]
    payload_schema: str
    schema_version: str = "stage-contract/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "stage-contract/v1":
            raise ValueError("unsupported StageContract schema")
        require_sha256(
            self.research_run_spec_hash, name="stage contract research_run_spec_hash"
        )
        stage = ResearchStage(self.stage)
        object.__setattr__(self, "stage", stage)
        bindings = dict(sorted(self.component_bindings.items()))
        if not bindings or any(not role.strip() for role in bindings):
            raise ValueError("stage contract requires named component bindings")
        for role, digest in bindings.items():
            require_sha256(digest, name=f"stage contract component:{role}")
        forbidden_partitions = sorted(
            set(bindings).intersection(_PROTECTED_PARTITION_BINDINGS)
        )
        if stage in _EARLY_DATA_STAGES and forbidden_partitions:
            raise ValueError(
                "pre-report stage cannot bind protected data partitions:"
                + ",".join(forbidden_partitions)
            )
        if stage is ResearchStage.FACTOR_GENERATION:
            observed_partitions = {
                role for role in bindings if role.startswith("data:partition:")
            }
            required_partitions = {
                "data:partition:discovery",
                "data:partition:train",
            }
            if observed_partitions != required_partitions:
                raise ValueError(
                    "factor-generation stage must bind only discovery/train partitions"
                )
        object.__setattr__(self, "component_bindings", MappingProxyType(bindings))
        parents = tuple(ResearchStage(item) for item in self.required_parent_stages)
        if len(parents) != len(set(parents)):
            raise ValueError("stage contract parent stages must be unique")
        current_position = list(ResearchStage).index(stage)
        if any(
            list(ResearchStage).index(parent) >= current_position for parent in parents
        ):
            raise ValueError("stage contract parent must precede the output stage")
        object.__setattr__(self, "required_parent_stages", parents)
        if not self.payload_schema.strip() or len(self.payload_schema) > 128:
            raise ValueError("stage contract payload schema is invalid")

    @classmethod
    def for_run(
        cls, run: ResearchRunSpec, stage: ResearchStage | str
    ) -> "StageContract":
        normalized = ResearchStage(stage)
        if normalized not in run.enabled_stages:
            raise ValueError(
                f"stage is not enabled by ResearchRunSpec:{normalized.value}"
            )
        bindings = _stage_bindings(run, normalized)
        parents = _stage_parents(run, normalized)
        return cls(
            research_run_spec_hash=run.content_hash,
            stage=normalized,
            component_bindings=bindings,
            required_parent_stages=parents,
            payload_schema=f"research-stage/{normalized.value}/v1",
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        stage = ResearchStage(self.stage)
        return {
            "schema_version": self.schema_version,
            "research_run_spec_hash": self.research_run_spec_hash,
            "stage": stage.value,
            "component_bindings": dict(self.component_bindings),
            "required_parent_stages": [
                ResearchStage(item).value for item in self.required_parent_stages
            ],
            "payload_schema": self.payload_schema,
        }

    def build_payload(
        self,
        payload: Mapping[str, object],
        *,
        parent_artifact_hashes: Mapping[ResearchStage | str, str],
    ) -> bytes:
        body = dict(payload)
        parent_hashes = self._normalize_parents(parent_artifact_hashes)
        payload_hash = hash_json(body)
        binding_hash = _binding_hash(
            payload_schema=self.payload_schema,
            research_run_spec_hash=self.research_run_spec_hash,
            stage=ResearchStage(self.stage),
            component_bindings=self.component_bindings,
            parent_artifact_hashes=parent_hashes,
            payload_hash=payload_hash,
        )
        envelope = {
            "schema_version": self.payload_schema,
            "research_run_spec_hash": self.research_run_spec_hash,
            "stage": ResearchStage(self.stage).value,
            "component_bindings": dict(self.component_bindings),
            "parent_artifact_hashes": {
                stage.value: digest for stage, digest in parent_hashes.items()
            },
            "payload": body,
            "payload_hash": payload_hash,
            "binding_hash": binding_hash,
        }
        return canonical_json_bytes(envelope)

    def validate_descriptor(self, descriptor: "StageArtifactDescriptor") -> None:
        if descriptor.contract_hash != self.content_hash:
            raise ValueError("stage artifact contract hash differs")
        if descriptor.research_run_spec_hash != self.research_run_spec_hash:
            raise ValueError("stage artifact research-run binding differs")
        if ResearchStage(descriptor.stage) is not ResearchStage(self.stage):
            raise ValueError("stage artifact stage differs")
        if descriptor.payload_schema != self.payload_schema:
            raise ValueError("stage artifact payload schema differs")
        if dict(descriptor.component_bindings) != dict(self.component_bindings):
            raise ValueError("stage artifact component bindings differ")
        if set(descriptor.parent_artifact_hashes) != set(self.required_parent_stages):
            raise ValueError("stage artifact parent dependency set differs")

    def _normalize_parents(
        self, values: Mapping[ResearchStage | str, str]
    ) -> Mapping[ResearchStage, str]:
        normalized = {ResearchStage(key): str(digest) for key, digest in values.items()}
        required = {ResearchStage(item) for item in self.required_parent_stages}
        if set(normalized) != required:
            missing = sorted(item.value for item in required.difference(normalized))
            extra = sorted(item.value for item in set(normalized).difference(required))
            raise ValueError(
                f"stage artifact parent dependencies differ:missing={missing},extra={extra}"
            )
        for stage, digest in normalized.items():
            require_sha256(digest, name=f"stage parent artifact:{stage.value}")
        return MappingProxyType(
            dict(sorted(normalized.items(), key=lambda item: item[0].value))
        )


@dataclass(frozen=True, slots=True)
class StageArtifactDescriptor:
    contract_hash: str
    research_run_spec_hash: str
    stage: ResearchStage | str
    artifact_hash: str
    payload_hash: str
    binding_hash: str
    payload_schema: str
    component_bindings: Mapping[str, str]
    parent_artifact_hashes: Mapping[ResearchStage, str]
    size_bytes: int
    media_type: str = "application/json"
    schema_version: str = "stage-artifact-descriptor/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "stage-artifact-descriptor/v1":
            raise ValueError("unsupported StageArtifactDescriptor schema")
        for name in (
            "contract_hash",
            "research_run_spec_hash",
            "artifact_hash",
            "payload_hash",
            "binding_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"stage artifact {name}")
        stage = ResearchStage(self.stage)
        object.__setattr__(self, "stage", stage)
        bindings = dict(sorted(self.component_bindings.items()))
        if not bindings:
            raise ValueError("stage artifact lacks component payload bindings")
        for role, digest in bindings.items():
            if not role.strip():
                raise ValueError("stage artifact component role is empty")
            require_sha256(digest, name=f"stage artifact component:{role}")
        object.__setattr__(self, "component_bindings", MappingProxyType(bindings))
        parents = {
            ResearchStage(parent): str(digest)
            for parent, digest in self.parent_artifact_hashes.items()
        }
        for parent, digest in parents.items():
            require_sha256(digest, name=f"stage artifact parent:{parent.value}")
        object.__setattr__(
            self,
            "parent_artifact_hashes",
            MappingProxyType(
                dict(sorted(parents.items(), key=lambda item: item[0].value))
            ),
        )
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes <= 0
        ):
            raise ValueError("stage artifact size_bytes must be positive")
        if self.media_type != "application/json":
            raise ValueError("typed stage artifacts require application/json")
        expected_binding = _binding_hash(
            payload_schema=self.payload_schema,
            research_run_spec_hash=self.research_run_spec_hash,
            stage=stage,
            component_bindings=self.component_bindings,
            parent_artifact_hashes=parents,
            payload_hash=self.payload_hash,
        )
        if self.binding_hash != expected_binding:
            raise ValueError("stage artifact payload binding hash differs")

    @classmethod
    def from_payload(
        cls, contract: StageContract, payload: bytes
    ) -> "StageArtifactDescriptor":
        if not isinstance(payload, bytes) or not payload:
            raise TypeError("typed stage artifact payload must be non-empty bytes")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("typed stage artifact payload is not valid JSON") from exc
        if not isinstance(decoded, Mapping) or set(decoded) != _ENVELOPE_FIELDS:
            raise ValueError("typed stage artifact envelope fields differ")
        if canonical_json_bytes(decoded) != payload:
            raise ValueError("typed stage artifact payload must be canonical JSON")
        if str(decoded["schema_version"]) != contract.payload_schema:
            raise ValueError("typed stage artifact payload schema differs")
        if str(decoded["research_run_spec_hash"]) != contract.research_run_spec_hash:
            raise ValueError("typed stage artifact research-run binding differs")
        if ResearchStage(str(decoded["stage"])) is not ResearchStage(contract.stage):
            raise ValueError("typed stage artifact stage differs")
        components = _string_mapping(
            decoded["component_bindings"], name="component_bindings"
        )
        if components != dict(contract.component_bindings):
            raise ValueError("typed stage artifact component bindings differ")
        raw_parents = _string_mapping(
            decoded["parent_artifact_hashes"], name="parent_artifact_hashes"
        )
        parents = contract._normalize_parents(raw_parents)
        body = decoded["payload"]
        if not isinstance(body, Mapping):
            raise TypeError("typed stage artifact body must be an object")
        payload_hash = str(decoded["payload_hash"])
        require_sha256(payload_hash, name="typed stage artifact payload_hash")
        if hash_json(dict(body)) != payload_hash:
            raise ValueError("typed stage artifact body hash differs")
        binding_hash = str(decoded["binding_hash"])
        require_sha256(binding_hash, name="typed stage artifact binding_hash")
        expected_binding = _binding_hash(
            payload_schema=contract.payload_schema,
            research_run_spec_hash=contract.research_run_spec_hash,
            stage=ResearchStage(contract.stage),
            component_bindings=components,
            parent_artifact_hashes=parents,
            payload_hash=payload_hash,
        )
        if binding_hash != expected_binding:
            raise ValueError("typed stage artifact binding hash differs")
        descriptor = cls(
            contract_hash=contract.content_hash,
            research_run_spec_hash=contract.research_run_spec_hash,
            stage=contract.stage,
            artifact_hash=hashlib.sha256(payload).hexdigest(),
            payload_hash=payload_hash,
            binding_hash=binding_hash,
            payload_schema=contract.payload_schema,
            component_bindings=components,
            parent_artifact_hashes=parents,
            size_bytes=len(payload),
        )
        contract.validate_descriptor(descriptor)
        return descriptor

    def verify_payload(self, contract: StageContract, payload: bytes) -> None:
        observed = type(self).from_payload(contract, payload)
        if observed.content_hash != self.content_hash:
            raise RuntimeError("stage artifact payload differs from descriptor")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_hash": self.contract_hash,
            "research_run_spec_hash": self.research_run_spec_hash,
            "stage": ResearchStage(self.stage).value,
            "artifact_hash": self.artifact_hash,
            "payload_hash": self.payload_hash,
            "binding_hash": self.binding_hash,
            "payload_schema": self.payload_schema,
            "component_bindings": dict(self.component_bindings),
            "parent_artifact_hashes": {
                ResearchStage(stage).value: digest
                for stage, digest in self.parent_artifact_hashes.items()
            },
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }


def _binding_hash(
    *,
    payload_schema: str,
    research_run_spec_hash: str,
    stage: ResearchStage,
    component_bindings: Mapping[str, str],
    parent_artifact_hashes: Mapping[ResearchStage, str],
    payload_hash: str,
) -> str:
    return hash_json(
        {
            "payload_schema": payload_schema,
            "research_run_spec_hash": research_run_spec_hash,
            "stage": stage.value,
            "component_bindings": dict(component_bindings),
            "parent_artifact_hashes": {
                parent.value: digest
                for parent, digest in parent_artifact_hashes.items()
            },
            "payload_hash": payload_hash,
        }
    )


def _stage_bindings(run: ResearchRunSpec, stage: ResearchStage) -> Mapping[str, str]:
    all_bindings = dict(run.component_bindings())
    partition_roles = {
        role for role in all_bindings if role.startswith("data:partition:")
    }
    common_data_roles = {
        role
        for role in all_bindings
        if role.startswith("data:") and role not in partition_roles
    }
    development_partitions = {
        role
        for role in partition_roles
        if role
        in {
            "data:partition:discovery",
            "data:partition:train",
            "data:partition:validation",
        }
    }
    generation_partitions = {
        role
        for role in partition_roles
        if role in {"data:partition:discovery", "data:partition:train"}
    }
    factor_roles = {role for role in all_bindings if role.startswith("factor:")}
    roles: set[str]
    if stage is ResearchStage.DATA_QUALITY:
        roles = common_data_roles | development_partitions
    elif stage is ResearchStage.FACTOR_GENERATION:
        roles = common_data_roles | generation_partitions | factor_roles
    elif stage is ResearchStage.LABEL_BUILDING:
        roles = common_data_roles | development_partitions | {"label"}
    elif stage is ResearchStage.VALIDATION_SPLIT:
        roles = {"label", "validation"}
    elif stage is ResearchStage.MODEL_TRAINING:
        roles = factor_roles | {"label", "validation", "model"}
    elif stage is ResearchStage.FACTOR_EVALUATION:
        roles = factor_roles | {"label", "validation", "evaluation"}
    elif stage is ResearchStage.SCORE_CONSTRUCTION:
        roles = factor_roles | {"evaluation", "score"}
        if "model" in all_bindings:
            roles.add("model")
    elif stage is ResearchStage.PORTFOLIO_CONSTRUCTION:
        roles = {"score", "portfolio"}
        if "risk" in all_bindings:
            roles.add("risk")
    elif stage is ResearchStage.BACKTEST:
        roles = {"portfolio", "cost", "execution", "backtest"}
        if "risk" in all_bindings:
            roles.add("risk")
    elif stage is ResearchStage.ROBUSTNESS:
        governed_model_readiness = {
            ResearchStage.MODEL_TRAINING,
            ResearchStage.SCORE_CONSTRUCTION,
            ResearchStage.ROBUSTNESS,
        }.issubset(run.enabled_stages)
        if governed_model_readiness:
            roles = common_data_roles | {
                "data:partition:validation",
                "label",
                "validation",
                "model",
                "score",
                "backtest",
                "robustness",
            }
        else:
            roles = {"backtest", "robustness"}
    elif stage is ResearchStage.REPORT:
        roles = set(all_bindings).difference(
            {
                "admission",
                "data:partition:test",
                "data:partition:holdout",
            }
        )
    else:
        roles = {"report", "admission"} | {
            role
            for role in partition_roles
            if role in {"data:partition:test", "data:partition:holdout"}
        }
        for optional in ("evaluation", "backtest", "robustness"):
            if optional in all_bindings:
                roles.add(optional)
    missing = sorted(roles.difference(all_bindings))
    if missing:  # pragma: no cover - guarded by ResearchRunSpec optional-chain rules
        raise RuntimeError(
            "stage component bindings are unavailable:" + ",".join(missing)
        )
    return MappingProxyType({role: all_bindings[role] for role in sorted(roles)})


def _stage_parents(
    run: ResearchRunSpec, stage: ResearchStage
) -> tuple[ResearchStage, ...]:
    fixed = {
        ResearchStage.DATA_QUALITY: (),
        ResearchStage.FACTOR_GENERATION: (ResearchStage.DATA_QUALITY,),
        ResearchStage.LABEL_BUILDING: (ResearchStage.DATA_QUALITY,),
        ResearchStage.VALIDATION_SPLIT: (ResearchStage.LABEL_BUILDING,),
        ResearchStage.MODEL_TRAINING: (
            ResearchStage.FACTOR_GENERATION,
            ResearchStage.LABEL_BUILDING,
            ResearchStage.VALIDATION_SPLIT,
        ),
        ResearchStage.FACTOR_EVALUATION: (
            ResearchStage.FACTOR_GENERATION,
            ResearchStage.LABEL_BUILDING,
            ResearchStage.VALIDATION_SPLIT,
        ),
        ResearchStage.SCORE_CONSTRUCTION: (ResearchStage.FACTOR_EVALUATION,),
        ResearchStage.PORTFOLIO_CONSTRUCTION: (ResearchStage.SCORE_CONSTRUCTION,),
        ResearchStage.BACKTEST: (ResearchStage.PORTFOLIO_CONSTRUCTION,),
        ResearchStage.ROBUSTNESS: (ResearchStage.BACKTEST,),
        ResearchStage.ADMISSION: (ResearchStage.REPORT,),
    }
    if stage in fixed:
        parents = list(fixed[stage])
        if (
            stage is ResearchStage.SCORE_CONSTRUCTION
            and ResearchStage.MODEL_TRAINING in run.enabled_stages
        ):
            parents.append(ResearchStage.MODEL_TRAINING)
        if (
            stage is ResearchStage.ROBUSTNESS
            and {
                ResearchStage.MODEL_TRAINING,
                ResearchStage.SCORE_CONSTRUCTION,
                ResearchStage.ROBUSTNESS,
            }.issubset(run.enabled_stages)
        ):
            parents = [
                ResearchStage.MODEL_TRAINING,
                ResearchStage.SCORE_CONSTRUCTION,
                ResearchStage.BACKTEST,
            ]
        return tuple(parents)
    terminals: list[ResearchStage] = [ResearchStage.FACTOR_EVALUATION]
    downstream = (
        ResearchStage.SCORE_CONSTRUCTION,
        ResearchStage.PORTFOLIO_CONSTRUCTION,
        ResearchStage.BACKTEST,
        ResearchStage.ROBUSTNESS,
    )
    enabled_downstream = [item for item in downstream if item in run.enabled_stages]
    if enabled_downstream:
        terminals = [enabled_downstream[-1]]
    if (
        ResearchStage.MODEL_TRAINING in run.enabled_stages
        and ResearchStage.SCORE_CONSTRUCTION not in run.enabled_stages
    ):
        terminals.append(ResearchStage.MODEL_TRAINING)
    return tuple(terminals)


def _string_mapping(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise TypeError(f"typed stage artifact {name} must be a string object")
    return {str(key): str(item) for key, item in value.items()}


__all__ = ["StageArtifactDescriptor", "StageContract"]
