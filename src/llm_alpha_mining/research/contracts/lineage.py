"""Typed scientific lineage for one governed research assembly.

The manifest deliberately stays separate from ``ResearchRunSpec``.  It closes
the scientific attribution boundary without changing the stable run-spec wire
format or introducing another registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

from llm_alpha_mining.research.core.hashing import hash_json, require_sha256
from llm_alpha_mining.research.contracts.spec import ResearchRunSpec


@dataclass(frozen=True, slots=True)
class FactorLogicBindingRef:
    """Exact MarketLogicRegistry binding selected for one FactorSpec."""

    factor_spec_hash: str
    market_logic_hash: str
    registry_binding_hash: str
    schema_version: str = "factor-logic-binding-ref/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "factor-logic-binding-ref/v1":
            raise ValueError("unsupported FactorLogicBindingRef schema")
        for name in (
            "factor_spec_hash",
            "market_logic_hash",
            "registry_binding_hash",
        ):
            require_sha256(
                str(getattr(self, name)),
                name=f"scientific lineage {name}",
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "factor_spec_hash": self.factor_spec_hash,
            "market_logic_hash": self.market_logic_hash,
            "registry_binding_hash": self.registry_binding_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FactorLogicBindingRef":
        expected = {
            "schema_version",
            "factor_spec_hash",
            "market_logic_hash",
            "registry_binding_hash",
        }
        if set(value) != expected or not all(
            isinstance(value[name], str) for name in expected
        ):
            raise ValueError("FactorLogicBindingRef wire fields differ")
        return cls(
            schema_version=str(value["schema_version"]),
            factor_spec_hash=str(value["factor_spec_hash"]),
            market_logic_hash=str(value["market_logic_hash"]),
            registry_binding_hash=str(value["registry_binding_hash"]),
        )


@dataclass(frozen=True, slots=True)
class ModelFeatureBinding:
    """Declares which FactorSpec produced one exact model feature frame."""

    feature_name: str
    factor_spec_hash: str
    signal_hash: str
    schema_version: str = "model-feature-binding/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "model-feature-binding/v1":
            raise ValueError("unsupported ModelFeatureBinding schema")
        if (
            not isinstance(self.feature_name, str)
            or not self.feature_name.strip()
            or self.feature_name != self.feature_name.strip()
        ):
            raise ValueError("model feature binding name is invalid")
        require_sha256(
            self.factor_spec_hash,
            name=f"model feature factor:{self.feature_name}",
        )
        require_sha256(
            self.signal_hash,
            name=f"model feature signal:{self.feature_name}",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "feature_name": self.feature_name,
            "factor_spec_hash": self.factor_spec_hash,
            "signal_hash": self.signal_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ModelFeatureBinding":
        expected = {
            "schema_version",
            "feature_name",
            "factor_spec_hash",
            "signal_hash",
        }
        if set(value) != expected or not all(
            isinstance(value[name], str) for name in expected
        ):
            raise ValueError("ModelFeatureBinding wire fields differ")
        return cls(
            schema_version=str(value["schema_version"]),
            feature_name=str(value["feature_name"]),
            factor_spec_hash=str(value["factor_spec_hash"]),
            signal_hash=str(value["signal_hash"]),
        )


@dataclass(frozen=True, slots=True)
class ResearchScientificLineageManifestV1:
    """Content-addressed MarketLogic -> FactorSpec -> model-feature lineage."""

    research_run_spec_hash: str
    factor_logic_bindings: tuple[FactorLogicBindingRef, ...]
    model_spec_hash: str | None = None
    model_feature_bindings: tuple[ModelFeatureBinding, ...] = ()
    schema_version: str = "research-scientific-lineage-manifest/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "research-scientific-lineage-manifest/v1":
            raise ValueError("unsupported ResearchScientificLineageManifestV1 schema")
        require_sha256(
            self.research_run_spec_hash,
            name="scientific lineage research_run_spec_hash",
        )
        raw_factors = tuple(self.factor_logic_bindings)
        if not raw_factors or any(
            not isinstance(item, FactorLogicBindingRef) for item in raw_factors
        ):
            raise TypeError("scientific lineage requires FactorLogicBindingRef entries")
        factors = tuple(
            sorted(
                raw_factors,
                key=lambda item: item.factor_spec_hash,
            )
        )
        factor_hashes = tuple(item.factor_spec_hash for item in factors)
        binding_hashes = tuple(item.registry_binding_hash for item in factors)
        if len(set(factor_hashes)) != len(factor_hashes):
            raise ValueError("scientific lineage FactorSpec hashes must be unique")
        if len(set(binding_hashes)) != len(binding_hashes):
            raise ValueError(
                "scientific lineage registry binding hashes must be unique"
            )

        raw_features = tuple(self.model_feature_bindings)
        if any(not isinstance(item, ModelFeatureBinding) for item in raw_features):
            raise TypeError("scientific lineage requires ModelFeatureBinding entries")
        features = tuple(
            sorted(
                raw_features,
                key=lambda item: item.feature_name,
            )
        )
        feature_names = tuple(item.feature_name for item in features)
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("scientific lineage model feature names must be unique")
        if self.model_spec_hash is None:
            if features:
                raise ValueError("model feature bindings require a model specification")
        else:
            require_sha256(
                self.model_spec_hash,
                name="scientific lineage model_spec_hash",
            )
            if not features:
                raise ValueError("model specification requires model feature bindings")
        signal_by_factor: dict[str, str] = {}
        for feature in features:
            prior = signal_by_factor.setdefault(
                feature.factor_spec_hash, feature.signal_hash
            )
            if prior != feature.signal_hash:
                raise ValueError(
                    "one FactorSpec cannot bind multiple model signal hashes"
                )
        object.__setattr__(self, "factor_logic_bindings", factors)
        object.__setattr__(self, "model_feature_bindings", features)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    @property
    def market_logic_hashes(self) -> tuple[str, ...]:
        return tuple(
            sorted({item.market_logic_hash for item in self.factor_logic_bindings})
        )

    def validate_for(self, run_spec: ResearchRunSpec) -> None:
        if not isinstance(run_spec, ResearchRunSpec):
            raise TypeError("scientific lineage requires ResearchRunSpec")
        if self.research_run_spec_hash != run_spec.content_hash:
            raise ValueError("scientific lineage research run differs")
        if {item.factor_spec_hash for item in self.factor_logic_bindings} != set(
            run_spec.factor_spec_hashes
        ):
            raise ValueError(
                "scientific lineage must bind every run FactorSpec exactly once"
            )
        if self.model_spec_hash != run_spec.model_spec_hash:
            raise ValueError("scientific lineage model specification differs")
        run_factors = set(run_spec.factor_spec_hashes)
        unknown = sorted(
            {item.factor_spec_hash for item in self.model_feature_bindings}.difference(
                run_factors
            )
        )
        if unknown:
            raise ValueError(
                "scientific lineage model features reference unknown FactorSpecs"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "research_run_spec_hash": self.research_run_spec_hash,
            "factor_logic_bindings": [
                item.to_dict() for item in self.factor_logic_bindings
            ],
            "model_spec_hash": self.model_spec_hash,
            "model_feature_bindings": [
                item.to_dict() for item in self.model_feature_bindings
            ],
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchScientificLineageManifestV1":
        expected = {
            "schema_version",
            "research_run_spec_hash",
            "factor_logic_bindings",
            "model_spec_hash",
            "model_feature_bindings",
        }
        if set(value) != expected:
            raise ValueError("ResearchScientificLineageManifestV1 wire fields differ")
        raw_factors = value["factor_logic_bindings"]
        raw_features = value["model_feature_bindings"]
        if not isinstance(raw_factors, list) or not all(
            isinstance(item, Mapping) for item in raw_factors
        ):
            raise TypeError("scientific lineage factor bindings must be objects")
        if not isinstance(raw_features, list) or not all(
            isinstance(item, Mapping) for item in raw_features
        ):
            raise TypeError("scientific lineage model features must be objects")
        model_hash = value["model_spec_hash"]
        if model_hash is not None and not isinstance(model_hash, str):
            raise TypeError("scientific lineage model_spec_hash must be a string")
        return cls(
            schema_version=str(value["schema_version"]),
            research_run_spec_hash=str(value["research_run_spec_hash"]),
            factor_logic_bindings=tuple(
                FactorLogicBindingRef.from_mapping(item) for item in raw_factors
            ),
            model_spec_hash=model_hash,
            model_feature_bindings=tuple(
                ModelFeatureBinding.from_mapping(item) for item in raw_features
            ),
        )


@dataclass(frozen=True, slots=True)
class ResearchScientificLineageManifestV2:
    """V1 scientific lineage plus the exact nested-selection search identity.

    The wrapper preserves the published V1 wire while making the candidate
    grid, fold policy, tie-break and fit budget part of the ExperimentSpec
    identity before any model-stage work can be registered.
    """

    base_manifest: ResearchScientificLineageManifestV1
    nested_selection_spec_hash: str
    schema_version: str = "research-scientific-lineage-manifest/v2"

    def __post_init__(self) -> None:
        if self.schema_version != "research-scientific-lineage-manifest/v2":
            raise ValueError("unsupported ResearchScientificLineageManifestV2 schema")
        if not isinstance(self.base_manifest, ResearchScientificLineageManifestV1):
            raise TypeError("V2 scientific lineage requires a V1 base manifest")
        if self.base_manifest.model_spec_hash is None:
            raise ValueError("nested-selection lineage requires a model specification")
        require_sha256(
            self.nested_selection_spec_hash,
            name="scientific lineage nested_selection_spec_hash",
        )

    @property
    def research_run_spec_hash(self) -> str:
        return self.base_manifest.research_run_spec_hash

    @property
    def factor_logic_bindings(self) -> tuple[FactorLogicBindingRef, ...]:
        return self.base_manifest.factor_logic_bindings

    @property
    def model_spec_hash(self) -> str:
        model_hash = self.base_manifest.model_spec_hash
        if model_hash is None:  # pragma: no cover - guarded in __post_init__.
            raise RuntimeError("nested-selection model specification disappeared")
        return model_hash

    @property
    def model_feature_bindings(self) -> tuple[ModelFeatureBinding, ...]:
        return self.base_manifest.model_feature_bindings

    @property
    def market_logic_hashes(self) -> tuple[str, ...]:
        return self.base_manifest.market_logic_hashes

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def validate_for(self, run_spec: ResearchRunSpec) -> None:
        self.base_manifest.validate_for(run_spec)
        if run_spec.model_spec_hash is None:
            raise ValueError("nested-selection lineage requires a model run")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "base_manifest": self.base_manifest.to_dict(),
            "nested_selection_spec_hash": self.nested_selection_spec_hash,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "ResearchScientificLineageManifestV2":
        expected = {
            "schema_version",
            "base_manifest",
            "nested_selection_spec_hash",
        }
        if set(value) != expected:
            raise ValueError("ResearchScientificLineageManifestV2 wire fields differ")
        raw_base = value["base_manifest"]
        nested_hash = value["nested_selection_spec_hash"]
        if not isinstance(raw_base, Mapping):
            raise TypeError("V2 scientific lineage base manifest must be an object")
        if not isinstance(nested_hash, str):
            raise TypeError(
                "V2 scientific lineage nested_selection_spec_hash must be a string"
            )
        return cls(
            schema_version=str(value["schema_version"]),
            base_manifest=ResearchScientificLineageManifestV1.from_mapping(raw_base),
            nested_selection_spec_hash=nested_hash,
        )


ResearchScientificLineageManifest: TypeAlias = (
    ResearchScientificLineageManifestV1 | ResearchScientificLineageManifestV2
)


__all__ = [
    "FactorLogicBindingRef",
    "ModelFeatureBinding",
    "ResearchScientificLineageManifest",
    "ResearchScientificLineageManifestV1",
    "ResearchScientificLineageManifestV2",
]
