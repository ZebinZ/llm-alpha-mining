from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.high_frequency.provider_semantics import (
    DataCapability,
    IntendedUse,
    ProviderRegimeSpec,
    ProviderSemanticCatalog,
    SemanticAdmissionError,
    SourceKind,
)
from alpha_research.market_logic.spec import (
    CapabilityRequirement,
    ConditionRelation,
    DataDomain,
    MarketLogicSpec,
    WindowUnit,
    iter_predicates,
)


_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class LogicCompilationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ObservableDefinition:
    """Trusted compiler-side meaning of one non-executable observable code."""

    observable: str
    required_fields: tuple[str, ...]
    allowed_fields: tuple[str, ...]
    allowed_operators: tuple[str, ...]
    allowed_aggregations: tuple[str, ...]
    allowed_relations: tuple[ConditionRelation | str, ...]
    required_capabilities: tuple[CapabilityRequirement, ...]

    def __post_init__(self) -> None:
        _require_code(self.observable, name="observable definition code")
        required = _normalize_codes(
            self.required_fields, field=True, name="observable required fields"
        )
        allowed = _normalize_codes(
            self.allowed_fields, field=True, name="observable allowed fields"
        )
        if not required or not set(required).issubset(allowed):
            raise ValueError(
                "observable required fields must be a non-empty allowed subset"
            )
        operators = _normalize_codes(
            self.allowed_operators, field=False, name="observable allowed operators"
        )
        aggregations = _normalize_codes(
            self.allowed_aggregations,
            field=False,
            name="observable allowed aggregations",
        )
        relations = tuple(
            sorted(
                {ConditionRelation(item) for item in self.allowed_relations},
                key=lambda item: item.value,
            )
        )
        if not operators or not aggregations or not relations:
            raise ValueError(
                "observable operators, aggregations and relations are required"
            )
        capabilities = tuple(
            sorted(self.required_capabilities, key=lambda item: item.key)
        )
        if not capabilities or len({item.key for item in capabilities}) != len(
            capabilities
        ):
            raise ValueError("observable capabilities must be non-empty and unique")
        object.__setattr__(self, "required_fields", required)
        object.__setattr__(self, "allowed_fields", allowed)
        object.__setattr__(self, "allowed_operators", operators)
        object.__setattr__(self, "allowed_aggregations", aggregations)
        object.__setattr__(self, "allowed_relations", relations)
        object.__setattr__(self, "required_capabilities", capabilities)

    def to_dict(self) -> dict[str, object]:
        return {
            "observable": self.observable,
            "required_fields": list(self.required_fields),
            "allowed_fields": list(self.allowed_fields),
            "allowed_operators": list(self.allowed_operators),
            "allowed_aggregations": list(self.allowed_aggregations),
            "allowed_relations": [
                ConditionRelation(item).value for item in self.allowed_relations
            ],
            "required_capabilities": [
                item.to_dict() for item in self.required_capabilities
            ],
        }


@dataclass(frozen=True, slots=True)
class CompilerComplexityPolicy:
    maximum_ast_nodes: int = 64
    maximum_call_depth: int = 6
    maximum_fields: int = 12
    maximum_operators: int = 24
    maximum_rolling_windows: int = 8
    maximum_window: int = 252

    def __post_init__(self) -> None:
        for name in (
            "maximum_ast_nodes",
            "maximum_call_depth",
            "maximum_fields",
            "maximum_operators",
            "maximum_rolling_windows",
            "maximum_window",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"compiler complexity {name} must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_ast_nodes": self.maximum_ast_nodes,
            "maximum_call_depth": self.maximum_call_depth,
            "maximum_fields": self.maximum_fields,
            "maximum_operators": self.maximum_operators,
            "maximum_rolling_windows": self.maximum_rolling_windows,
            "maximum_window": self.maximum_window,
        }


@dataclass(frozen=True, slots=True)
class LogicCompilerPolicy:
    policy_id: str
    version: str
    observables: tuple[ObservableDefinition, ...]
    forbidden_operators: tuple[str, ...]
    complexity: CompilerComplexityPolicy
    schema_version: str = "logic-compiler-policy/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "logic-compiler-policy/v1":
            raise ValueError("unsupported LogicCompilerPolicy schema")
        _require_code(self.policy_id, name="logic compiler policy_id")
        _require_code(self.version, name="logic compiler version")
        observables = tuple(sorted(self.observables, key=lambda item: item.observable))
        names = [item.observable for item in observables]
        if not names or len(names) != len(set(names)):
            raise ValueError(
                "compiler observable definitions must be non-empty and unique"
            )
        forbidden = _normalize_codes(
            self.forbidden_operators,
            field=False,
            name="compiler forbidden operators",
        )
        admitted = {
            operator
            for definition in observables
            for operator in definition.allowed_operators
        }
        overlap = sorted(admitted.intersection(forbidden))
        if overlap:
            raise ValueError(
                "compiler operators are both allowed and forbidden:" + ",".join(overlap)
            )
        object.__setattr__(self, "observables", observables)
        object.__setattr__(self, "forbidden_operators", forbidden)

    @property
    def observable_map(self) -> Mapping[str, ObservableDefinition]:
        return MappingProxyType({item.observable: item for item in self.observables})

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "version": self.version,
            "observables": [item.to_dict() for item in self.observables],
            "forbidden_operators": list(self.forbidden_operators),
            "complexity": self.complexity.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CompiledWindowBound:
    predicate_hash: str
    observable: str
    minimum: int
    maximum: int
    unit: WindowUnit | str

    def __post_init__(self) -> None:
        require_sha256(self.predicate_hash, name="compiled predicate hash")
        _require_code(self.observable, name="compiled window observable")
        object.__setattr__(self, "unit", WindowUnit(self.unit))
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"compiled window {name} must be positive")
        if self.minimum > self.maximum:
            raise ValueError("compiled window bound is inverted")

    def to_dict(self) -> dict[str, object]:
        unit = self.unit
        if not isinstance(unit, WindowUnit):  # pragma: no cover
            raise RuntimeError("compiled window unit was not normalized")
        return {
            "predicate_hash": self.predicate_hash,
            "observable": self.observable,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unit": unit.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CompiledWindowBound":
        _exact_keys(
            value,
            {"predicate_hash", "observable", "minimum", "maximum", "unit"},
            path="window_bound",
        )
        return cls(
            predicate_hash=_string(value["predicate_hash"], path="predicate_hash"),
            observable=_string(value["observable"], path="observable"),
            minimum=_integer(value["minimum"], path="minimum"),
            maximum=_integer(value["maximum"], path="maximum"),
            unit=_string(value["unit"], path="unit"),
        )


@dataclass(frozen=True, slots=True)
class LogicComplexitySpec:
    maximum_ast_nodes: int
    maximum_call_depth: int
    maximum_fields: int
    maximum_operators: int
    maximum_rolling_windows: int

    def __post_init__(self) -> None:
        for name in (
            "maximum_ast_nodes",
            "maximum_call_depth",
            "maximum_fields",
            "maximum_operators",
            "maximum_rolling_windows",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"logic complexity {name} must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_ast_nodes": self.maximum_ast_nodes,
            "maximum_call_depth": self.maximum_call_depth,
            "maximum_fields": self.maximum_fields,
            "maximum_operators": self.maximum_operators,
            "maximum_rolling_windows": self.maximum_rolling_windows,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LogicComplexitySpec":
        expected = {
            "maximum_ast_nodes",
            "maximum_call_depth",
            "maximum_fields",
            "maximum_operators",
            "maximum_rolling_windows",
        }
        _exact_keys(value, expected, path="logic_complexity")
        return cls(**{name: _integer(value[name], path=name) for name in expected})


@dataclass(frozen=True, slots=True)
class LogicConstraintSpec:
    """Executable search envelope emitted only by the deterministic compiler."""

    market_logic_hash: str
    compiler_policy_hash: str
    semantic_catalog_hash: str
    eligible_provider_regime_hashes: tuple[str, ...]
    required_fields: tuple[str, ...]
    allowed_fields: tuple[str, ...]
    allowed_operators: tuple[str, ...]
    forbidden_operators: tuple[str, ...]
    window_bounds: tuple[CompiledWindowBound, ...]
    allowed_aggregations: tuple[str, ...]
    complexity: LogicComplexitySpec
    schema_version: str = "logic-constraint-spec/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "logic-constraint-spec/v1":
            raise ValueError("unsupported LogicConstraintSpec schema")
        for name in (
            "market_logic_hash",
            "compiler_policy_hash",
            "semantic_catalog_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"logic constraint {name}")
        regimes = _normalize_hashes(
            self.eligible_provider_regime_hashes, name="eligible provider regime hashes"
        )
        required = _normalize_codes(
            self.required_fields, field=True, name="constraint required fields"
        )
        allowed = _normalize_codes(
            self.allowed_fields, field=True, name="constraint allowed fields"
        )
        operators = _normalize_codes(
            self.allowed_operators, field=False, name="constraint allowed operators"
        )
        forbidden = _normalize_codes(
            self.forbidden_operators,
            field=False,
            name="constraint forbidden operators",
        )
        aggregations = _normalize_codes(
            self.allowed_aggregations,
            field=False,
            name="constraint allowed aggregations",
        )
        if not required or not set(required).issubset(allowed):
            raise ValueError("constraint required fields must be an allowed subset")
        if not operators or not aggregations:
            raise ValueError("constraint operators and aggregations must be non-empty")
        if set(operators).intersection(forbidden):
            raise ValueError(
                "constraint operators cannot be both allowed and forbidden"
            )
        bounds = tuple(sorted(self.window_bounds, key=lambda item: item.predicate_hash))
        if not bounds or len({item.predicate_hash for item in bounds}) != len(bounds):
            raise ValueError("compiled window bounds must be non-empty and unique")
        object.__setattr__(self, "eligible_provider_regime_hashes", regimes)
        object.__setattr__(self, "required_fields", required)
        object.__setattr__(self, "allowed_fields", allowed)
        object.__setattr__(self, "allowed_operators", operators)
        object.__setattr__(self, "forbidden_operators", forbidden)
        object.__setattr__(self, "allowed_aggregations", aggregations)
        object.__setattr__(self, "window_bounds", bounds)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "market_logic_hash": self.market_logic_hash,
            "compiler_policy_hash": self.compiler_policy_hash,
            "semantic_catalog_hash": self.semantic_catalog_hash,
            "eligible_provider_regime_hashes": list(
                self.eligible_provider_regime_hashes
            ),
            "required_fields": list(self.required_fields),
            "allowed_fields": list(self.allowed_fields),
            "allowed_operators": list(self.allowed_operators),
            "forbidden_operators": list(self.forbidden_operators),
            "window_bounds": [item.to_dict() for item in self.window_bounds],
            "allowed_aggregations": list(self.allowed_aggregations),
            "complexity": self.complexity.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LogicConstraintSpec":
        expected = {
            "schema_version",
            "market_logic_hash",
            "compiler_policy_hash",
            "semantic_catalog_hash",
            "eligible_provider_regime_hashes",
            "required_fields",
            "allowed_fields",
            "allowed_operators",
            "forbidden_operators",
            "window_bounds",
            "allowed_aggregations",
            "complexity",
        }
        _exact_keys(value, expected, path="logic_constraint")
        return cls(
            schema_version=_string(value["schema_version"], path="schema_version"),
            market_logic_hash=_string(
                value["market_logic_hash"], path="market_logic_hash"
            ),
            compiler_policy_hash=_string(
                value["compiler_policy_hash"], path="compiler_policy_hash"
            ),
            semantic_catalog_hash=_string(
                value["semantic_catalog_hash"], path="semantic_catalog_hash"
            ),
            eligible_provider_regime_hashes=_strings(
                value["eligible_provider_regime_hashes"],
                path="eligible_provider_regime_hashes",
            ),
            required_fields=_strings(value["required_fields"], path="required_fields"),
            allowed_fields=_strings(value["allowed_fields"], path="allowed_fields"),
            allowed_operators=_strings(
                value["allowed_operators"], path="allowed_operators"
            ),
            forbidden_operators=_strings(
                value["forbidden_operators"], path="forbidden_operators"
            ),
            window_bounds=tuple(
                CompiledWindowBound.from_mapping(item)
                for item in _mapping_list(value["window_bounds"], path="window_bounds")
            ),
            allowed_aggregations=_strings(
                value["allowed_aggregations"], path="allowed_aggregations"
            ),
            complexity=LogicComplexitySpec.from_mapping(
                _mapping(value["complexity"], path="complexity")
            ),
        )


class MarketLogicCompiler:
    """Compile market logic through trusted policy and provider capabilities.

    The input type intentionally has no executable-constraint fields.  Operator,
    field, aggregation and complexity allowances originate solely from this
    compiler's versioned policy.
    """

    def __init__(self, policy: LogicCompilerPolicy | None = None) -> None:
        self.policy = policy or default_logic_compiler_policy()

    def compile(
        self,
        logic: MarketLogicSpec,
        *,
        semantic_catalog: ProviderSemanticCatalog,
        intended_use: IntendedUse | str = IntendedUse.RESEARCH,
    ) -> LogicConstraintSpec:
        if not isinstance(logic, MarketLogicSpec):
            raise TypeError("compiler accepts only a validated MarketLogicSpec")
        use = IntendedUse(intended_use)
        if logic.semantic_catalog_hash != semantic_catalog.content_hash:
            raise SemanticAdmissionError(
                "market_logic_semantic_catalog_hash_mismatch:"
                f"logic={logic.semantic_catalog_hash}:catalog={semantic_catalog.content_hash}"
            )
        profiles = self._eligible_profiles(logic, semantic_catalog)
        predicates = iter_predicates(logic.hypothesis.condition)
        observable_names = {predicate.observable for predicate in predicates} | {
            cast(str, predicate.reference_observable)
            for predicate in predicates
            if predicate.reference_observable is not None
        }
        unknown = sorted(observable_names.difference(self.policy.observable_map))
        if unknown:
            raise LogicCompilationError(
                "market logic references unregistered observables:" + ",".join(unknown)
            )
        definitions = tuple(
            self.policy.observable_map[name] for name in sorted(observable_names)
        )
        for predicate in predicates:
            definition = self.policy.observable_map[predicate.observable]
            relation = ConditionRelation(predicate.relation)
            if relation not in set(definition.allowed_relations):
                raise LogicCompilationError(
                    "observable relation is not compiler-admitted:"
                    f"{predicate.observable}:{relation.value}"
                )
            if predicate.window.maximum > self.policy.complexity.maximum_window:
                raise LogicCompilationError(
                    "condition window exceeds compiler policy:"
                    f"{predicate.observable}:{predicate.window.maximum}"
                )

        declared = {item.key: item for item in logic.capability_requirements}
        inferred = {
            item.key: item
            for definition in definitions
            for item in definition.required_capabilities
        }
        missing = sorted(set(inferred).difference(declared))
        if missing:
            raise LogicCompilationError(
                "market logic omitted compiler-inferred capabilities:"
                + ",".join("/".join(item) for item in missing)
            )
        self._gate_capabilities(
            tuple(declared.values()), profiles=profiles, intended_use=use
        )

        required_fields = tuple(
            sorted({field for item in definitions for field in item.required_fields})
        )
        allowed_fields = tuple(
            sorted({field for item in definitions for field in item.allowed_fields})
        )
        if len(allowed_fields) > self.policy.complexity.maximum_fields:
            raise LogicCompilationError(
                "compiled allowed fields exceed compiler complexity policy"
            )
        allowed_operators = tuple(
            sorted(
                {
                    operator
                    for item in definitions
                    for operator in item.allowed_operators
                }
            )
        )
        aggregation_sets = [set(item.allowed_aggregations) for item in definitions]
        allowed_aggregations = tuple(sorted(set.intersection(*aggregation_sets)))
        if not allowed_aggregations:
            raise LogicCompilationError(
                "observable definitions have no common admitted aggregation"
            )
        window_bounds = tuple(
            CompiledWindowBound(
                predicate_hash=predicate.content_hash,
                observable=predicate.observable,
                minimum=predicate.window.minimum,
                maximum=predicate.window.maximum,
                unit=predicate.window.unit,
            )
            for predicate in predicates
        )
        complexity = self._compile_complexity(
            predicate_count=len(predicates),
            field_count=len(allowed_fields),
            operator_count=len(allowed_operators),
            window_count=len(window_bounds),
        )
        return LogicConstraintSpec(
            market_logic_hash=logic.content_hash,
            compiler_policy_hash=self.policy.content_hash,
            semantic_catalog_hash=semantic_catalog.content_hash,
            eligible_provider_regime_hashes=logic.eligible_provider_regime_hashes,
            required_fields=required_fields,
            allowed_fields=allowed_fields,
            allowed_operators=allowed_operators,
            forbidden_operators=self.policy.forbidden_operators,
            window_bounds=window_bounds,
            allowed_aggregations=allowed_aggregations,
            complexity=complexity,
        )

    @staticmethod
    def _eligible_profiles(
        logic: MarketLogicSpec, catalog: ProviderSemanticCatalog
    ) -> tuple[ProviderRegimeSpec, ...]:
        by_hash: dict[str, ProviderRegimeSpec] = {}
        for profile in catalog.profiles:
            digest = profile.content_hash
            if digest in by_hash:
                raise SemanticAdmissionError(
                    f"provider_regime_content_hash_collision:{digest}"
                )
            by_hash[digest] = profile
        unknown = sorted(set(logic.eligible_provider_regime_hashes).difference(by_hash))
        if unknown:
            raise SemanticAdmissionError(
                "market_logic_unknown_provider_regime_hashes:" + ",".join(unknown)
            )
        return tuple(
            by_hash[digest] for digest in logic.eligible_provider_regime_hashes
        )

    @staticmethod
    def _gate_capabilities(
        requirements: tuple[CapabilityRequirement, ...],
        *,
        profiles: tuple[ProviderRegimeSpec, ...],
        intended_use: IntendedUse,
    ) -> None:
        requested_sources: set[SourceKind] = set()
        for requirement in requirements:
            if DataDomain(requirement.data_domain) is not DataDomain.MARKET_DATA:
                raise SemanticAdmissionError(
                    "provider_catalog_cannot_admit_non_market_data_capability:"
                    + "/".join(requirement.key)
                )
            source = SourceKind(requirement.source_kind)
            requested_sources.add(source)
            matching = [
                profile
                for profile in profiles
                if SourceKind(profile.source_kind) is source
            ]
            if not matching:
                raise SemanticAdmissionError(
                    "market_logic_capability_has_no_eligible_regime:"
                    + "/".join(requirement.key)
                )
            for profile in matching:
                profile.require_capabilities(
                    [DataCapability(requirement.capability)],
                    intended_use=intended_use,
                )
        unused = sorted(
            profile.profile_id
            for profile in profiles
            if SourceKind(profile.source_kind) not in requested_sources
        )
        if unused:
            raise SemanticAdmissionError(
                "market_logic_eligible_regimes_have_no_capability_requirement:"
                + ",".join(unused)
            )

    def _compile_complexity(
        self,
        *,
        predicate_count: int,
        field_count: int,
        operator_count: int,
        window_count: int,
    ) -> LogicComplexitySpec:
        ceiling = self.policy.complexity
        return LogicComplexitySpec(
            maximum_ast_nodes=min(
                ceiling.maximum_ast_nodes,
                max(16, 8 + 8 * predicate_count + 2 * field_count),
            ),
            maximum_call_depth=min(
                ceiling.maximum_call_depth, max(3, 2 + predicate_count)
            ),
            maximum_fields=min(ceiling.maximum_fields, max(1, field_count)),
            maximum_operators=min(
                ceiling.maximum_operators,
                max(4, min(operator_count, 3 * predicate_count + 4)),
            ),
            maximum_rolling_windows=min(
                ceiling.maximum_rolling_windows, max(1, window_count)
            ),
        )


def _capability(
    source: SourceKind, capability: DataCapability
) -> CapabilityRequirement:
    return CapabilityRequirement(
        data_domain=DataDomain.MARKET_DATA,
        source_kind=source,
        capability=capability,
    )


_TREND_RELATIONS = (
    ConditionRelation.ABOVE_BASELINE,
    ConditionRelation.BELOW_BASELINE,
    ConditionRelation.INCREASING,
    ConditionRelation.DECREASING,
    ConditionRelation.POSITIVE,
    ConditionRelation.NEGATIVE,
    ConditionRelation.GREATER_THAN,
    ConditionRelation.LESS_THAN,
    ConditionRelation.DIVERGING_FROM,
    ConditionRelation.CONVERGING_TO,
)

_SPREAD_RELATIONS = (
    ConditionRelation.ABOVE_BASELINE,
    ConditionRelation.BELOW_BASELINE,
    ConditionRelation.WIDENING,
    ConditionRelation.NARROWING,
    ConditionRelation.GREATER_THAN,
    ConditionRelation.LESS_THAN,
)

_STANDARD_OPERATORS = (
    "Add",
    "CsRank",
    "Div",
    "Greater",
    "IfElse",
    "Less",
    "Mul",
    "Neg",
    "Sub",
    "TsCorr",
    "TsDelta",
    "TsKurt",
    "TsMean",
    "TsRank",
    "TsSkew",
    "TsStd",
)

_STANDARD_AGGREGATIONS = (
    "calendar_daily:last",
    "calendar_daily:mean",
    "calendar_daily:sum",
)


def default_logic_compiler_policy() -> LogicCompilerPolicy:
    """Return the immutable v1 observable registry used by the compiler."""

    return LogicCompilerPolicy(
        policy_id="market_logic_default",
        version="1.0.0",
        observables=(
            ObservableDefinition(
                observable="snapshot_spread",
                required_fields=("relative_spread",),
                allowed_fields=("midpoint", "relative_spread", "spread"),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=("calendar_daily:last", "calendar_daily:mean"),
                allowed_relations=_SPREAD_RELATIONS,
                required_capabilities=(
                    _capability(
                        SourceKind.SNAPSHOT, DataCapability.SNAPSHOT_BOOK_STATE
                    ),
                ),
            ),
            ObservableDefinition(
                observable="snapshot_depth_imbalance",
                required_fields=("depth_imbalance",),
                allowed_fields=(
                    "ask_depth",
                    "bid_depth",
                    "depth_imbalance",
                    "top_depth_imbalance",
                ),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=(
                    "calendar_daily:last",
                    "calendar_daily:mean",
                    "calendar_daily:std",
                ),
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=(
                    _capability(
                        SourceKind.SNAPSHOT, DataCapability.SNAPSHOT_BOOK_STATE
                    ),
                ),
            ),
            ObservableDefinition(
                observable="snapshot_microprice_deviation",
                required_fields=("microprice", "midpoint"),
                allowed_fields=("microprice", "midpoint"),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=(
                    "calendar_daily:last",
                    "calendar_daily:mean",
                    "calendar_daily:std",
                ),
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=(
                    _capability(
                        SourceKind.SNAPSHOT, DataCapability.SNAPSHOT_BOOK_STATE
                    ),
                ),
            ),
            ObservableDefinition(
                observable="trade_imbalance",
                required_fields=("trade_imbalance",),
                allowed_fields=("trade_imbalance",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=_STANDARD_AGGREGATIONS,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=(
                    _capability(SourceKind.TRADE, DataCapability.BASIC_TRADE_FLOW),
                    _capability(SourceKind.TRADE, DataCapability.DIRECTIONAL_FLOW),
                ),
            ),
            ObservableDefinition(
                observable="order_flow_imbalance",
                required_fields=("order_flow_imbalance",),
                allowed_fields=("order_add_volume", "order_flow_imbalance"),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=_STANDARD_AGGREGATIONS,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=(
                    _capability(SourceKind.ORDER, DataCapability.BASIC_ORDER_FLOW),
                    _capability(SourceKind.ORDER, DataCapability.DIRECTIONAL_FLOW),
                ),
            ),
            ObservableDefinition(
                observable="cancel_pressure",
                required_fields=("cancel_pressure",),
                allowed_fields=("cancel_pressure", "cancel_volume", "order_add_volume"),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=_STANDARD_AGGREGATIONS,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=(
                    _capability(SourceKind.ORDER, DataCapability.CANCEL_FLOW),
                ),
            ),
            ObservableDefinition(
                observable="trade_intensity",
                required_fields=("message_count",),
                allowed_fields=("message_count",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=_STANDARD_AGGREGATIONS,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=(
                    _capability(SourceKind.TRADE, DataCapability.BASIC_TRADE_FLOW),
                ),
            ),
        ),
        forbidden_operators=(
            "BackwardFill",
            "CenteredRolling",
            "FutureReturn",
            "Lead",
            "NegativeLag",
            "TestMetricLookup",
        ),
        complexity=CompilerComplexityPolicy(),
    )


def joint_daily_logic_compiler_policy_v1() -> LogicCompilerPolicy:
    """Return the policy for the governed NOW/ORDER/TRADE daily view.

    The default policy predates :mod:`alpha_research.high_frequency.joint_daily`
    and intentionally keeps its historical field namespace.  The joint view
    prefixes every model field with its source name so an ORDER feature can
    never be confused with a similarly named TRADE or SNAPSHOT feature.  This
    separate policy preserves the default policy hash while making the actual
    nine-field joint input surface available to result-blind market logic.

    These inputs have already been aggregated to one row per security/session,
    so the only admitted aggregation is the daily ``last`` identity.  Temporal
    smoothing and differencing remain controlled by the factor DSL.
    """

    daily_last = ("calendar_daily:last",)
    snapshot_capability = (
        _capability(SourceKind.SNAPSHOT, DataCapability.SNAPSHOT_BOOK_STATE),
    )
    return LogicCompilerPolicy(
        policy_id="market_logic_joint_daily",
        version="1.0.0",
        observables=(
            ObservableDefinition(
                observable="joint_snapshot_spread",
                required_fields=("snapshot__mean_relative_spread",),
                allowed_fields=("snapshot__mean_relative_spread",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=daily_last,
                allowed_relations=_SPREAD_RELATIONS,
                required_capabilities=snapshot_capability,
            ),
            ObservableDefinition(
                observable="joint_snapshot_depth_imbalance",
                required_fields=("snapshot__mean_depth_imbalance",),
                allowed_fields=("snapshot__mean_depth_imbalance",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=daily_last,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=snapshot_capability,
            ),
            ObservableDefinition(
                observable="joint_snapshot_top_depth_imbalance",
                required_fields=("snapshot__mean_top_depth_imbalance",),
                allowed_fields=("snapshot__mean_top_depth_imbalance",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=daily_last,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=snapshot_capability,
            ),
            ObservableDefinition(
                observable="joint_snapshot_realized_volatility",
                required_fields=("snapshot__midpoint_realized_volatility",),
                allowed_fields=("snapshot__midpoint_realized_volatility",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=daily_last,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=snapshot_capability,
            ),
            ObservableDefinition(
                observable="joint_snapshot_quote_churn",
                required_fields=("snapshot__mean_snapshot_quote_churn",),
                allowed_fields=("snapshot__mean_snapshot_quote_churn",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=daily_last,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=snapshot_capability,
            ),
            ObservableDefinition(
                observable="joint_order_flow_imbalance",
                required_fields=("order__order_flow_imbalance",),
                allowed_fields=("order__order_flow_imbalance",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=daily_last,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=(
                    _capability(
                        SourceKind.ORDER, DataCapability.BASIC_ORDER_FLOW
                    ),
                    _capability(
                        SourceKind.ORDER, DataCapability.DIRECTIONAL_FLOW
                    ),
                ),
            ),
            ObservableDefinition(
                observable="joint_cancel_pressure",
                required_fields=("order__cancel_pressure",),
                allowed_fields=("order__cancel_pressure",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=daily_last,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=(
                    _capability(SourceKind.ORDER, DataCapability.CANCEL_FLOW),
                ),
            ),
            ObservableDefinition(
                observable="joint_trade_imbalance",
                required_fields=("trade__trade_imbalance",),
                allowed_fields=("trade__trade_imbalance",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=daily_last,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=(
                    _capability(
                        SourceKind.TRADE, DataCapability.BASIC_TRADE_FLOW
                    ),
                    _capability(
                        SourceKind.TRADE, DataCapability.DIRECTIONAL_FLOW
                    ),
                ),
            ),
            ObservableDefinition(
                observable="joint_trade_directional_coverage",
                required_fields=("trade__trade_directional_coverage",),
                allowed_fields=("trade__trade_directional_coverage",),
                allowed_operators=_STANDARD_OPERATORS,
                allowed_aggregations=daily_last,
                allowed_relations=_TREND_RELATIONS,
                required_capabilities=(
                    _capability(
                        SourceKind.TRADE, DataCapability.BASIC_TRADE_FLOW
                    ),
                    _capability(
                        SourceKind.TRADE, DataCapability.DIRECTIONAL_FLOW
                    ),
                ),
            ),
        ),
        forbidden_operators=(
            "BackwardFill",
            "CenteredRolling",
            "FutureReturn",
            "Lead",
            "NegativeLag",
            "TestMetricLookup",
        ),
        complexity=CompilerComplexityPolicy(),
    )


def _normalize_hashes(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(values))
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be non-empty and unique")
    for digest in normalized:
        require_sha256(digest, name=name)
    return normalized


def _normalize_codes(
    values: tuple[str, ...], *, field: bool, name: str
) -> tuple[str, ...]:
    pattern = _FIELD if field else _CODE
    normalized = tuple(sorted(values))
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique")
    if any(
        not isinstance(item, str) or pattern.fullmatch(item) is None
        for item in normalized
    ):
        raise ValueError(f"{name} contain unsupported characters")
    return normalized


def _require_code(value: str, *, name: str) -> None:
    if not isinstance(value, str) or _CODE.fullmatch(value) is None:
        raise ValueError(f"{name} contains unsupported characters")


def _exact_keys(value: Mapping[str, object], expected: set[str], *, path: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{path} wire fields differ")


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object")
    return cast(Mapping[str, object], value)


def _mapping_list(value: object, *, path: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{path} must be a list of objects")
    return tuple(_mapping(item, path=f"{path}[]") for item in value)


def _strings(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{path} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{path} must contain only strings")
    return tuple(cast(str, item) for item in value)


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be text")
    return value


def _integer(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path} must be an integer")
    return value


__all__ = [
    "CompiledWindowBound",
    "CompilerComplexityPolicy",
    "LogicCompilationError",
    "LogicCompilerPolicy",
    "LogicComplexitySpec",
    "LogicConstraintSpec",
    "MarketLogicCompiler",
    "ObservableDefinition",
    "default_logic_compiler_policy",
]
