from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import pandas as pd

from alpha_research.core.frequency import FrequencyMode
from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.factors.complexity import (
    FactorComplexity,
    analyze_expression,
    enforce_complexity,
)
from alpha_research.factors.spec import FactorDirection, FactorSpec
from alpha_research.factors.view import FactorDataView
from alpha_research.high_frequency.provider_semantics import (
    IntendedUse,
    ProviderSemanticCatalog,
)
from alpha_research.market_logic.compiler import (
    LogicCompilerPolicy,
    LogicConstraintSpec,
    MarketLogicCompiler,
)
from alpha_research.market_logic.lineage import (
    ParentLineageError,
    ParentLineageVerificationReceipt,
    ParentLogicFactorLineage,
)
from alpha_research.market_logic.registry import MarketLogicRegistry
from alpha_research.market_logic.spec import (
    LogicOrigin,
    MarketLogicSpec,
    PredictionDirection,
    WindowUnit,
)
from factor_production.v5.dsl import OperatorRegistry, SafeExpressionInterpreter
from factor_production.v5.llm.schemas import canonical_expression_ast


class LogicFactorBindingError(ValueError):
    """A factor is not an exact executable implementation of its market logic."""


@dataclass(frozen=True, slots=True)
class BoundWindowUse:
    """One rolling-window use recovered from the validated expression AST."""

    operator: str
    window: int
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.operator or not isinstance(self.window, int) or self.window <= 0:
            raise ValueError("bound window use is invalid")
        fields = tuple(sorted(self.fields))
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("bound window fields must be non-empty and unique")
        object.__setattr__(self, "fields", fields)

    def to_dict(self) -> dict[str, object]:
        return {
            "operator": self.operator,
            "window": self.window,
            "fields": list(self.fields),
        }


@dataclass(frozen=True, slots=True)
class LogicFactorBindingReceipt:
    """Content-addressed proof that a factor passed the logic envelope."""

    market_logic_hash: str
    logic_constraint_hash: str
    compiler_policy_hash: str
    semantic_catalog_hash: str
    data_profile_hash: str
    provider_regime_hashes: tuple[str, ...]
    factor_spec_hash: str
    factor_definition_hash: str
    factor_semantic_hash: str
    operator_registry_version: str
    operator_registry_digest: str
    expression_ast: str
    expression_fields: tuple[str, ...]
    expression_operators: tuple[str, ...]
    window_uses: tuple[BoundWindowUse, ...]
    aggregation_key: str
    complexity: FactorComplexity
    intended_use: IntendedUse | str
    parent_lineage_verification_hash: str | None = None
    schema_version: str = "logic-factor-binding/v1"

    def __post_init__(self) -> None:
        expected_schema = (
            "logic-factor-binding/v2"
            if self.parent_lineage_verification_hash is not None
            else "logic-factor-binding/v1"
        )
        if self.schema_version != expected_schema:
            raise ValueError("unsupported LogicFactorBindingReceipt schema")
        for name in (
            "market_logic_hash",
            "logic_constraint_hash",
            "compiler_policy_hash",
            "semantic_catalog_hash",
            "data_profile_hash",
            "factor_spec_hash",
            "factor_definition_hash",
            "factor_semantic_hash",
            "operator_registry_digest",
        ):
            require_sha256(
                str(getattr(self, name)), name=f"logic factor binding {name}"
            )
        regimes = _hashes(self.provider_regime_hashes, name="provider regime hashes")
        fields = _codes(self.expression_fields, name="expression fields")
        operators = _codes(self.expression_operators, name="expression operators")
        if not self.operator_registry_version or not self.expression_ast:
            raise ValueError("binding registry version and expression AST are required")
        if not self.aggregation_key:
            raise ValueError("binding aggregation key is required")
        if not isinstance(self.complexity, FactorComplexity):
            raise TypeError("binding complexity must be FactorComplexity")
        object.__setattr__(self, "provider_regime_hashes", regimes)
        object.__setattr__(self, "expression_fields", fields)
        object.__setattr__(self, "expression_operators", operators)
        object.__setattr__(self, "intended_use", IntendedUse(self.intended_use))
        if self.parent_lineage_verification_hash is not None:
            require_sha256(
                self.parent_lineage_verification_hash,
                name="parent lineage verification hash",
            )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        intended_use = self.intended_use
        if not isinstance(intended_use, IntendedUse):  # pragma: no cover
            raise RuntimeError("binding intended use was not normalized")
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "market_logic_hash": self.market_logic_hash,
            "logic_constraint_hash": self.logic_constraint_hash,
            "compiler_policy_hash": self.compiler_policy_hash,
            "semantic_catalog_hash": self.semantic_catalog_hash,
            "data_profile_hash": self.data_profile_hash,
            "provider_regime_hashes": list(self.provider_regime_hashes),
            "factor_spec_hash": self.factor_spec_hash,
            "factor_definition_hash": self.factor_definition_hash,
            "factor_semantic_hash": self.factor_semantic_hash,
            "operator_registry_version": self.operator_registry_version,
            "operator_registry_digest": self.operator_registry_digest,
            "expression_ast": self.expression_ast,
            "expression_fields": list(self.expression_fields),
            "expression_operators": list(self.expression_operators),
            "window_uses": [item.to_dict() for item in self.window_uses],
            "aggregation_key": self.aggregation_key,
            "complexity": self.complexity.to_dict(),
            "intended_use": intended_use.value,
        }
        if self.parent_lineage_verification_hash is not None:
            payload["parent_lineage_verification_hash"] = (
                self.parent_lineage_verification_hash
            )
        return payload


@dataclass(frozen=True, slots=True)
class FactorExecutionPreflightReceipt:
    """Binds a logic/factor receipt to the exact immutable execution view."""

    binding_hash: str
    factor_view_hash: str
    dataset_id: str
    snapshot_id: str
    schema_hash: str
    availability_hash: str
    security_contract_hash: str
    frequency_hash: str
    strict_point_in_time: bool
    production_ready: bool
    schema_version: str = "logic-factor-execution-preflight/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "logic-factor-execution-preflight/v1":
            raise ValueError("unsupported FactorExecutionPreflightReceipt schema")
        for name in (
            "binding_hash",
            "factor_view_hash",
            "snapshot_id",
            "schema_hash",
            "availability_hash",
            "security_contract_hash",
            "frequency_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"factor preflight {name}")
        if not self.dataset_id.strip():
            raise ValueError("factor preflight dataset_id must not be empty")
        if not isinstance(self.strict_point_in_time, bool) or not isinstance(
            self.production_ready, bool
        ):
            raise TypeError("factor preflight readiness flags must be boolean")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "binding_hash": self.binding_hash,
            "factor_view_hash": self.factor_view_hash,
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "schema_hash": self.schema_hash,
            "availability_hash": self.availability_hash,
            "security_contract_hash": self.security_contract_hash,
            "frequency_hash": self.frequency_hash,
            "strict_point_in_time": self.strict_point_in_time,
            "production_ready": self.production_ready,
        }


class MarketLogicFactorBinder:
    """Fail-closed binding from trusted market logic to a governed FactorSpec."""

    def bind(
        self,
        *,
        logic: MarketLogicSpec,
        constraint: LogicConstraintSpec,
        factor: FactorSpec,
        compiler_policy: LogicCompilerPolicy,
        semantic_catalog: ProviderSemanticCatalog,
        data_profile_hash: str,
        provider_regime_hashes: tuple[str, ...],
        operator_registry: OperatorRegistry,
        available_fields: Iterable[str],
        intended_use: IntendedUse | str = IntendedUse.RESEARCH,
        parent_lineage: ParentLogicFactorLineage | None = None,
        parent_lineage_receipt: ParentLineageVerificationReceipt | None = None,
        lineage_registry: MarketLogicRegistry | None = None,
    ) -> LogicFactorBindingReceipt:
        if not isinstance(logic, MarketLogicSpec):
            raise TypeError("logic factor binder requires MarketLogicSpec")
        if not isinstance(constraint, LogicConstraintSpec):
            raise TypeError("logic factor binder requires LogicConstraintSpec")
        if not isinstance(factor, FactorSpec):
            raise TypeError("logic factor binder requires FactorSpec")
        if not isinstance(compiler_policy, LogicCompilerPolicy):
            raise TypeError("logic factor binder requires LogicCompilerPolicy")
        if not isinstance(semantic_catalog, ProviderSemanticCatalog):
            raise TypeError("logic factor binder requires ProviderSemanticCatalog")
        if not isinstance(operator_registry, OperatorRegistry):
            raise TypeError("logic factor binder requires OperatorRegistry")
        use = IntendedUse(intended_use)
        require_sha256(data_profile_hash, name="runtime data_profile_hash")

        expected_constraint = MarketLogicCompiler(compiler_policy).compile(
            logic,
            semantic_catalog=semantic_catalog,
            intended_use=use,
        )
        if constraint.content_hash != expected_constraint.content_hash:
            raise LogicFactorBindingError(
                "logic_constraint_differs_from_recompiled_policy"
            )
        if logic.semantic_catalog_hash != semantic_catalog.content_hash:
            raise LogicFactorBindingError("logic_semantic_catalog_hash_mismatch")
        if constraint.semantic_catalog_hash != semantic_catalog.content_hash:
            raise LogicFactorBindingError("constraint_semantic_catalog_hash_mismatch")
        if constraint.compiler_policy_hash != compiler_policy.content_hash:
            raise LogicFactorBindingError("constraint_compiler_policy_hash_mismatch")
        if logic.data_profile_hash != data_profile_hash:
            raise LogicFactorBindingError("logic_data_profile_hash_mismatch")

        selected_regimes = _hashes(
            provider_regime_hashes, name="runtime provider regime hashes"
        )
        catalog_regimes = {
            profile.content_hash for profile in semantic_catalog.profiles
        }
        unknown_regimes = sorted(set(selected_regimes).difference(catalog_regimes))
        if unknown_regimes:
            raise LogicFactorBindingError(
                "runtime_provider_regime_hash_unknown:" + ",".join(unknown_regimes)
            )
        if selected_regimes != logic.eligible_provider_regime_hashes:
            raise LogicFactorBindingError(
                "runtime_provider_regime_hashes_differ_from_logic"
            )
        if selected_regimes != constraint.eligible_provider_regime_hashes:
            raise LogicFactorBindingError(
                "runtime_provider_regime_hashes_differ_from_constraint"
            )
        if factor.availability_hash != logic.availability_hash:
            raise LogicFactorBindingError("factor_availability_hash_differs_from_logic")

        self._validate_identity_and_provenance(logic, factor)
        lineage_verification_hash = self._validate_authoritative_parent_lineage(
            logic=logic,
            factor=factor,
            parent_lineage=parent_lineage,
            parent_lineage_receipt=parent_lineage_receipt,
            lineage_registry=lineage_registry,
        )
        if factor.operator_registry_version != operator_registry.version:
            raise LogicFactorBindingError("factor_operator_registry_version_mismatch")
        if factor.operator_registry_digest != operator_registry.digest:
            raise LogicFactorBindingError("factor_operator_registry_digest_mismatch")

        maximum_window = max(item.maximum for item in constraint.window_bounds)
        interpreter = SafeExpressionInterpreter(
            operator_registry,
            max_call_depth=constraint.complexity.maximum_call_depth,
            maximum_window=maximum_window,
        )
        validation = interpreter.validate(
            factor.expression,
            allowed_fields=frozenset(constraint.allowed_fields),
        )
        if not validation.is_valid:
            raise LogicFactorBindingError(
                "logic_factor_expression_rejected:" + ";".join(validation.reasons)
            )
        expression_fields = tuple(validation.fields)
        expression_operators = tuple(validation.operators)
        forbidden = sorted(
            set(expression_operators).intersection(constraint.forbidden_operators)
        )
        unallowed = sorted(
            set(expression_operators).difference(constraint.allowed_operators)
        )
        if forbidden:
            raise LogicFactorBindingError(
                "logic_factor_uses_forbidden_operators:" + ",".join(forbidden)
            )
        if unallowed:
            raise LogicFactorBindingError(
                "logic_factor_uses_unallowed_operators:" + ",".join(unallowed)
            )

        exposure_fields = tuple(
            sorted(
                {
                    field
                    for step in factor.preprocessing
                    for field in step.exposure_fields
                }
            )
        )
        bound_fields = tuple(sorted({*expression_fields, *exposure_fields}))
        if bound_fields != factor.required_fields:
            raise LogicFactorBindingError(
                "factor_required_fields_differ_from_expression_and_preprocessing"
            )
        if not set(constraint.required_fields).issubset(expression_fields):
            missing = sorted(
                set(constraint.required_fields).difference(expression_fields)
            )
            raise LogicFactorBindingError(
                "factor_omits_logic_required_fields:" + ",".join(missing)
            )
        if not set(factor.required_fields).issubset(constraint.allowed_fields):
            raise LogicFactorBindingError(
                "factor_required_fields_exceed_logic_constraint"
            )
        available = _codes(tuple(available_fields), name="available fields")
        missing_inputs = sorted(set(factor.required_fields).difference(available))
        if missing_inputs:
            raise LogicFactorBindingError(
                "factor_execution_inputs_missing:" + ",".join(missing_inputs)
            )

        aggregation_key = _aggregation_key(factor)
        if aggregation_key not in set(constraint.allowed_aggregations):
            raise LogicFactorBindingError(
                f"factor_aggregation_not_admitted:{aggregation_key}"
            )

        complexity = analyze_expression(factor.expression, registry=operator_registry)
        try:
            enforce_complexity(complexity, factor.complexity_budget)
        except ValueError as exc:
            raise LogicFactorBindingError(str(exc)) from exc
        window_uses = _window_uses(factor.expression, operator_registry)
        self._validate_logic_complexity(
            factor=factor,
            constraint=constraint,
            complexity=complexity,
            window_uses=window_uses,
        )
        self._validate_windows(
            factor=factor,
            constraint=constraint,
            compiler_policy=compiler_policy,
            window_uses=window_uses,
        )

        return LogicFactorBindingReceipt(
            market_logic_hash=logic.content_hash,
            logic_constraint_hash=constraint.content_hash,
            compiler_policy_hash=compiler_policy.content_hash,
            semantic_catalog_hash=semantic_catalog.content_hash,
            data_profile_hash=data_profile_hash,
            provider_regime_hashes=selected_regimes,
            factor_spec_hash=factor.content_hash,
            factor_definition_hash=factor.definition_hash,
            factor_semantic_hash=factor.semantic_hash,
            operator_registry_version=operator_registry.version,
            operator_registry_digest=operator_registry.digest,
            expression_ast=canonical_expression_ast(factor.expression),
            expression_fields=expression_fields,
            expression_operators=expression_operators,
            window_uses=window_uses,
            aggregation_key=aggregation_key,
            complexity=complexity,
            intended_use=use,
            parent_lineage_verification_hash=lineage_verification_hash,
            schema_version=(
                "logic-factor-binding/v2"
                if lineage_verification_hash is not None
                else "logic-factor-binding/v1"
            ),
        )

    def preflight(
        self,
        *,
        logic: MarketLogicSpec,
        constraint: LogicConstraintSpec,
        factor: FactorSpec,
        compiler_policy: LogicCompilerPolicy,
        semantic_catalog: ProviderSemanticCatalog,
        data_profile_hash: str,
        provider_regime_hashes: tuple[str, ...],
        view: FactorDataView,
        dataset_id: str,
        intended_use: IntendedUse | str = IntendedUse.RESEARCH,
        require_point_in_time: bool = True,
        require_production_ready: bool = True,
        expected_binding_hash: str | None = None,
        parent_lineage: ParentLogicFactorLineage | None = None,
        parent_lineage_receipt: ParentLineageVerificationReceipt | None = None,
        lineage_registry: MarketLogicRegistry | None = None,
    ) -> tuple[LogicFactorBindingReceipt, FactorExecutionPreflightReceipt]:
        """Revalidate every binding immediately before FactorEngine execution."""

        if not isinstance(view, FactorDataView):
            raise TypeError("factor execution preflight requires FactorDataView")
        view.verify_content()
        if not dataset_id.strip() or dataset_id != factor.dataset_id:
            raise LogicFactorBindingError("factor_dataset_binding_mismatch")
        view_bindings = {
            "snapshot_id": (factor.snapshot_id, view.snapshot_id),
            "schema_hash": (factor.schema_hash, view.schema_hash),
            "availability_hash": (factor.availability_hash, view.availability_hash),
            "security_contract_hash": (
                factor.security_contract_hash,
                view.security_contract_hash,
            ),
            "frequency_hash": (factor.frequency.content_hash, view.frequency_hash),
        }
        drift = [
            name for name, values in view_bindings.items() if values[0] != values[1]
        ]
        if drift:
            raise LogicFactorBindingError(
                "factor_view_bindings_differ:" + ",".join(sorted(drift))
            )
        if require_production_ready and not view.production_ready:
            raise LogicFactorBindingError("factor_view_not_production_ready")
        if require_point_in_time:
            if not view.strict_point_in_time or not view.has_row_availability:
                raise LogicFactorBindingError("factor_view_not_strict_point_in_time")
            view.point_in_time_mask(require=True)

        registry = OperatorRegistry.dataframe_pit_v3(
            view.cross_section_mask,
            view.history_mask,
        )
        binding = self.bind(
            logic=logic,
            constraint=constraint,
            factor=factor,
            compiler_policy=compiler_policy,
            semantic_catalog=semantic_catalog,
            data_profile_hash=data_profile_hash,
            provider_regime_hashes=provider_regime_hashes,
            operator_registry=registry,
            available_fields=view.fields,
            intended_use=intended_use,
            parent_lineage=parent_lineage,
            parent_lineage_receipt=parent_lineage_receipt,
            lineage_registry=lineage_registry,
        )
        if expected_binding_hash is not None:
            require_sha256(
                expected_binding_hash, name="expected logic factor binding hash"
            )
            if binding.content_hash != expected_binding_hash:
                raise LogicFactorBindingError(
                    "stale_or_mismatched_logic_factor_binding"
                )
        preflight = FactorExecutionPreflightReceipt(
            binding_hash=binding.content_hash,
            factor_view_hash=view.view_hash,
            dataset_id=dataset_id,
            snapshot_id=view.snapshot_id,
            schema_hash=view.schema_hash,
            availability_hash=view.availability_hash,
            security_contract_hash=view.security_contract_hash,
            frequency_hash=view.frequency_hash,
            strict_point_in_time=view.strict_point_in_time,
            production_ready=view.production_ready,
        )
        return binding, preflight

    @staticmethod
    def _validate_identity_and_provenance(
        logic: MarketLogicSpec, factor: FactorSpec
    ) -> None:
        expected_direction = {
            PredictionDirection.POSITIVE: FactorDirection.POSITIVE,
            PredictionDirection.HIGHER: FactorDirection.POSITIVE,
            PredictionDirection.NEGATIVE: FactorDirection.NEGATIVE,
            PredictionDirection.LOWER: FactorDirection.NEGATIVE,
        }[PredictionDirection(logic.hypothesis.belief.direction)]
        if factor.direction is not expected_direction:
            raise LogicFactorBindingError("factor_direction_differs_from_market_belief")
        if factor.family != logic.mechanism_code:
            raise LogicFactorBindingError("factor_family_differs_from_logic_mechanism")
        if factor.economic_rationale != logic.economic_rationale:
            raise LogicFactorBindingError(
                "factor_economic_rationale_differs_from_logic"
            )
        if factor.falsification_criterion != logic.falsification.failure_description:
            raise LogicFactorBindingError("factor_falsification_differs_from_logic")

        logic_provenance = logic.provenance
        factor_provenance = factor.provenance
        logic_origin = LogicOrigin(logic_provenance.origin).value
        if factor_provenance.origin != logic_origin:
            raise LogicFactorBindingError("factor_origin_differs_from_logic_origin")
        for name in ("actor_id", "model_id", "prompt_hash"):
            if getattr(factor_provenance, name) != getattr(logic_provenance, name):
                raise LogicFactorBindingError(
                    f"factor_provenance_{name}_differs_from_logic"
                )
        if len(factor_provenance.parent_factor_hashes) != len(
            logic_provenance.parent_logic_hashes
        ):
            raise LogicFactorBindingError(
                "factor_parent_lineage_shape_differs_from_logic"
            )

    @staticmethod
    def _validate_authoritative_parent_lineage(
        *,
        logic: MarketLogicSpec,
        factor: FactorSpec,
        parent_lineage: ParentLogicFactorLineage | None,
        parent_lineage_receipt: ParentLineageVerificationReceipt | None,
        lineage_registry: MarketLogicRegistry | None,
    ) -> str | None:
        has_parent_lineage = bool(
            logic.provenance.parent_logic_hashes
            or factor.provenance.parent_factor_hashes
        )
        supplied = (
            parent_lineage is not None,
            parent_lineage_receipt is not None,
            lineage_registry is not None,
        )
        if not has_parent_lineage:
            if any(supplied):
                raise LogicFactorBindingError(
                    "genesis_factor_must_not_supply_parent_lineage"
                )
            return None
        if not all(supplied):
            raise LogicFactorBindingError(
                "evolutionary_factor_requires_authoritative_parent_lineage"
            )
        assert parent_lineage is not None
        assert parent_lineage_receipt is not None
        assert lineage_registry is not None
        try:
            verified = parent_lineage.revalidate_receipt(
                parent_lineage_receipt,
                logic_hash=logic.content_hash,
                factor_hash=factor.content_hash,
                logic_provenance=logic.provenance,
                factor_provenance=factor.provenance,
                registry=lineage_registry,
            )
        except (
            ParentLineageError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise LogicFactorBindingError(
                "authoritative_parent_lineage_failed"
            ) from exc
        return cast(str, verified.content_hash)

    @staticmethod
    def _validate_logic_complexity(
        *,
        factor: FactorSpec,
        constraint: LogicConstraintSpec,
        complexity: FactorComplexity,
        window_uses: tuple[BoundWindowUse, ...],
    ) -> None:
        limits = {
            "ast_nodes": (
                complexity.ast_nodes,
                constraint.complexity.maximum_ast_nodes,
            ),
            "call_depth": (
                complexity.maximum_call_depth,
                constraint.complexity.maximum_call_depth,
            ),
            "fields": (
                complexity.field_count,
                constraint.complexity.maximum_fields,
            ),
            "operators": (
                complexity.call_count,
                constraint.complexity.maximum_operators,
            ),
            "rolling_windows": (
                len(window_uses),
                constraint.complexity.maximum_rolling_windows,
            ),
        }
        violations = [
            f"{name}:{actual}>{maximum}"
            for name, (actual, maximum) in limits.items()
            if actual > maximum
        ]
        smoothing = factor.aggregation.smoothing_span
        if smoothing > factor.complexity_budget.maximum_window:
            violations.append(
                "aggregation_smoothing_window:"
                f"{smoothing}>{factor.complexity_budget.maximum_window}"
            )
        if violations:
            raise LogicFactorBindingError(
                "logic_factor_complexity_exceeded:" + ",".join(violations)
            )

    @staticmethod
    def _validate_windows(
        *,
        factor: FactorSpec,
        constraint: LogicConstraintSpec,
        compiler_policy: LogicCompilerPolicy,
        window_uses: tuple[BoundWindowUse, ...],
    ) -> None:
        definition_map = compiler_policy.observable_map
        satisfied: set[str] = set()
        for use in window_uses:
            candidates = []
            for bound in constraint.window_bounds:
                definition = definition_map.get(bound.observable)
                if definition is None:
                    raise LogicFactorBindingError(
                        f"constraint_observable_missing_from_policy:{bound.observable}"
                    )
                if not set(definition.required_fields).issubset(use.fields):
                    continue
                if not _window_unit_matches_factor(bound.unit, factor):
                    continue
                candidates.append(bound)
            admitted = [
                bound
                for bound in candidates
                if bound.minimum <= use.window <= bound.maximum
            ]
            if not admitted:
                raise LogicFactorBindingError(
                    "expression_window_not_bound_to_logic:"
                    f"{use.operator}:{use.window}:{','.join(use.fields)}"
                )
            satisfied.update(item.predicate_hash for item in admitted)

        smoothing = factor.aggregation.smoothing_span
        if smoothing:
            smoothing_bounds = [
                bound
                for bound in constraint.window_bounds
                if WindowUnit(bound.unit) in {WindowUnit.SESSIONS, WindowUnit.DAYS}
                and bound.minimum <= smoothing <= bound.maximum
            ]
            if not smoothing_bounds:
                raise LogicFactorBindingError(
                    f"aggregation_smoothing_not_bound_to_logic:{smoothing}"
                )
            satisfied.update(item.predicate_hash for item in smoothing_bounds)

        missing = sorted(
            item.predicate_hash
            for item in constraint.window_bounds
            if item.predicate_hash not in satisfied
        )
        if missing:
            raise LogicFactorBindingError(
                "logic_condition_windows_not_implemented:" + ",".join(missing)
            )


def _aggregation_key(factor: FactorSpec) -> str:
    aggregation = factor.aggregation
    if aggregation.method == "calendar_daily":
        return f"calendar_daily:{aggregation.reducer}"
    if _is_daily_bar(factor):
        return "calendar_daily:last"
    return "native:last"


def _is_daily_bar(factor: FactorSpec) -> bool:
    frequency = factor.frequency
    if (
        FrequencyMode(frequency.mode) is not FrequencyMode.BAR
        or frequency.interval is None
    ):
        return False
    return cast(bool, pd.Timedelta(frequency.interval) >= pd.Timedelta(days=1))


def _window_unit_matches_factor(unit: WindowUnit | str, factor: FactorSpec) -> bool:
    normalized = WindowUnit(unit)
    mode = FrequencyMode(factor.frequency.mode)
    if normalized is WindowUnit.EVENTS:
        return mode is FrequencyMode.EVENT
    if normalized is WindowUnit.BARS:
        return mode in {FrequencyMode.BAR, FrequencyMode.VOLUME}
    return _is_daily_bar(factor)


def _window_uses(
    expression: str, registry: OperatorRegistry
) -> tuple[BoundWindowUse, ...]:
    """Read windows from the already safe-DSL-validated AST, never from text."""

    tree = ast.parse(expression, mode="eval")
    operators = frozenset(registry.names)
    uses: list[BoundWindowUse] = []
    for node in ast.walk(tree.body):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        spec = registry.get(node.func.id)
        if spec is None or spec.window_argument is None:
            continue
        window_node = node.args[spec.window_argument]
        if (
            not isinstance(window_node, ast.Constant)
            or isinstance(window_node.value, bool)
            or not isinstance(window_node.value, int)
        ):
            raise LogicFactorBindingError(
                f"validated_window_is_not_integer:{node.func.id}"
            )
        fields = tuple(
            sorted(
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name) and child.id not in operators
            )
        )
        uses.append(
            BoundWindowUse(
                operator=node.func.id,
                window=int(window_node.value),
                fields=tuple(dict.fromkeys(fields)),
            )
        )
    return tuple(
        sorted(uses, key=lambda item: (item.operator, item.window, item.fields))
    )


def _hashes(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(values))
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be non-empty and unique")
    for digest in normalized:
        require_sha256(digest, name=name)
    return normalized


def _codes(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(str(item).strip() for item in values))
    if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be non-empty and unique")
    return normalized


__all__ = [
    "BoundWindowUse",
    "FactorExecutionPreflightReceipt",
    "LogicFactorBindingError",
    "LogicFactorBindingReceipt",
    "MarketLogicFactorBinder",
]
