from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Mapping

import pandas as pd

from llm_alpha_mining.mining.dsl import operators as dataframe_operators


_FUTURE_FIELD_TOKENS = (
    "future",
    "forward",
    "nextreturn",
    "next_return",
    "label",
    "target",
)

DAILY_IFELSE_SEMANTICS_REVISION = "daily_ifelse_condition_axis_broadcast_v2"


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    function: Callable[..., object]
    window_argument: int | None = None


@dataclass(frozen=True)
class ExpressionValidation:
    is_valid: bool
    reasons: tuple[str, ...]
    fields: tuple[str, ...]
    operators: tuple[str, ...]
    max_call_depth: int


class OperatorRegistry:
    """Versioned allow-list for the declarative factor expression language."""

    def __init__(
        self,
        specs: Mapping[str, OperatorSpec],
        *,
        version: str,
        semantics_revision: str | None = None,
    ):
        self._specs = dict(specs)
        self.version = str(version)
        self.semantics_revision = (
            None if semantics_revision is None else str(semantics_revision)
        )

    @classmethod
    def dataframe_v1(cls) -> "OperatorRegistry":
        definitions = {
            "Add": (2, None),
            "Sub": (2, None),
            "Mul": (2, None),
            "Div": (2, None),
            "Neg": (1, None),
            "TsRank": (2, 1),
            "CsRank": (1, None),
            "TsMean": (2, 1),
            "TsStd": (2, 1),
            "TsDelta": (2, 1),
            "TsCorr": (3, 2),
            "TsSkew": (2, 1),
            "TsKurt": (2, 1),
            "Greater": (2, None),
            "Less": (2, None),
            "IfElse": (3, None),
        }
        return cls(
            {
                name: OperatorSpec(
                    name=name,
                    arity=arity,
                    function=(
                        dataframe_operators._if_else_v1
                        if name == "IfElse"
                        else getattr(dataframe_operators, name)
                    ),
                    window_argument=window_argument,
                )
                for name, (arity, window_argument) in definitions.items()
            },
            version="dataframe_v1",
        )

    @classmethod
    def dataframe_v2(cls) -> "OperatorRegistry":
        """Daily operators with condition-axis ``IfElse`` broadcasting."""

        legacy = cls.dataframe_v1()
        specs = dict(legacy._specs)
        specs["IfElse"] = OperatorSpec(
            "IfElse",
            3,
            dataframe_operators._if_else_v2,
        )
        return cls(
            specs,
            version="dataframe_v2_ifelse_condition_axis_broadcast",
            semantics_revision=DAILY_IFELSE_SEMANTICS_REVISION,
        )

    @classmethod
    def dataframe_pit_v1(cls, point_in_time_mask: pd.DataFrame) -> "OperatorRegistry":
        """Build a registry whose cross-sections obey a PIT stock mask.

        Raw-field masking alone is insufficient for conditional formulas:
        ``IfElse(Less(NaN, 0), x, 0)`` can otherwise turn an ineligible code
        into a numeric zero before ``CsRank``.  This registry therefore masks
        conditional output and every cross-sectional rank input/output.
        """

        base = cls.dataframe_v1()
        mask = pd.DataFrame(point_in_time_mask).copy()
        base_if_else = base.get("IfElse")
        if base_if_else is None:  # pragma: no cover - construction invariant
            raise RuntimeError("base registry is missing IfElse")

        def aligned_like(value: pd.DataFrame) -> pd.DataFrame:
            return (
                mask.reindex(index=value.index, columns=value.columns)
                .fillna(False)
                .astype(bool)
            )

        def pit_cs_rank(value: pd.DataFrame) -> pd.DataFrame:
            active = aligned_like(value)
            return dataframe_operators.CsRank(value.where(active)).where(active)

        def pit_if_else(
            condition: pd.DataFrame,
            when_true: pd.DataFrame,
            when_false: object,
        ) -> pd.DataFrame:
            result = base_if_else.function(
                condition,
                when_true,
                when_false,
            )
            return result.where(aligned_like(result))

        specs = dict(base._specs)
        specs["CsRank"] = OperatorSpec("CsRank", 1, pit_cs_rank)
        specs["IfElse"] = OperatorSpec("IfElse", 3, pit_if_else)
        return cls(specs, version="dataframe_pit_v1")

    @classmethod
    def dataframe_pit_v2(
        cls,
        cross_section_mask: pd.DataFrame,
        security_master_mask: pd.DataFrame,
    ) -> "OperatorRegistry":
        """Separate stock identity/history from signal-date eligibility.

        Historical time-series inputs are governed by the point-in-time stock
        security master.  A possibly narrower tradable/universe mask is applied
        immediately around ``CsRank``.  This prevents a one-day eligibility
        exclusion from erasing a 20--60 day history while still guaranteeing
        that the excluded code cannot enter that day's cross-section.
        """

        base = cls.dataframe_v1()
        rank_mask = pd.DataFrame(cross_section_mask).copy()
        history_mask = pd.DataFrame(security_master_mask).copy()
        base_if_else = base.get("IfElse")
        if base_if_else is None:  # pragma: no cover - construction invariant
            raise RuntimeError("base registry is missing IfElse")

        def aligned(mask: pd.DataFrame, value: pd.DataFrame) -> pd.DataFrame:
            return (
                mask.reindex(index=value.index, columns=value.columns)
                .fillna(False)
                .astype(bool)
            )

        def pit_cs_rank(value: pd.DataFrame) -> pd.DataFrame:
            active = aligned(rank_mask, value)
            return dataframe_operators.CsRank(value.where(active)).where(active)

        def pit_if_else(
            condition: pd.DataFrame,
            when_true: pd.DataFrame,
            when_false: object,
        ) -> pd.DataFrame:
            result = base_if_else.function(
                condition,
                when_true,
                when_false,
            )
            return result.where(aligned(history_mask, result))

        specs = dict(base._specs)
        specs["CsRank"] = OperatorSpec("CsRank", 1, pit_cs_rank)
        specs["IfElse"] = OperatorSpec("IfElse", 3, pit_if_else)
        return cls(specs, version="dataframe_pit_v2_split_history_and_rank_masks")

    @classmethod
    def dataframe_pit_v3(
        cls,
        cross_section_mask: pd.DataFrame,
        security_master_mask: pd.DataFrame,
    ) -> "OperatorRegistry":
        """PIT split-mask registry with condition-axis scalar broadcasting.

        This is a new immutable operator contract.  ``dataframe_pit_v2`` keeps
        the original x-axis conditional implementation so frozen G0 artifacts
        remain replayable under their recorded version and digest.
        """

        base = cls.dataframe_v2()
        rank_mask = pd.DataFrame(cross_section_mask).copy()
        history_mask = pd.DataFrame(security_master_mask).copy()
        base_if_else = base.get("IfElse")
        if base_if_else is None:  # pragma: no cover - construction invariant
            raise RuntimeError("base registry is missing IfElse")

        def aligned(mask: pd.DataFrame, value: pd.DataFrame) -> pd.DataFrame:
            return (
                mask.reindex(index=value.index, columns=value.columns)
                .fillna(False)
                .astype(bool)
            )

        def pit_cs_rank(value: pd.DataFrame) -> pd.DataFrame:
            active = aligned(rank_mask, value)
            return dataframe_operators.CsRank(value.where(active)).where(active)

        def pit_if_else(
            condition: pd.DataFrame,
            when_true: object,
            when_false: object,
        ) -> pd.DataFrame:
            result = base_if_else.function(
                condition,
                when_true,
                when_false,
            )
            return result.where(aligned(history_mask, result))

        specs = dict(base._specs)
        specs["CsRank"] = OperatorSpec("CsRank", 1, pit_cs_rank)
        specs["IfElse"] = OperatorSpec("IfElse", 3, pit_if_else)
        return cls(
            specs,
            version=(
                "dataframe_pit_v3_split_history_and_rank_masks_"
                "ifelse_condition_axis_broadcast"
            ),
            semantics_revision=DAILY_IFELSE_SEMANTICS_REVISION,
        )

    def get(self, name: str) -> OperatorSpec | None:
        return self._specs.get(str(name))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    @property
    def digest(self) -> str:
        payload = {
            "version": self.version,
            "operators": [
                {
                    "name": spec.name,
                    "arity": spec.arity,
                    "window_argument": spec.window_argument,
                }
                for spec in sorted(self._specs.values(), key=lambda item: item.name)
            ],
        }
        if self.semantics_revision is not None:
            payload["semantics_revision"] = self.semantics_revision
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def descriptor(self) -> dict[str, object]:
        """Serializable operator identity for protocol and artifact binding."""

        return {
            "version": self.version,
            "semantics_revision": self.semantics_revision,
            "digest": self.digest,
            "operators": list(self.names),
        }


class SafeExpressionInterpreter:
    """Execute the factor DSL without Python ``eval`` or attribute access.

    LLM output is data, never executable Python.  Only numeric constants, input
    field names and direct calls registered in ``OperatorRegistry`` are accepted.
    """

    def __init__(
        self,
        registry: OperatorRegistry | None = None,
        *,
        max_call_depth: int = 6,
        maximum_window: int = 252,
    ):
        self.registry = registry or OperatorRegistry.dataframe_v1()
        self.max_call_depth = int(max_call_depth)
        self.maximum_window = int(maximum_window)

    def validate(
        self,
        expression: str,
        *,
        allowed_fields: set[str] | frozenset[str] | None = None,
    ) -> ExpressionValidation:
        reasons: list[str] = []
        try:
            tree = ast.parse(str(expression), mode="eval")
        except SyntaxError as exc:
            return ExpressionValidation(
                is_valid=False,
                reasons=(f"syntax_error:{exc.msg}",),
                fields=(),
                operators=(),
                max_call_depth=0,
            )

        fields: set[str] = set()
        operators: set[str] = set()
        self._validate_node(
            tree.body,
            fields=fields,
            operators=operators,
            reasons=reasons,
        )
        depth = _max_call_depth(tree.body)
        if depth > self.max_call_depth:
            reasons.append(f"max_call_depth_exceeded:{depth}>{self.max_call_depth}")
        for field in fields:
            lowered = field.lower().replace("-", "").replace(" ", "")
            if any(token in lowered for token in _FUTURE_FIELD_TOKENS):
                reasons.append(f"future_or_label_field_forbidden:{field}")
            if allowed_fields is not None and field not in allowed_fields:
                reasons.append(f"field_not_allowed:{field}")
        return ExpressionValidation(
            is_valid=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            fields=tuple(sorted(fields)),
            operators=tuple(sorted(operators)),
            max_call_depth=depth,
        )

    def evaluate(
        self,
        expression: str,
        fields: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        validation = self.validate(
            expression,
            allowed_fields=frozenset(str(name) for name in fields),
        )
        if not validation.is_valid:
            raise ValueError(
                "invalid_factor_expression:" + ";".join(validation.reasons)
            )
        tree = ast.parse(str(expression), mode="eval")
        result = self._evaluate_node(tree.body, fields)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"factor expression must return DataFrame, got {type(result)!r}"
            )
        return result

    def _validate_node(
        self,
        node: ast.AST,
        *,
        fields: set[str],
        operators: set[str],
        reasons: list[str],
    ) -> None:
        if isinstance(node, ast.Name):
            fields.add(node.id)
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                reasons.append(
                    f"non_numeric_constant_forbidden:{type(node.value).__name__}"
                )
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                reasons.append("direct_operator_calls_only")
                return
            name = node.func.id
            operators.add(name)
            spec = self.registry.get(name)
            if spec is None:
                reasons.append(f"operator_not_registered:{name}")
            else:
                if len(node.args) != spec.arity:
                    reasons.append(
                        f"operator_arity:{name}:{len(node.args)}!={spec.arity}"
                    )
                if (
                    spec.window_argument is not None
                    and len(node.args) > spec.window_argument
                ):
                    window_node = node.args[spec.window_argument]
                    if (
                        not isinstance(window_node, ast.Constant)
                        or isinstance(window_node.value, bool)
                        or not isinstance(window_node.value, int)
                    ):
                        reasons.append(f"window_must_be_integer:{name}")
                    else:
                        window = int(window_node.value)
                        if window <= 0:
                            reasons.append(f"window_must_be_positive:{name}:{window}")
                        if window > self.maximum_window:
                            reasons.append(
                                f"window_exceeds_limit:{name}:{window}>{self.maximum_window}"
                            )
            if node.keywords:
                reasons.append(f"keyword_arguments_forbidden:{name}")
            for argument in node.args:
                self._validate_node(
                    argument,
                    fields=fields,
                    operators=operators,
                    reasons=reasons,
                )
            return
        reasons.append(f"syntax_node_forbidden:{type(node).__name__}")

    def _evaluate_node(
        self,
        node: ast.AST,
        fields: Mapping[str, pd.DataFrame],
    ) -> object:
        if isinstance(node, ast.Name):
            return fields[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            spec = self.registry.get(node.func.id)
            if spec is None:  # validate() already protects this branch.
                raise ValueError(f"operator_not_registered:{node.func.id}")
            arguments = [self._evaluate_node(item, fields) for item in node.args]
            return spec.function(*arguments)
        raise ValueError(f"syntax_node_forbidden:{type(node).__name__}")


def _max_call_depth(node: ast.AST) -> int:
    if isinstance(node, ast.Call):
        return 1 + max(
            (_max_call_depth(item) for item in node.args),
            default=0,
        )
    return 0
