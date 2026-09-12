from __future__ import annotations

import ast
from dataclasses import dataclass

from llm_alpha_mining.research.factors.spec import ComplexityBudget
from llm_alpha_mining.mining.dsl import OperatorRegistry


@dataclass(frozen=True, slots=True)
class FactorComplexity:
    ast_nodes: int
    call_count: int
    maximum_call_depth: int
    maximum_window: int
    field_count: int
    estimated_relative_ops_per_cell: int
    estimated_state_bytes_per_security: int = 0
    estimated_flops_per_cell: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "ast_nodes": self.ast_nodes,
            "call_count": self.call_count,
            "maximum_call_depth": self.maximum_call_depth,
            "maximum_window": self.maximum_window,
            "field_count": self.field_count,
            "estimated_relative_ops_per_cell": self.estimated_relative_ops_per_cell,
            "estimated_state_bytes_per_security": self.estimated_state_bytes_per_security,
            "estimated_flops_per_cell": self.estimated_flops_per_cell,
        }


def analyze_expression(
    expression: str,
    *,
    registry: OperatorRegistry,
) -> FactorComplexity:
    tree = ast.parse(str(expression), mode="eval")
    relevant = tuple(
        node
        for node in ast.walk(tree.body)
        if isinstance(node, (ast.Call, ast.Name, ast.Constant))
    )
    calls = tuple(node for node in relevant if isinstance(node, ast.Call))
    called_names = {node.func.id for node in calls if isinstance(node.func, ast.Name)}
    fields = {
        node.id
        for node in relevant
        if isinstance(node, ast.Name) and node.id not in called_names
    }
    maximum_window = 0
    estimated = 0
    for node in calls:
        if not isinstance(node.func, ast.Name):
            continue
        spec = registry.get(node.func.id)
        if spec is None:
            continue
        window = 1
        if spec.window_argument is not None and len(node.args) > spec.window_argument:
            value = node.args[spec.window_argument]
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                window = int(value.value)
                maximum_window = max(maximum_window, window)
        multiplier = 2 if node.func.id in {"TsCorr", "TsSkew", "TsKurt"} else 1
        estimated += max(1, window) * multiplier
    return FactorComplexity(
        ast_nodes=len(relevant),
        call_count=len(calls),
        maximum_call_depth=_call_depth(tree.body),
        maximum_window=maximum_window,
        field_count=len(fields),
        estimated_relative_ops_per_cell=estimated,
        estimated_state_bytes_per_security=(maximum_window * max(1, len(fields)) * 8),
        estimated_flops_per_cell=estimated,
    )


def enforce_complexity(
    complexity: FactorComplexity,
    budget: ComplexityBudget,
) -> None:
    limits = {
        "ast_nodes": (complexity.ast_nodes, budget.maximum_ast_nodes),
        "call_depth": (
            complexity.maximum_call_depth,
            budget.maximum_call_depth,
        ),
        "window": (complexity.maximum_window, budget.maximum_window),
        "fields": (complexity.field_count, budget.maximum_fields),
        "relative_ops_per_cell": (
            complexity.estimated_relative_ops_per_cell,
            budget.maximum_relative_ops_per_cell,
        ),
        "estimated_state_bytes_per_security": (
            complexity.estimated_state_bytes_per_security,
            budget.maximum_estimated_state_bytes_per_security,
        ),
        "estimated_flops_per_cell": (
            complexity.estimated_flops_per_cell,
            budget.maximum_estimated_flops_per_cell,
        ),
    }
    violations = [
        f"{name}:{actual}>{maximum}"
        for name, (actual, maximum) in limits.items()
        if actual > maximum
    ]
    if violations:
        raise ValueError("factor_complexity_budget_exceeded:" + ",".join(violations))


def _call_depth(node: ast.AST) -> int:
    if isinstance(node, ast.Call):
        return 1 + max((_call_depth(item) for item in node.args), default=0)
    return 0


__all__ = ["FactorComplexity", "analyze_expression", "enforce_complexity"]
