from __future__ import annotations

import ast
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from llm_alpha_mining.mining.dsl.interpreter import (
    OperatorRegistry,
    SafeExpressionInterpreter,
)
from llm_alpha_mining.mining.llm.agents import (
    ArbiterAgent,
    CriticAgent,
    ProposerAgent,
    RiskAgent,
    StructuredCallExecutor,
)
from llm_alpha_mining.mining.llm.safe_context import (
    SafeResearchContext,
    assert_safe_context,
)
from llm_alpha_mining.mining.llm.schemas import (
    CandidateDraft,
    Review,
    StructuredOutputError,
)


@dataclass(frozen=True, slots=True)
class PanelResult:
    selected: tuple[CandidateDraft, ...]
    proposed: tuple[CandidateDraft, ...]
    dsl_valid: tuple[CandidateDraft, ...]
    critic_reviews: tuple[Review, ...]
    risk_assessments: tuple[Review, ...]
    arbiter_decisions: tuple[Review, ...]
    local_rejections: Mapping[str, tuple[str, ...]]


def _windows(expression: str) -> tuple[int, ...]:
    tree = ast.parse(expression, mode="eval")
    windows: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id in {"TsRank", "TsMean", "TsStd", "TsDelta", "TsSkew", "TsKurt"}:
            position = 1
        elif node.func.id == "TsCorr":
            position = 2
        else:
            continue
        if len(node.args) > position and isinstance(node.args[position], ast.Constant):
            value = node.args[position].value
            if isinstance(value, int) and not isinstance(value, bool):
                windows.add(value)
    return tuple(sorted(windows))


class AlphaResearchPanel:
    """Proposer -> local DSL -> Critic/Risk -> Arbiter -> deterministic reducer."""

    def __init__(
        self, executor: StructuredCallExecutor, *, max_expression_depth: int = 6
    ) -> None:
        self.proposer = ProposerAgent(executor)
        self.critic = CriticAgent(executor)
        self.risk = RiskAgent(executor)
        self.arbiter = ArbiterAgent(executor)
        self.max_expression_depth = int(max_expression_depth)

    @staticmethod
    def _require_exact_coverage(
        role: str, reviews: tuple[Review, ...], ids: set[str]
    ) -> None:
        actual = {item.candidate_id for item in reviews}
        if actual != ids:
            raise StructuredOutputError(
                f"{role} candidate coverage mismatch; missing={sorted(ids - actual)}, extra={sorted(actual - ids)}"
            )

    def _preflight(
        self,
        draft: CandidateDraft,
        context: SafeResearchContext,
    ) -> tuple[str, ...]:
        # Apply the same DLP/teacher firewall to model output before it can be
        # forwarded to another role or persisted as a candidate.
        assert_safe_context(draft.to_dict())
        registry = OperatorRegistry.dataframe_v2()
        interpreter = SafeExpressionInterpreter(
            registry,
            max_call_depth=self.max_expression_depth,
            maximum_window=max(context.allowed_windows),
        )
        validation = interpreter.validate(
            draft.expression,
            allowed_fields=frozenset(context.allowed_fields),
        )
        reasons = list(validation.reasons)
        unsupported = set(validation.operators) - set(context.allowed_operators)
        if unsupported:
            reasons.append("operators_not_allowed:" + ",".join(sorted(unsupported)))
        invalid_windows = set(_windows(draft.expression)) - set(context.allowed_windows)
        if invalid_windows:
            reasons.append(
                "windows_not_allowed:" + ",".join(map(str, sorted(invalid_windows)))
            )
        if set(draft.required_fields) != set(validation.fields):
            reasons.append("required_fields_do_not_match_expression")
        if draft.frequency == "daily" and draft.aggregation["method"] != "none":
            reasons.append("daily_aggregation_must_be_none")
        if draft.frequency == "minute" and draft.aggregation["method"] == "none":
            reasons.append("minute_aggregation_must_be_explicit")
        if context.generation == 0 and draft.parent_ids:
            reasons.append("generation_zero_must_not_have_parents")
        if context.generation > 0 and not draft.parent_ids:
            reasons.append("derived_generation_requires_parent")
        catalog_ids = {item.candidate_id for item in context.parent_catalog}
        unknown_parents = set(draft.parent_ids) - catalog_ids
        if unknown_parents:
            reasons.append("unknown_parent_ids:" + ",".join(sorted(unknown_parents)))
        return tuple(dict.fromkeys(reasons))

    def run(self, context: SafeResearchContext) -> PanelResult:
        proposed = tuple(self.proposer.call(context.to_dict()))
        if len(proposed) > context.requested_count:
            raise StructuredOutputError("proposer exceeded requested_count")
        local_rejections: dict[str, tuple[str, ...]] = {}
        valid: list[CandidateDraft] = []
        seen_semantics: set[str] = set()
        prior_semantics = {item.semantic_hash for item in context.parent_catalog}
        for draft in proposed:
            reasons = self._preflight(draft, context)
            if (
                draft.semantic_hash in seen_semantics
                or draft.semantic_hash in prior_semantics
            ):
                reasons = tuple(reasons) + ("duplicate_semantic_signal",)
            if reasons:
                local_rejections[draft.candidate_id] = reasons
            else:
                valid.append(draft)
                seen_semantics.add(draft.semantic_hash)
        valid_tuple = tuple(valid)
        if not valid_tuple:
            return PanelResult(
                selected=(),
                proposed=proposed,
                dsl_valid=(),
                critic_reviews=(),
                risk_assessments=(),
                arbiter_decisions=(),
                local_rejections=MappingProxyType(local_rejections),
            )

        role_context = context.to_dict(candidate_drafts=valid_tuple)
        critic = tuple(self.critic.call(role_context))
        risk = tuple(self.risk.call(role_context))
        ids = {item.candidate_id for item in valid_tuple}
        self._require_exact_coverage("critic", critic, ids)
        self._require_exact_coverage("risk", risk, ids)
        arbiter_context = context.to_dict(
            candidate_drafts=valid_tuple,
            critic_reviews=critic,
            risk_assessments=risk,
        )
        arbiter = tuple(self.arbiter.call(arbiter_context))
        self._require_exact_coverage("arbiter", arbiter, ids)

        critic_by_id = {item.candidate_id: item for item in critic}
        risk_by_id = {item.candidate_id: item for item in risk}
        arbiter_by_id = {item.candidate_id: item for item in arbiter}
        selected = [
            item
            for item in valid_tuple
            if critic_by_id[item.candidate_id].decision == "approve"
            and risk_by_id[item.candidate_id].decision == "allow"
            and arbiter_by_id[item.candidate_id].decision == "select"
        ]
        # Risk blocks are an immutable veto; priority cannot override them.
        selected.sort(
            key=lambda item: (
                -int(arbiter_by_id[item.candidate_id].priority or 0),
                item.candidate_id,
            )
        )
        return PanelResult(
            selected=tuple(selected[: context.requested_count]),
            proposed=proposed,
            dsl_valid=valid_tuple,
            critic_reviews=critic,
            risk_assessments=risk,
            arbiter_decisions=arbiter,
            local_rejections=MappingProxyType(local_rejections),
        )
