from __future__ import annotations

from dataclasses import dataclass

from alpha_research.agents.contracts import AgentContext, AgentRole, AgentTask
from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.experiments.spec import ExperimentSpec
from factor_production.v5.llm.agents import ProposerAgent, StructuredCallExecutor
from factor_production.v5.llm.safe_context import SafeResearchContext
from factor_production.v5.llm.schemas import CandidateDraft, parse_role_output


@dataclass(frozen=True, slots=True)
class LLMCallEvidence:
    agent_task_hash: str
    agent_context_hash: str
    request_hash: str
    response_hash: str
    model_id_hash: str
    provider_request_hash: str
    usage_hash: str
    transport_authority_hash: str
    ledger_head_hash: str
    ledger_file_hash: str
    ledger_record_count: int
    output_semantic_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "agent_task_hash",
            "agent_context_hash",
            "request_hash",
            "response_hash",
            "model_id_hash",
            "provider_request_hash",
            "usage_hash",
            "transport_authority_hash",
            "ledger_head_hash",
            "ledger_file_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"LLM evidence {name}")
        if (
            not isinstance(self.ledger_record_count, int)
            or self.ledger_record_count <= 0
        ):
            raise ValueError("LLM evidence ledger record count must be positive")
        if len(set(self.output_semantic_hashes)) != len(self.output_semantic_hashes):
            raise ValueError("LLM evidence output semantic hashes must be unique")
        for digest in self.output_semantic_hashes:
            require_sha256(digest, name="LLM evidence output semantic hash")

    @property
    def content_hash(self) -> str:
        return hash_json(
            {
                "agent_task_hash": self.agent_task_hash,
                "agent_context_hash": self.agent_context_hash,
                "request_hash": self.request_hash,
                "response_hash": self.response_hash,
                "model_id_hash": self.model_id_hash,
                "provider_request_hash": self.provider_request_hash,
                "usage_hash": self.usage_hash,
                "transport_authority_hash": self.transport_authority_hash,
                "ledger_head_hash": self.ledger_head_hash,
                "ledger_file_hash": self.ledger_file_hash,
                "ledger_record_count": self.ledger_record_count,
                "output_semantic_hashes": list(self.output_semantic_hashes),
            }
        )


@dataclass(frozen=True, slots=True)
class ProposalCallResult:
    proposals: tuple[CandidateDraft, ...]
    evidence: LLMCallEvidence


class V5StructuredLLMBridge:
    """Bind the mature V5 safe LLM ledger to the Phase 4 AgentTask contract."""

    def __init__(
        self,
        executor: StructuredCallExecutor,
        *,
        transport_authority_hash: str,
    ) -> None:
        self.executor = executor
        self.transport_authority_hash = require_sha256(
            transport_authority_hash, name="LLM bridge transport authority"
        )

    def propose(
        self,
        *,
        spec: ExperimentSpec,
        task: AgentTask,
        context: AgentContext,
        requested_count: int,
    ) -> ProposalCallResult:
        if task.role is not AgentRole.RESEARCHER:
            raise ValueError("only a Researcher AgentTask can propose factors")
        if task.experiment_spec_hash != spec.content_hash:
            raise ValueError("LLM task belongs to another experiment")
        if task.context_hash != context.content_hash:
            raise ValueError("LLM task context binding differs")
        if spec.llm_policy_hash != self.executor.policy.content_hash:
            raise ValueError("LLM policy differs from ExperimentSpec")
        if spec.llm_transport_hash != self.transport_authority_hash:
            raise ValueError("LLM transport authority differs from ExperimentSpec")
        limits = self.executor.budget.limits
        budget = spec.resource_budget
        if (
            limits.max_calls > budget.maximum_llm_calls
            or limits.max_total_tokens > budget.maximum_llm_tokens
            or limits.max_cost_microusd > budget.maximum_llm_cost_microusd
        ):
            raise ValueError("V5 LLM budget exceeds ExperimentSpec authority")
        protocol_hash = hash_json(
            {
                "experiment_spec_hash": spec.content_hash,
                "agent_task_hash": task.content_hash,
                "agent_context_hash": context.content_hash,
                "llm_policy_hash": self.executor.policy.content_hash,
                "llm_transport_hash": self.transport_authority_hash,
            }
        )
        safe = SafeResearchContext(
            campaign_hash=spec.content_hash,
            protocol_hash=protocol_hash,
            generation=0,
            requested_count=requested_count,
            allowed_fields=context.allowed_fields,
            allowed_operators=context.allowed_operators,
            allowed_windows=context.allowed_windows,
        )
        agent = ProposerAgent(self.executor)
        request = agent.build_request(safe.to_dict())
        response = self.executor.execute(request)
        parsed = parse_role_output(request.role, response.output)
        if not isinstance(parsed, tuple) or any(
            not isinstance(item, CandidateDraft) for item in parsed
        ):
            raise RuntimeError("V5 proposer returned an unexpected typed payload")
        proposals: tuple[CandidateDraft, ...] = parsed
        pin = self.executor.ledger.pin
        semantic_hashes = tuple(item.semantic_hash for item in proposals)
        evidence = LLMCallEvidence(
            agent_task_hash=task.content_hash,
            agent_context_hash=context.content_hash,
            request_hash=request.request_hash,
            response_hash=hash_json(response.to_dict()),
            model_id_hash=hash_json({"model_id": response.model_id}),
            provider_request_hash=hash_json(
                {"provider_request_id": response.provider_request_id}
            ),
            usage_hash=hash_json(response.usage.to_dict()),
            transport_authority_hash=self.transport_authority_hash,
            ledger_head_hash=pin.head_hash,
            ledger_file_hash=pin.file_hash,
            ledger_record_count=pin.record_count,
            output_semantic_hashes=semantic_hashes,
        )
        return ProposalCallResult(proposals=proposals, evidence=evidence)


__all__ = ["LLMCallEvidence", "ProposalCallResult", "V5StructuredLLMBridge"]
