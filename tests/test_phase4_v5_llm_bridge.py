from __future__ import annotations

from dataclasses import replace

import pytest

from alpha_research.agents import (
    AgentContext,
    AgentRole,
    AgentTask,
    V5StructuredLLMBridge,
)
from alpha_research.experiments import (
    DataPartition,
    ExperimentProfile,
    ExperimentSpec,
    ResourceBudget,
    RetryPolicy,
)
from factor_production.v5.llm.agents import StructuredCallExecutor
from factor_production.v5.llm.budget import LLMBudget, LLMBudgetLimits
from factor_production.v5.llm.domain import LLMRole, Usage
from factor_production.v5.llm.ledger import CallLedger
from factor_production.v5.llm.policy import LLMPolicy
from factor_production.v5.llm.schemas import schema_hash
from factor_production.v5.llm.transport.fake import FakeTransport


def _hash(character: str) -> str:
    return character * 64


def _candidate() -> dict[str, object]:
    return {
        "candidate_id": "liquidity_reversal",
        "hypothesis": "A short-lived mechanism can leave a cross-sectional signal.",
        "expression": "Neg(TsMean(Returns, 5))",
        "direction": 1,
        "frequency": "daily",
        "family": "behavioral_reversal",
        "required_fields": ["Returns"],
        "parent_ids": [],
        "aggregation": {
            "method": "none",
            "window": "full_day",
            "smoothing_span": 0,
        },
        "tags": ["mechanism_first"],
    }


def _setup(tmp_path):
    policy = LLMPolicy()
    transport_authority_hash = _hash("8")
    spec = ExperimentSpec(
        experiment_id="phase4-llm-bridge",
        version="1",
        profile=ExperimentProfile.RESEARCH,
        data_partitions={
            DataPartition.DISCOVERY: _hash("a"),
            DataPartition.TRAIN: _hash("b"),
            DataPartition.TEST: _hash("c"),
        },
        agent_visible_partitions=(DataPartition.DISCOVERY, DataPartition.TRAIN),
        factor_spec_hashes=(_hash("d"),),
        label_spec_hash=_hash("e"),
        validation_spec_hash=_hash("f"),
        evaluation_spec_hash=_hash("1"),
        model_spec_hash=None,
        portfolio_spec_hash=None,
        cost_model_hash=None,
        robustness_spec_hash=None,
        code_snapshot_hash=_hash("2"),
        environment_hash=_hash("3"),
        llm_policy_hash=policy.content_hash,
        llm_transport_hash=transport_authority_hash,
        stages=("factor_generation", "factor_evaluation", "report"),
        random_seed=29,
        resource_budget=ResourceBudget(
            maximum_attempts=10,
            maximum_wall_seconds=100.0,
            maximum_cpu_seconds=100.0,
            maximum_peak_memory_bytes=1_000_000,
            maximum_disk_write_bytes=1_000_000,
            maximum_parallel_tasks=1,
            per_stage_timeout_seconds=20.0,
            maximum_llm_calls=5,
            maximum_llm_tokens=200_000,
            maximum_llm_cost_microusd=2_000_000,
        ),
        retry_policy=RetryPolicy(
            maximum_attempts_per_stage=2,
            initial_backoff_seconds=1.0,
            maximum_backoff_seconds=2.0,
            backoff_multiplier=2.0,
            jitter_fraction=0.0,
            retryable_failure_codes=("provider_429",),
        ),
    )
    context = AgentContext.bind(
        spec,
        data_schema_hashes=(_hash("4"),),
        allowed_fields=("Returns", "Amount", "Volume", "Close", "VWAP"),
        allowed_operators=("Add", "Sub", "Mul", "Div", "Neg", "TsMean"),
        allowed_windows=(3, 5, 10, 20),
    )
    task = AgentTask.create(
        task_id="researcher-proposal-1",
        role=AgentRole.RESEARCHER,
        spec=spec,
        context=context,
        input_artifact_hashes=(_hash("5"),),
        output_schema_hash=schema_hash(LLMRole.PROPOSER),
        maximum_output_artifacts=2,
    )
    transport = FakeTransport(
        {
            LLMRole.PROPOSER: [
                {
                    "schema_version": "llm-proposer-output/v1",
                    "proposals": [_candidate()],
                }
            ]
        },
        usage=Usage(input_tokens=50, output_tokens=30, cost_microusd=1000),
    )
    limits = LLMBudgetLimits(
        max_calls=5,
        max_input_tokens=100_000,
        max_output_tokens=50_000,
        max_total_tokens=150_000,
        max_cost_microusd=2_000_000,
    )
    executor = StructuredCallExecutor(
        transport=transport,
        ledger=CallLedger(tmp_path / "llm.jsonl"),
        budget=LLMBudget(limits),
        policy=policy,
    )
    return spec, context, task, transport, executor, transport_authority_hash


def test_v5_llm_bridge_binds_task_context_request_response_usage_and_ledger(tmp_path) -> None:
    spec, context, task, transport, executor, transport_hash = _setup(tmp_path)
    bridge = V5StructuredLLMBridge(
        executor, transport_authority_hash=transport_hash
    )
    first = bridge.propose(
        spec=spec, task=task, context=context, requested_count=1
    )
    second = bridge.propose(
        spec=spec, task=task, context=context, requested_count=1
    )
    assert len(first.proposals) == 1
    assert first.proposals[0].candidate_id == "liquidity_reversal"
    assert first.evidence.agent_task_hash == task.content_hash
    assert first.evidence.agent_context_hash == context.content_hash
    assert first.evidence.transport_authority_hash == transport_hash
    assert first.evidence.output_semantic_hashes == (
        first.proposals[0].semantic_hash,
    )
    assert first.evidence.content_hash == second.evidence.content_hash
    assert transport.call_count == 1
    executor.ledger.verify_against_pin(executor.ledger.pin)


def test_bridge_rejects_wrong_role_context_or_excess_external_budget(tmp_path) -> None:
    spec, context, task, _, executor, transport_hash = _setup(tmp_path)
    bridge = V5StructuredLLMBridge(
        executor, transport_authority_hash=transport_hash
    )
    with pytest.raises(ValueError, match="Researcher"):
        bridge.propose(
            spec=spec,
            task=replace(task, role=AgentRole.PLANNER, allowed_capabilities=(
                "contract_catalog_read",
                "experiment_plan_write",
            )),
            context=context,
            requested_count=1,
        )
    with pytest.raises(ValueError, match="context binding"):
        bridge.propose(
            spec=spec,
            task=replace(task, context_hash=_hash("9")),
            context=context,
            requested_count=1,
        )
    smaller = replace(
        spec,
        resource_budget=replace(spec.resource_budget, maximum_llm_calls=1),
    )
    smaller_context = AgentContext.bind(
        smaller,
        data_schema_hashes=context.data_schema_hashes,
        allowed_fields=context.allowed_fields,
        allowed_operators=context.allowed_operators,
        allowed_windows=context.allowed_windows,
    )
    smaller_task = AgentTask.create(
        task_id="researcher-small-budget",
        role=AgentRole.RESEARCHER,
        spec=smaller,
        context=smaller_context,
        input_artifact_hashes=(_hash("5"),),
        output_schema_hash=schema_hash(LLMRole.PROPOSER),
        maximum_output_artifacts=1,
    )
    with pytest.raises(ValueError, match="exceeds ExperimentSpec"):
        bridge.propose(
            spec=smaller,
            task=smaller_task,
            context=smaller_context,
            requested_count=1,
        )
