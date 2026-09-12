from __future__ import annotations

import json
import concurrent.futures
import hashlib
import threading
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

from llm_alpha_mining.mining.artifacts.hashing import (
    canonical_json_bytes,
    hash_file,
    hash_json,
)
from llm_alpha_mining.mining.llm.agents import (
    CriticAgent,
    ProposerAgent,
    StructuredCallExecutor,
    StructuredCallFailed,
)
from llm_alpha_mining.mining.llm.budget import (
    LLMBudget,
    LLMBudgetExceeded,
    LLMBudgetLimits,
)
from llm_alpha_mining.mining.llm.domain import (
    LLMContractError,
    LLMRole,
    StructuredCallResponse,
    Usage,
    hash_provider_request_id,
)
from llm_alpha_mining.mining.llm.ledger import (
    CallLedger,
    IndeterminateCallError,
    LedgerIntegrityError,
)
from llm_alpha_mining.mining.llm.panel import AlphaResearchPanel
from llm_alpha_mining.mining.llm.policy import DEFAULT_PROMPTS, LLMPolicy
from llm_alpha_mining.mining.llm.safe_context import (
    ParentCatalogEntry,
    SafeResearchContext,
    UnsafeLLMContext,
    assert_safe_context,
    validate_safe_context_payload,
)
from llm_alpha_mining.mining.llm.schemas import (
    ROLE_SCHEMAS,
    StructuredOutputError,
    parse_role_output,
    CandidateDraft,
    validate_strict_json,
)
from llm_alpha_mining.mining.llm.transport.base import (
    NetworkDeniedTransport,
    StructuredTransport,
    TransportError,
)
from llm_alpha_mining.mining.llm.transport.fake import FakeTransport
from llm_alpha_mining.mining.llm.transport.replay import (
    ExactReplayTransport,
    ReplayEntry,
    load_replay_cassette,
    write_replay_cassette,
)
from llm_alpha_mining.mining.protocol import default_protocol
from llm_alpha_mining.mining.domain.models import CandidateSpec
from llm_alpha_mining.mining.orchestration.hypotheses import semantic_signal_hash
from llm_alpha_mining.mining.providers.base import LocalFeedback, ProposalContext
from llm_alpha_mining.mining.providers.llm_panel import LLMPanelProvider


def _candidate(
    candidate_id: str, expression: str, fields: list[str]
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "hypothesis": "A short-lived mechanism can leave a cross-sectional signal.",
        "expression": expression,
        "direction": 1,
        "frequency": "daily",
        "family": "behavioral_reversal",
        "required_fields": fields,
        "parent_ids": [],
        "aggregation": {"method": "none", "window": "full_day", "smoothing_span": 0},
        "tags": ["mechanism_first"],
    }


def _scripts() -> dict[LLMRole, list[dict[str, object]]]:
    return {
        LLMRole.PROPOSER: [
            {
                "schema_version": "llm-proposer-output/v1",
                "proposals": [
                    _candidate(
                        "liquidity_reversal", "Neg(TsMean(Returns, 5))", ["Returns"]
                    ),
                    _candidate(
                        "impact_blocked", "Div(Returns, Amount)", ["Returns", "Amount"]
                    ),
                ],
            }
        ],
        LLMRole.CRITIC: [
            {
                "schema_version": "llm-critic-output/v1",
                "reviews": [
                    {
                        "candidate_id": "liquidity_reversal",
                        "verdict": "approve",
                        "reason_codes": ["mechanism_supported"],
                    },
                    {
                        "candidate_id": "impact_blocked",
                        "verdict": "approve",
                        "reason_codes": ["mechanism_supported"],
                    },
                ],
            }
        ],
        LLMRole.RISK: [
            {
                "schema_version": "llm-risk-output/v1",
                "assessments": [
                    {
                        "candidate_id": "liquidity_reversal",
                        "decision": "allow",
                        "reason_codes": ["no_blocker"],
                    },
                    {
                        "candidate_id": "impact_blocked",
                        "decision": "block",
                        "reason_codes": ["unsafe_semantics"],
                    },
                ],
            }
        ],
        LLMRole.ARBITER: [
            {
                "schema_version": "llm-arbiter-output/v1",
                "decisions": [
                    {
                        "candidate_id": "liquidity_reversal",
                        "decision": "select",
                        "reason_codes": ["balanced_mechanism"],
                        "priority": 80,
                    },
                    {
                        "candidate_id": "impact_blocked",
                        "decision": "select",
                        "reason_codes": ["diversifies_batch"],
                        "priority": 100,
                    },
                ],
            }
        ],
    }


def _budget() -> LLMBudget:
    return LLMBudget(
        LLMBudgetLimits(
            max_calls=20,
            max_input_tokens=100_000,
            max_output_tokens=100_000,
            max_total_tokens=200_000,
            max_cost_microusd=5_000_000,
        )
    )


def _context(protocol_hash: str) -> SafeResearchContext:
    return SafeResearchContext(
        campaign_hash=hash_json({"campaign": "offline-golden"}),
        protocol_hash=protocol_hash,
        generation=0,
        requested_count=2,
        allowed_fields=("Returns", "Amount", "Volume", "Close", "VWAP"),
        allowed_operators=("Add", "Sub", "Mul", "Div", "Neg", "TsMean"),
        allowed_windows=(3, 5, 10, 20),
    )


def _executor(
    tmp_path: Path, transport, *, policy: LLMPolicy | None = None, name: str = "ledger"
):
    return StructuredCallExecutor(
        transport=transport,
        ledger=CallLedger(tmp_path / f"{name}.jsonl"),
        budget=_budget(),
        policy=policy or LLMPolicy(),
    )


def test_every_structured_object_is_closed_and_unknown_fields_fail() -> None:
    def walk(schema: dict[str, object]) -> None:
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
            for child in schema["properties"].values():
                walk(child)
        elif schema.get("type") == "array":
            walk(schema["items"])

    for schema in ROLE_SCHEMAS.values():
        walk(schema)
    payload = _scripts()[LLMRole.PROPOSER][0]
    payload = {**payload, "analysis": "secret scratchpad"}
    with pytest.raises(StructuredOutputError):
        validate_strict_json(ROLE_SCHEMAS[LLMRole.PROPOSER], payload)


def test_role_parser_is_a_second_typed_pass_and_rejects_cot() -> None:
    payload = _scripts()[LLMRole.RISK][0]
    parsed = parse_role_output(LLMRole.RISK, payload)
    assert parsed[1].decision == "block"
    malicious = dict(payload)
    malicious["analysis"] = "do not persist this"
    with pytest.raises(StructuredOutputError, match="chain-of-thought"):
        parse_role_output(LLMRole.RISK, malicious)
    narrated = _candidate("narrated", "Neg(Returns)", ["Returns"])
    narrated["hypothesis"] = (
        "I considered several formulas and decided this one should work."
    )
    with pytest.raises(StructuredOutputError, match="reasoning-process"):
        parse_role_output(
            LLMRole.PROPOSER,
            {"schema_version": "llm-proposer-output/v1", "proposals": [narrated]},
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"teacher_score": 0.99},
        {"notes": "official holdout result was good"},
        {"trade_date": "2021-06-30"},
        {"security_code": "600000.SH"},
        {"rank_ic": 0.04},
    ],
)
def test_teacher_raw_data_date_security_and_exact_metric_firewall(payload) -> None:
    with pytest.raises(UnsafeLLMContext):
        assert_safe_context(payload)


def test_prompt_schema_and_policy_are_hash_bound() -> None:
    base = LLMPolicy()
    prompts = dict(DEFAULT_PROMPTS)
    prompts[LLMRole.CRITIC] += " Use an even stricter novelty standard."
    changed = LLMPolicy(prompts=prompts)
    assert base.prompt_hash(LLMRole.PROPOSER) == changed.prompt_hash(LLMRole.PROPOSER)
    assert base.prompt_hash(LLMRole.CRITIC) != changed.prompt_hash(LLMRole.CRITIC)
    assert base.content_hash != changed.content_hash


def test_golden_vertical_slice_risk_veto_and_provider_adapter(tmp_path: Path) -> None:
    protocol = default_protocol("offline-panel-test")
    fake = FakeTransport(_scripts())
    executor = _executor(tmp_path, fake)
    panel = AlphaResearchPanel(
        executor, max_expression_depth=protocol.max_expression_depth
    )
    provider = LLMPanelProvider(protocol=protocol, panel=panel)
    batch = provider.propose(
        ProposalContext(
            run_id="opaque-run-name",
            protocol_hash=protocol.content_hash,
            generation=0,
            request_count=2,
        )
    )
    assert [item.candidate_id for item in batch.candidates] == ["liquidity_reversal"]
    assert provider.last_result is not None
    assert provider.last_result.risk_assessments[1].decision == "block"
    # Arbiter ranked the blocked item first, but the deterministic reducer cannot override Risk.
    assert provider.last_result.arbiter_decisions[1].priority == 100
    assert fake.call_count == 4
    assert executor.budget.snapshot.calls == 4
    assert executor.budget.snapshot.cost_microusd == 400
    for request in fake.requests:
        assert request.policy_hash == executor.policy.content_hash
        assert request.prompt_hash == executor.policy.prompt_hash(request.role)
        assert request.schema_hash
        assert "opaque-run-name" not in json.dumps(request.to_dict())
        if request.role is not LLMRole.PROPOSER:
            serialized = json.dumps(request.to_dict())
            assert "hypothesis" not in serialized
            assert "short-lived mechanism" not in serialized
            assert request.context["candidate_descriptors"]


def test_local_dsl_preflight_runs_before_review_roles(tmp_path: Path) -> None:
    bad = _candidate("future_leak", "TsMean(FutureReturn, 5)", ["FutureReturn"])
    scripts = {
        LLMRole.PROPOSER: [
            {"schema_version": "llm-proposer-output/v1", "proposals": [bad]}
        ]
    }
    protocol = default_protocol("dsl-preflight")
    fake = FakeTransport(scripts)
    panel = AlphaResearchPanel(_executor(tmp_path, fake))
    result = panel.run(_context(protocol.content_hash))
    assert result.selected == ()
    assert "future_leak" in result.local_rejections
    assert fake.call_count_by_role[LLMRole.PROPOSER] == 1
    assert fake.call_count_by_role[LLMRole.CRITIC] == 0


def test_exact_replay_reproduces_all_four_roles_without_network(tmp_path: Path) -> None:
    protocol = default_protocol("exact-replay")
    context = _context(protocol.content_hash)
    fake = FakeTransport(_scripts())
    first = AlphaResearchPanel(_executor(tmp_path, fake, name="record")).run(context)
    entries = [
        ReplayEntry.from_response(response, request=request)
        for request, response in zip(
            fake.requests, fake.recorded_responses, strict=True
        )
    ]
    request_by_hash = {request.request_hash: request for request in fake.requests}
    replay = ExactReplayTransport(entries, request_by_hash=request_by_hash)
    second = AlphaResearchPanel(_executor(tmp_path, replay, name="replay")).run(context)
    assert [item.to_dict() for item in first.selected] == [
        item.to_dict() for item in second.selected
    ]
    assert replay.call_count == 4
    denied = NetworkDeniedTransport()
    assert denied.call_count == 0


def test_replay_cassette_is_content_bound_and_tamper_evident(tmp_path: Path) -> None:
    protocol = default_protocol("cassette")
    fake = FakeTransport({LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]})
    ProposerAgent(_executor(tmp_path, fake, name="cassette-record")).call(
        _context(protocol.content_hash).to_dict()
    )
    request = fake.requests[0]
    request_by_hash = {request.request_hash: request}
    path = write_replay_cassette(
        [ReplayEntry.from_response(fake.recorded_responses[0], request=request)],
        tmp_path / "cassette.jsonl",
        request_by_hash=request_by_hash,
    )
    cassette_hash = hash_file(path)
    with pytest.raises(ValueError, match="external expected"):
        load_replay_cassette(path, request_by_hash=request_by_hash)
    loaded = load_replay_cassette(
        path,
        request_by_hash=request_by_hash,
        expected_file_hash=cassette_hash,
    )
    assert loaded[0].request_hash == fake.recorded_responses[0].request_hash
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "liquidity_reversal", "liquidity_tampered"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cassette file hash mismatch"):
        load_replay_cassette(
            path,
            request_by_hash=request_by_hash,
            expected_file_hash=cassette_hash,
        )


def test_exact_replay_refuses_near_match(tmp_path: Path) -> None:
    protocol = default_protocol("replay-miss")
    fake = FakeTransport(_scripts())
    agent = ProposerAgent(_executor(tmp_path, fake, name="capture"))
    original = _context(protocol.content_hash).to_dict()
    agent.call(original)
    request = fake.requests[0]
    replay = ExactReplayTransport(
        [ReplayEntry.from_response(fake.recorded_responses[0], request=request)],
        request_by_hash={request.request_hash: request},
    )
    replay_agent = ProposerAgent(_executor(tmp_path, replay, name="miss"))
    changed = dict(original)
    changed["requested_count"] = 1
    with pytest.raises(StructuredCallFailed):
        replay_agent.call(changed)
    assert replay.call_count == 1


def test_replay_entry_cannot_persist_unsafe_output_or_metadata(tmp_path: Path) -> None:
    request = ProposerAgent(
        _executor(tmp_path, FakeTransport({}), name="unsafe-replay-entry")
    ).build_request(
        _context(default_protocol("unsafe-replay-entry").content_hash).to_dict()
    )
    with pytest.raises(UnsafeLLMContext):
        ReplayEntry(
            request=request,
            output={"notes": "off\u200bicial holdout"},
            usage=Usage(1, 1, 1),
        )
    with pytest.raises(UnsafeLLMContext):
        ReplayEntry(
            request=request,
            output={"safe": "value"},
            usage=Usage(1, 1, 1),
            model_id="teacher-model",
        )


def test_network_denied_transport_is_executable_guard(tmp_path: Path) -> None:
    protocol = default_protocol("network-denied")
    denied = NetworkDeniedTransport()
    agent = ProposerAgent(_executor(tmp_path, denied))
    with pytest.raises(StructuredCallFailed, match="transport_terminal_failure"):
        agent.call(_context(protocol.content_hash).to_dict())
    assert denied.call_count == 1
    assert agent.executor.budget.snapshot.cost_microusd == 0
    assert agent.executor.budget.snapshot.pending_reservations == 0


def test_finite_retry_uses_same_idempotency_key_and_counts_calls(
    tmp_path: Path,
) -> None:
    output = _scripts()[LLMRole.PROPOSER][0]
    fake = FakeTransport(
        {
            LLMRole.PROPOSER: [
                TransportError("pre-dispatch rate limit", retryable=True),
                output,
            ]
        }
    )
    policy = LLMPolicy(max_retries=1, circuit_failure_threshold=3)
    executor = _executor(tmp_path, fake, policy=policy)
    protocol = default_protocol("retry")
    result = ProposerAgent(executor).call(_context(protocol.content_hash).to_dict())
    assert len(result) == 2
    assert fake.call_count == 2
    assert len(set(fake.idempotency_keys)) == 1
    assert executor.budget.snapshot.calls == 2
    assert executor.budget.snapshot.cost_microusd == fake.usage.cost_microusd


def test_timeout_is_indeterminate_and_restart_never_duplicates_call(
    tmp_path: Path,
) -> None:
    protocol = default_protocol("crash-recovery")
    fake = FakeTransport(
        {LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]},
        delay_seconds=0.05,
    )
    policy = LLMPolicy(timeout_seconds=0.001, max_retries=2)
    ledger_path = tmp_path / "crash.jsonl"
    first_ledger = CallLedger(ledger_path)
    first_executor = StructuredCallExecutor(
        transport=fake,
        ledger=first_ledger,
        budget=_budget(),
        policy=policy,
    )
    context = _context(protocol.content_hash).to_dict()
    with pytest.raises(IndeterminateCallError):
        ProposerAgent(first_executor).call(context)
    assert fake.call_count == 1
    pin = first_ledger.pin
    reopened = CallLedger(
        ledger_path,
        expected_head_hash=pin.head_hash,
        expected_file_hash=pin.file_hash,
    )
    recovered_budget = LLMBudget.from_ledger(_budget().limits, reopened)
    assert recovered_budget.snapshot.calls == 1
    assert recovered_budget.snapshot.pending_reservations == 0
    restarted = StructuredCallExecutor(
        transport=fake,
        ledger=reopened,
        budget=recovered_budget,
        policy=policy,
    )
    with pytest.raises(IndeterminateCallError, match="duplicate call denied"):
        ProposerAgent(restarted).call(context)
    assert fake.call_count == 1


def test_crash_after_atomic_reserve_retains_pending_maximum_and_denies_redispatch(
    tmp_path: Path,
) -> None:
    protocol = default_protocol("reserve-crash")
    path = tmp_path / "reserve-crash.jsonl"
    ledger = CallLedger(path)
    budget = _budget()
    fake = FakeTransport({LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]})
    executor = StructuredCallExecutor(
        transport=fake,
        ledger=ledger,
        budget=budget,
        policy=LLMPolicy(),
    )
    agent = ProposerAgent(executor)
    context = _context(protocol.content_hash).to_dict()
    request = agent.build_request(context)
    reservation = {
        "input_tokens": 100,
        "output_tokens": request.max_output_tokens,
        "cost_microusd": request.max_cost_microusd,
    }
    ledger.reserve_and_start(
        request,
        attempt=1,
        reservation=reservation,
        limits=budget.limits,
    )
    pin = ledger.pin
    reopened = CallLedger(
        path,
        expected_head_hash=pin.head_hash,
        expected_file_hash=pin.file_hash,
    )
    recovered = LLMBudget.from_ledger(budget.limits, reopened)
    assert recovered.snapshot.calls == 1
    assert recovered.snapshot.pending_reservations == 1
    assert recovered.snapshot.cost_microusd == request.max_cost_microusd
    restarted = StructuredCallExecutor(
        transport=fake,
        ledger=reopened,
        budget=recovered,
        policy=LLMPolicy(),
    )
    with pytest.raises(IndeterminateCallError, match="duplicate call denied"):
        ProposerAgent(restarted).call(context)
    assert fake.call_count == 0


def test_ledger_hash_chain_detects_tampering(tmp_path: Path) -> None:
    protocol = default_protocol("ledger-tamper")
    fake = FakeTransport({LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]})
    path = tmp_path / "ledger.jsonl"
    executor = StructuredCallExecutor(
        transport=fake,
        ledger=CallLedger(path),
        budget=_budget(),
        policy=LLMPolicy(),
    )
    ProposerAgent(executor).call(_context(protocol.content_hash).to_dict())
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("call_succeeded", "call_tampered", 1), encoding="utf-8"
    )
    with pytest.raises(LedgerIntegrityError, match="hash mismatch"):
        CallLedger(path)


def test_budget_reserves_calls_tokens_and_integer_microusd() -> None:
    budget = LLMBudget(
        LLMBudgetLimits(
            max_calls=1,
            max_input_tokens=10,
            max_output_tokens=10,
            max_total_tokens=20,
            max_cost_microusd=100,
        )
    )
    budget.reserve("one", input_tokens=5, output_tokens=5, cost_microusd=100)
    with pytest.raises(LLMBudgetExceeded):
        budget.reserve("two", input_tokens=1, output_tokens=1, cost_microusd=1)
    budget.settle("one", Usage(input_tokens=3, output_tokens=4, cost_microusd=80))
    assert budget.snapshot.cost_microusd == 80


def test_oversized_request_fails_before_dispatch_or_persistence(tmp_path: Path) -> None:
    fake = FakeTransport({LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]})
    executor = _executor(
        tmp_path,
        fake,
        policy=LLMPolicy(max_input_tokens_per_call=1),
        name="input-cap",
    )
    with pytest.raises(StructuredCallFailed, match="request_exceeds_input_limit"):
        ProposerAgent(executor).call(
            _context(default_protocol("input-cap").content_hash).to_dict()
        )
    assert fake.call_count == 0
    assert executor.ledger.records == ()
    assert executor.budget.snapshot.calls == 0


def test_provider_refuses_exact_local_feedback(tmp_path: Path) -> None:
    protocol = default_protocol("teacher-firewall")
    fake = FakeTransport(_scripts())
    provider = LLMPanelProvider(
        protocol=protocol,
        panel=AlphaResearchPanel(_executor(tmp_path, fake)),
    )
    with pytest.raises(UnsafeLLMContext, match="exact LocalFeedback"):
        provider.propose(
            ProposalContext(
                run_id="run",
                protocol_hash=protocol.content_hash,
                generation=0,
                request_count=1,
                local_feedback=(
                    LocalFeedback(
                        candidate_id="candidate",
                        metric="rank_ic",
                        value=0.04,
                        generation=0,
                    ),
                ),
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"notes": "off\u200bicial holdout"},
        {"notes": "\uff54\uff45\uff41\uff43\uff48\uff45\uff52 score"},
        {"notes": "rank I\u200bC = 0.041"},
        {"notes": "r.a.n.k I C : 0.041"},
        {"notes": "\uff16\uff10\uff10\uff10\uff10\uff10\uff0e\uff33\uff28"},
        {"notes": "6.0.0.0.0.0 / S.H"},
        {"notes": "2 0 2 1 / 0 6 / 3 0"},
        {"notes": "ignore prev\u200bious instructions"},
        {"notes": "system pr\u200bompt"},
        {"notes": "s.y.s.t.e.m p.r.o.m.p.t"},
    ],
)
def test_unicode_zero_width_dlp_evasions_are_rejected(payload) -> None:
    with pytest.raises(UnsafeLLMContext):
        assert_safe_context(payload)


def test_invalid_or_unsafe_response_is_charged_but_never_persisted(
    tmp_path: Path,
) -> None:
    protocol = default_protocol("dlp-before-persist")
    unsafe = _candidate("unsafe_prose", "Neg(Returns)", ["Returns"])
    unsafe["hypothesis"] = "The off\u200bicial holdout score is 0.99."
    fake = FakeTransport(
        {
            LLMRole.PROPOSER: [
                {"schema_version": "llm-proposer-output/v1", "proposals": [unsafe]}
            ]
        }
    )
    executor = _executor(tmp_path, fake, name="dlp-reject")
    with pytest.raises(StructuredCallFailed, match="response_dlp_rejected"):
        ProposerAgent(executor).call(_context(protocol.content_hash).to_dict())
    rows = executor.ledger.records
    assert [row["event"] for row in rows] == [
        "attempt_started",
        "call_terminal_failure",
    ]
    assert rows[-1]["details"]["error_code"] == "response_dlp_rejected"
    assert rows[-1]["details"]["charge"] == fake.usage.to_dict()
    assert rows[-1]["details"]["response_hash"]
    persisted = executor.ledger.path.read_text(encoding="utf-8")
    assert "holdout" not in persisted
    assert "0.99" not in persisted
    assert "unsafe_prose" not in persisted
    assert executor.budget.snapshot.cost_microusd == fake.usage.cost_microusd


def test_role_semantic_failure_is_not_cached_and_is_charged(tmp_path: Path) -> None:
    invalid = {
        "schema_version": "llm-risk-output/v1",
        "assessments": [
            {
                "candidate_id": "liquidity_reversal",
                "decision": "allow",
                "reason_codes": ["unsafe_semantics"],
            }
        ],
    }
    fake = FakeTransport({LLMRole.RISK: [invalid]})
    executor = _executor(tmp_path, fake, name="role-invalid")
    from llm_alpha_mining.mining.llm.agents import RiskAgent

    with pytest.raises(StructuredCallFailed, match="structured_output_invalid"):
        RiskAgent(executor).call(
            _context(default_protocol("role-invalid").content_hash).to_dict()
        )
    assert all(row["event"] != "call_succeeded" for row in executor.ledger.records)
    assert executor.budget.snapshot.cost_microusd == fake.usage.cost_microusd


class _UnsafeMetadataTransport(FakeTransport):
    def complete(self, request, *, idempotency_key, timeout_seconds):
        response = super().complete(
            request,
            idempotency_key=idempotency_key,
            timeout_seconds=timeout_seconds,
        )
        return StructuredCallResponse(
            request_hash=response.request_hash,
            output=response.output,
            usage=response.usage,
            model_id="off\u200bicial-secret-model",
            provider_request_id="provider-safe-id",
        )


def test_provider_metadata_and_transport_error_secrets_never_persist(
    tmp_path: Path,
) -> None:
    protocol = default_protocol("metadata-dlp")
    metadata = _UnsafeMetadataTransport(
        {LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]}
    )
    executor = _executor(tmp_path, metadata, name="metadata")
    with pytest.raises(StructuredCallFailed, match="response_dlp_rejected"):
        ProposerAgent(executor).call(_context(protocol.content_hash).to_dict())
    persisted = executor.ledger.path.read_text(encoding="utf-8")
    assert "secret-model" not in persisted
    assert "official" not in persisted.replace("\u200b", "")

    raw_secret = "API_KEY_XYZ teacher holdout score 99"
    failure = FakeTransport(
        {LLMRole.PROPOSER: [TransportError(raw_secret, retryable=False)]}
    )
    failed_executor = _executor(tmp_path, failure, name="error-secret")
    with pytest.raises(StructuredCallFailed, match="transport_terminal_failure"):
        ProposerAgent(failed_executor).call(_context(protocol.content_hash).to_dict())
    assert raw_secret not in failed_executor.ledger.path.read_text(encoding="utf-8")
    # A non-retryable provider failure without an explicit definitely-unsent
    # attestation is conservatively charged at its full reservation.
    assert (
        failed_executor.budget.snapshot.cost_microusd
        == LLMPolicy().max_cost_microusd_per_call
    )
    assert failed_executor.budget.snapshot.pending_reservations == 0


def test_over_reservation_usage_is_terminal_charged_and_cannot_be_cached(
    tmp_path: Path,
) -> None:
    protocol = default_protocol("overuse")
    limits = LLMBudgetLimits(
        max_calls=3,
        max_input_tokens=100_000,
        max_output_tokens=4096,
        max_total_tokens=110_000,
        max_cost_microusd=1_000_000,
    )
    fake = FakeTransport(
        {LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]},
        usage=Usage(input_tokens=20, output_tokens=4097, cost_microusd=100),
    )
    ledger = CallLedger(tmp_path / "overuse.jsonl")
    executor = StructuredCallExecutor(
        transport=fake,
        ledger=ledger,
        budget=LLMBudget(limits),
        policy=LLMPolicy(),
    )
    context = _context(protocol.content_hash).to_dict()
    with pytest.raises(
        StructuredCallFailed, match="provider_usage_exceeded_reservation"
    ):
        ProposerAgent(executor).call(context)
    assert fake.call_count == 1
    assert executor.budget.snapshot.output_tokens == 4097
    assert executor.budget.snapshot.pending_reservations == 0
    with pytest.raises(StructuredCallFailed, match="previously failed terminally"):
        ProposerAgent(executor).call(context)
    assert fake.call_count == 1
    pin = ledger.pin
    reopened = CallLedger(
        ledger.path,
        expected_head_hash=pin.head_hash,
        expected_file_hash=pin.file_hash,
    )
    with pytest.raises(LLMBudgetExceeded, match="recovered"):
        LLMBudget.from_ledger(limits, reopened)


def test_two_executors_share_one_atomic_persisted_budget(tmp_path: Path) -> None:
    limits = LLMBudgetLimits(
        max_calls=1,
        max_input_tokens=100_000,
        max_output_tokens=10_000,
        max_total_tokens=110_000,
        max_cost_microusd=1_000_000,
    )
    path = tmp_path / "shared-budget.jsonl"
    # Both handles are intentionally opened while the ledger is empty.  The
    # authoritative check still occurs under the per-dispatch file lock.
    ledgers = (CallLedger(path), CallLedger(path))
    transports = (
        FakeTransport({LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]}),
        FakeTransport({LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]}),
    )
    executors = tuple(
        StructuredCallExecutor(
            transport=transport,
            ledger=ledger,
            budget=LLMBudget(limits),
            policy=LLMPolicy(),
        )
        for transport, ledger in zip(transports, ledgers, strict=True)
    )
    contexts = (
        _context(default_protocol("shared-one").content_hash).to_dict(),
        SafeResearchContext(
            campaign_hash=hash_json({"campaign": "shared-two"}),
            protocol_hash=default_protocol("shared-two").content_hash,
            generation=0,
            requested_count=2,
            allowed_fields=("Returns", "Amount", "Volume", "Close", "VWAP"),
            allowed_operators=("Add", "Sub", "Mul", "Div", "Neg", "TsMean"),
            allowed_windows=(3, 5, 10, 20),
        ).to_dict(),
    )

    def invoke(index: int):
        return ProposerAgent(executors[index]).call(contexts[index])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke, index) for index in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(("ok", future.result()))
            except Exception as exc:  # checked below
                outcomes.append((type(exc).__name__, str(exc)))
    assert [item[0] for item in outcomes].count("ok") == 1
    assert [item[0] for item in outcomes].count("LLMBudgetExceeded") == 1
    assert sum(item.call_count for item in transports) == 1
    assert ledgers[0].budget_snapshot(limits).calls == 1


def test_external_ledger_pin_detects_valid_full_rechain(tmp_path: Path) -> None:
    protocol = default_protocol("external-pin")
    fake = FakeTransport({LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]})
    executor = _executor(tmp_path, fake, name="rechain")
    ProposerAgent(executor).call(_context(protocol.content_hash).to_dict())
    pin = executor.ledger.pin
    path = executor.ledger.path
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    previous = "0" * 64
    rewritten = []
    for row in records:
        body = {key: value for key, value in row.items() if key != "record_hash"}
        body["created_ns"] += 1
        body["prev_hash"] = previous
        record = {**body, "record_hash": hash_json(body)}
        previous = record["record_hash"]
        rewritten.append(record)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rewritten))
    with pytest.raises(LedgerIntegrityError, match="prefix was rewritten"):
        _ = executor.ledger.records
    with pytest.raises(LedgerIntegrityError, match="external expected"):
        CallLedger(path)
    with pytest.raises(LedgerIntegrityError, match="external expected ledger"):
        CallLedger(
            path,
            expected_head_hash=pin.head_hash,
            expected_file_hash=pin.file_hash,
        )


def test_terminal_state_machine_rejects_post_success_mutation(tmp_path: Path) -> None:
    protocol = default_protocol("terminal-state")
    fake = FakeTransport({LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]})
    executor = _executor(tmp_path, fake, name="terminal-state")
    ProposerAgent(executor).call(_context(protocol.content_hash).to_dict())
    before = executor.ledger.path.read_bytes()
    with pytest.raises(LedgerIntegrityError, match="invalid terminal transition"):
        executor.ledger.terminal_failure(
            fake.requests[0],
            attempt=1,
            error_code="structured_output_invalid",
            usage=Usage(0, 0, 0),
        )
    assert executor.ledger.path.read_bytes() == before


def test_parent_catalog_binds_lineage_without_metrics(tmp_path: Path) -> None:
    parent_draft = CandidateDraft.from_dict(
        _candidate("parent_factor", "Returns", ["Returns"])
    )
    parent = ParentCatalogEntry.from_draft(
        parent_draft,
        lineage_depth=3,
    )
    child = _candidate("child_factor", "Neg(TsMean(Returns, 5))", ["Returns"])
    child["parent_ids"] = ["parent_factor"]
    scripts = {
        LLMRole.PROPOSER: [
            {"schema_version": "llm-proposer-output/v1", "proposals": [child]}
        ],
        LLMRole.CRITIC: [
            {
                "schema_version": "llm-critic-output/v1",
                "reviews": [
                    {
                        "candidate_id": "child_factor",
                        "verdict": "approve",
                        "reason_codes": ["mechanism_supported"],
                    }
                ],
            }
        ],
        LLMRole.RISK: [
            {
                "schema_version": "llm-risk-output/v1",
                "assessments": [
                    {
                        "candidate_id": "child_factor",
                        "decision": "allow",
                        "reason_codes": ["no_blocker"],
                    }
                ],
            }
        ],
        LLMRole.ARBITER: [
            {
                "schema_version": "llm-arbiter-output/v1",
                "decisions": [
                    {
                        "candidate_id": "child_factor",
                        "decision": "select",
                        "reason_codes": ["balanced_mechanism"],
                        "priority": 50,
                    }
                ],
            }
        ],
    }
    protocol = default_protocol("parent-catalog")
    fake = FakeTransport(scripts)
    provider = LLMPanelProvider(
        protocol=protocol,
        panel=AlphaResearchPanel(_executor(tmp_path, fake, name="parent")),
        parent_catalog=(parent,),
    )
    batch = provider.propose(
        ProposalContext(
            run_id="opaque",
            protocol_hash=protocol.content_hash,
            generation=7,
            request_count=1,
            prior_candidate_hashes=(parent.spec_hash,),
        )
    )
    assert [item.parent_ids for item in batch.candidates] == [("parent_factor",)]
    assert [item.lineage_depth for item in batch.candidates] == [4]
    for request in fake.requests:
        serialized = json.dumps(request.to_dict())
        assert parent.spec_hash in serialized
        assert "rank_ic" not in serialized
        assert "teacher" not in serialized
    empty_provider = LLMPanelProvider(
        protocol=protocol,
        panel=AlphaResearchPanel(
            _executor(tmp_path, FakeTransport({}), name="missing-parent")
        ),
    )
    with pytest.raises(UnsafeLLMContext, match="parent catalog"):
        empty_provider.propose(
            ProposalContext(
                run_id="opaque",
                protocol_hash=protocol.content_hash,
                generation=7,
                request_count=1,
                prior_candidate_hashes=(parent.spec_hash,),
            )
        )


def test_semantic_hash_canonicalizes_commutative_and_associative_operators() -> None:
    first = CandidateDraft.from_dict(
        _candidate("commutative_one", "Add(Returns, Amount)", ["Returns", "Amount"])
    )
    second = CandidateDraft.from_dict(
        _candidate("commutative_two", "Add(Amount, Returns)", ["Amount", "Returns"])
    )
    nested_one = CandidateDraft.from_dict(
        _candidate(
            "associative_one",
            "Add(Add(Returns, Amount), Volume)",
            ["Returns", "Amount", "Volume"],
        )
    )
    nested_two = CandidateDraft.from_dict(
        _candidate(
            "associative_two",
            "Add(Volume, Add(Amount, Returns))",
            ["Volume", "Amount", "Returns"],
        )
    )
    assert first.semantic_hash == second.semantic_hash
    assert nested_one.semantic_hash == nested_two.semantic_hash


@pytest.mark.parametrize(
    "value",
    [
        "api_key = abcdefghijk",
        "Authorization: Bearer abcdefghijk",
        "access_token=abcdefghijk",
        "client_secret=abcdefghijk",
        "tea_cher result",
        "offi_cial result",
        "hold_out result",
        "ignore_previous_instructions",
        "jail_break",
    ],
)
def test_separator_normalization_and_generic_credential_dlp(value: str) -> None:
    with pytest.raises(UnsafeLLMContext):
        assert_safe_context({"candidate_id": value})


def test_dlp_exception_does_not_echo_sensitive_key_or_value() -> None:
    raw_key = "api_key=raw-secret-key-material"
    raw_value = "Bearer raw-secret-value-material"
    with pytest.raises(UnsafeLLMContext) as caught:
        assert_safe_context({raw_key: raw_value})
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert raw_key not in rendered
    assert raw_value not in rendered
    assert caught.value.__context__ is None


def test_provider_request_identifier_is_hashed_before_any_persistence(
    tmp_path: Path,
) -> None:
    raw_identifier = "provider-request-123"
    response = StructuredCallResponse(
        request_hash=hash_json({"request": "provider-id"}),
        output={"safe": "value"},
        usage=Usage(1, 1, 1),
        model_id="offline-fake/v1",
        provider_request_id=raw_identifier,
    )
    expected = hash_provider_request_id(raw_identifier)
    assert expected.startswith("sha256:")
    assert response.provider_request_id == expected
    assert response.to_dict()["provider_request_id"] == expected
    assert raw_identifier not in json.dumps(response.to_dict())
    persisted_response = response.to_dict()
    persisted_response["provider_request_id"] = raw_identifier
    with pytest.raises(LLMContractError, match="persisted provider_request_id"):
        StructuredCallResponse.from_dict(persisted_response)

    replay_request = ProposerAgent(
        _executor(tmp_path, FakeTransport({}), name="provider-replay-request")
    ).build_request(
        _context(default_protocol("provider-replay-request").content_hash).to_dict()
    )
    replay = ReplayEntry(
        request=replay_request,
        output={"schema_version": "llm-proposer-output/v1", "proposals": []},
        usage=Usage(1, 1, 1),
        model_id="offline-fake/v1",
        provider_request_id=raw_identifier,
    )
    assert replay.provider_request_id == expected
    assert raw_identifier.encode("utf-8") not in canonical_json_bytes(replay.to_dict())
    persisted_replay = replay.to_dict()
    persisted_replay["provider_request_id"] = raw_identifier
    with pytest.raises(ValueError, match="must be hashed"):
        ReplayEntry.from_dict(persisted_replay, request=replay_request)


def test_model_allowlist_is_closed_and_hash_bound(tmp_path: Path) -> None:
    policy = LLMPolicy(allowed_model_ids=("offline-exact-replay/v1",))
    fake = FakeTransport({LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]})
    executor = _executor(tmp_path, fake, policy=policy, name="model-allowlist")
    with pytest.raises(StructuredCallFailed, match="response_dlp_rejected"):
        ProposerAgent(executor).call(
            _context(default_protocol("model-allowlist").content_hash).to_dict()
        )
    assert (
        executor.ledger.records[-1]["details"]["error_code"] == "response_dlp_rejected"
    )
    assert all(row["event"] != "call_succeeded" for row in executor.ledger.records)
    assert policy.content_hash != LLMPolicy().content_hash


def test_sensitive_provider_request_metadata_fails_closed_without_persistence(
    tmp_path: Path,
) -> None:
    raw_identifier = "Bearer provider-request-secret-material"

    class UnsafeProviderRequestTransport(FakeTransport):
        def complete(self, request, *, idempotency_key, timeout_seconds):
            response = super().complete(
                request,
                idempotency_key=idempotency_key,
                timeout_seconds=timeout_seconds,
            )
            return StructuredCallResponse(
                request_hash=response.request_hash,
                output=response.output,
                usage=response.usage,
                model_id="offline-fake/v1",
                provider_request_id=raw_identifier,
            )

    executor = _executor(
        tmp_path,
        UnsafeProviderRequestTransport(
            {LLMRole.PROPOSER: [_scripts()[LLMRole.PROPOSER][0]]}
        ),
        name="provider-request-dlp",
    )
    with pytest.raises(StructuredCallFailed, match="response_dlp_rejected") as caught:
        ProposerAgent(executor).call(
            _context(default_protocol("provider-request-dlp").content_hash).to_dict()
        )
    persisted = executor.ledger.path.read_text(encoding="utf-8")
    assert raw_identifier not in persisted
    assert "provider-request-secret-material" not in persisted
    assert caught.value.__context__ is None
    assert (
        executor.ledger.records[-1]["details"]["error_code"] == "response_dlp_rejected"
    )
    assert (
        executor.budget.snapshot.cost_microusd == LLMPolicy().max_cost_microusd_per_call
    )


def test_provider_errors_have_no_raw_exception_chain_or_traceback(
    tmp_path: Path,
) -> None:
    raw_secret = "Bearer raw-provider-secret-value"
    transport_error = TransportError(raw_secret, retryable=False)
    assert raw_secret not in str(transport_error)
    assert (
        transport_error.diagnostic_hash
        == hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()
    )
    executor = _executor(
        tmp_path,
        FakeTransport({LLMRole.PROPOSER: [transport_error]}),
        name="traceback-transport",
    )
    with pytest.raises(StructuredCallFailed) as caught:
        ProposerAgent(executor).call(
            _context(default_protocol("traceback-transport").content_hash).to_dict()
        )
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert raw_secret not in rendered
    assert "raw-provider-secret-value" not in rendered
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None

    untyped_raw = "api_key=untyped-provider-secret"
    untyped = _executor(
        tmp_path,
        FakeTransport({LLMRole.PROPOSER: [RuntimeError(untyped_raw)]}),
        name="traceback-untyped",
    )
    with pytest.raises(IndeterminateCallError) as untyped_caught:
        ProposerAgent(untyped).call(
            _context(default_protocol("traceback-untyped").content_hash).to_dict()
        )
    untyped_rendered = "".join(
        traceback.format_exception(
            untyped_caught.type,
            untyped_caught.value,
            untyped_caught.tb,
        )
    )
    assert untyped_raw not in untyped_rendered
    assert untyped_caught.value.__context__ is None


def test_candidate_descriptor_rebuild_rejects_tampered_authoritative_fields() -> None:
    protocol = default_protocol("descriptor-binding")
    draft = CandidateDraft.from_dict(
        _candidate("descriptor_factor", "Add(Returns, Amount)", ["Returns", "Amount"])
    )
    payload = _context(protocol.content_hash).to_dict(candidate_drafts=(draft,))
    validate_safe_context_payload(payload)

    mutations = (
        ("canonical_expression_ast", "Expression(body=Name(id='Returns', ctx=Load()))"),
        ("direction", -1),
        ("frequency", "minute"),
        ("mechanism_code", "volatility"),
        ("aggregation_window", "close30"),
        ("semantic_hash", "0" * 64),
        ("candidate_id", "offi_cial_factor"),
    )
    for field, replacement in mutations:
        tampered = json.loads(json.dumps(payload))
        tampered["candidate_descriptors"][0][field] = replacement
        with pytest.raises(UnsafeLLMContext):
            validate_safe_context_payload(tampered)


def test_parent_catalog_spec_hash_binds_every_safe_field() -> None:
    draft = CandidateDraft.from_dict(
        _candidate("bound_parent", "Neg(Returns)", ["Returns"])
    )
    parent = ParentCatalogEntry.from_draft(draft, lineage_depth=2)
    with pytest.raises(UnsafeLLMContext, match="does not bind draft"):
        ParentCatalogEntry.from_draft(
            draft,
            lineage_depth=2,
            spec_hash="0" * 64,
        )
    with pytest.raises(UnsafeLLMContext, match="does not bind catalog"):
        replace(parent, direction=-1)
    wire = parent.to_dict()
    wire["aggregation_window"] = "close30"
    with pytest.raises(UnsafeLLMContext):
        ParentCatalogEntry.from_dict(wire)


def test_panel_and_campaign_registry_share_one_semantic_identity() -> None:
    protocol = default_protocol("canonical-registry")
    first = CandidateDraft.from_dict(
        _candidate("panel_first", "Add(Returns, Amount)", ["Returns", "Amount"])
    )
    second = CandidateDraft.from_dict(
        _candidate("panel_second", "Add(Amount, Returns)", ["Amount", "Returns"])
    )

    def to_spec(draft: CandidateDraft) -> CandidateSpec:
        return CandidateSpec(
            candidate_id=draft.candidate_id,
            hypothesis=draft.hypothesis,
            expression=draft.expression,
            direction=draft.direction,
            frequency=draft.frequency,
            generation=0,
            family=draft.family,
            required_fields=draft.required_fields,
            protocol_hash=protocol.content_hash,
            provider="offline-shadow-test",
        )

    first_campaign_hash = semantic_signal_hash(to_spec(first))
    second_campaign_hash = semantic_signal_hash(to_spec(second))
    assert first.semantic_hash == first_campaign_hash
    assert second.semantic_hash == second_campaign_hash
    assert first_campaign_hash == second_campaign_hash


def test_terminal_failure_without_usage_keeps_conservative_reservation(
    tmp_path: Path,
) -> None:
    protocol = default_protocol("terminal-usage-required")
    ledger = CallLedger(tmp_path / "terminal-usage-required.jsonl")
    budget = _budget()
    executor = StructuredCallExecutor(
        transport=NetworkDeniedTransport(),
        ledger=ledger,
        budget=budget,
        policy=LLMPolicy(),
    )
    request = ProposerAgent(executor).build_request(
        _context(protocol.content_hash).to_dict()
    )
    reservation = {"input_tokens": 100, "output_tokens": 100, "cost_microusd": 100}
    ledger.reserve_and_start(
        request,
        attempt=1,
        reservation=reservation,
        limits=budget.limits,
    )
    with pytest.raises(LedgerIntegrityError, match="explicit usage"):
        ledger.terminal_failure(
            request,
            attempt=1,
            error_code="transport_terminal_failure",
            usage=None,
        )
    snapshot = ledger.budget_snapshot(budget.limits)
    assert snapshot.pending_reservations == 1
    assert snapshot.cost_microusd == reservation["cost_microusd"]


def test_live_hex_provider_identifier_is_domain_hashed_not_trusted() -> None:
    raw = "a" * 64
    expected = hash_provider_request_id(raw)
    response = StructuredCallResponse(
        request_hash=hash_json({"request": "hex-provider-id"}),
        output={},
        usage=Usage(1, 1, 1),
        model_id="offline-fake/v1",
        provider_request_id=raw,
    )
    assert response.provider_request_id == expected
    assert response.provider_request_id.startswith("sha256:")
    assert response.provider_request_id != raw
    # A persisted plain string is not a trusted in-memory digest marker.
    assert hash_provider_request_id(str(response.provider_request_id)) != expected
    assert (
        StructuredCallResponse.from_dict(response.to_dict()).provider_request_id
        == expected
    )


@pytest.mark.parametrize(
    "value",
    [
        "IC=0.041",
        "neutral IC is 0.032",
        "IC值为0.025",
        "R A N K-I_C : -.04",
        "中性 I/C ≈ 3.1%",
        "该因子的IC为0.04",
        "信息系数为0.03",
        "秩相关系数=0.02",
    ],
)
def test_exact_ic_language_and_separator_variants_are_rejected(value: str) -> None:
    with pytest.raises(UnsafeLLMContext, match="exact local metric"):
        assert_safe_context({"hypothesis": value})


class _MalformedWireResponse:
    def to_dict(self):
        raise RuntimeError("Bearer malformed-response-secret")


class _MalformedWireTransport(StructuredTransport):
    name = "malformed-wire-offline"

    def complete(self, request, *, idempotency_key: str, timeout_seconds: float):
        return _MalformedWireResponse()


def test_malformed_transport_return_is_sanitized_terminal_and_conservative(
    tmp_path: Path,
) -> None:
    executor = _executor(
        tmp_path,
        _MalformedWireTransport(),
        name="malformed-wire",
    )
    with pytest.raises(
        StructuredCallFailed, match="malformed_transport_response"
    ) as caught:
        ProposerAgent(executor).call(
            _context(default_protocol("malformed-wire").content_hash).to_dict()
        )
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert "malformed-response-secret" not in rendered
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    rows = executor.ledger.records
    assert [row["event"] for row in rows] == [
        "attempt_started",
        "call_terminal_failure",
    ]
    assert rows[-1]["details"]["error_code"] == "malformed_transport_response"
    assert rows[-1]["details"]["response_hash"] is None
    assert rows[-1]["details"]["charge"] == {
        "input_tokens": LLMPolicy().max_input_tokens_per_call,
        "output_tokens": LLMPolicy().max_output_tokens,
        "cost_microusd": LLMPolicy().max_cost_microusd_per_call,
    }
    assert executor.budget.snapshot.pending_reservations == 0


def test_review_coverage_failure_is_terminal_before_success_or_replay_cache(
    tmp_path: Path,
) -> None:
    draft = CandidateDraft.from_dict(
        _candidate("expected_candidate", "Neg(Returns)", ["Returns"])
    )
    role_context = SafeResearchContext(
        campaign_hash=hash_json({"campaign": "coverage-binding"}),
        protocol_hash=default_protocol("coverage-binding").content_hash,
        generation=0,
        requested_count=1,
        allowed_fields=("Returns",),
        allowed_operators=("Neg",),
        allowed_windows=(5,),
    ).to_dict(candidate_drafts=(draft,))
    wrong = {
        "schema_version": "llm-critic-output/v1",
        "reviews": [
            {
                "candidate_id": "wrong_candidate",
                "verdict": "approve",
                "reason_codes": ["mechanism_supported"],
            }
        ],
    }
    executor = _executor(
        tmp_path,
        FakeTransport({LLMRole.CRITIC: [wrong]}),
        name="coverage-binding",
    )
    with pytest.raises(StructuredCallFailed, match="structured_output_invalid"):
        CriticAgent(executor).call(role_context)
    assert executor.ledger.records[-1]["event"] == "call_terminal_failure"
    assert (
        executor.ledger.records[-1]["details"]["error_code"]
        == "structured_output_invalid"
    )
    assert all(row["event"] != "call_succeeded" for row in executor.ledger.records)


def _critic_replay_binding_case(tmp_path: Path, name: str):
    draft = CandidateDraft.from_dict(
        _candidate("expected_candidate", "Neg(Returns)", ["Returns"])
    )
    context = SafeResearchContext(
        campaign_hash=hash_json({"campaign": f"replay-binding-{name}"}),
        protocol_hash=default_protocol(f"replay-binding-{name}").content_hash,
        generation=0,
        requested_count=1,
        allowed_fields=("Returns",),
        allowed_operators=("Neg",),
        allowed_windows=(5,),
    ).to_dict(candidate_drafts=(draft,))
    request = CriticAgent(
        _executor(tmp_path, FakeTransport({}), name=f"request-{name}")
    ).build_request(context)
    valid = {
        "schema_version": "llm-critic-output/v1",
        "reviews": [
            {
                "candidate_id": "expected_candidate",
                "verdict": "approve",
                "reason_codes": ["mechanism_supported"],
            }
        ],
    }
    wrong = {
        "schema_version": "llm-critic-output/v1",
        "reviews": [
            {
                "candidate_id": "wrong_candidate",
                "verdict": "approve",
                "reason_codes": ["mechanism_supported"],
            }
        ],
    }
    response = StructuredCallResponse(
        request_hash=request.request_hash,
        output=valid,
        usage=Usage(20, 10, 100),
        model_id="offline-exact-replay/v1",
        provider_request_id=f"replay-binding-{name}",
    )
    return context, request, response, wrong


def test_wrong_reviewer_id_fails_replay_entry_construction(tmp_path: Path) -> None:
    _context_value, request, response, wrong = _critic_replay_binding_case(
        tmp_path, "construct"
    )
    wrong_response = StructuredCallResponse(
        request_hash=request.request_hash,
        output=wrong,
        usage=response.usage,
        model_id=response.model_id,
        provider_request_id="wrong-reviewer-construct",
    )
    with pytest.raises(StructuredOutputError, match="coverage mismatch"):
        ReplayEntry.from_response(wrong_response, request=request)


def test_wrong_reviewer_id_fails_before_replay_cassette_write(tmp_path: Path) -> None:
    _context_value, request, response, wrong = _critic_replay_binding_case(
        tmp_path, "write"
    )
    entry = ReplayEntry.from_response(response, request=request)
    object.__setattr__(entry, "output", wrong)
    destination = tmp_path / "wrong-reviewer-write.jsonl"
    with pytest.raises(StructuredOutputError, match="coverage mismatch"):
        write_replay_cassette(
            [entry],
            destination,
            request_by_hash={request.request_hash: request},
        )
    assert not destination.exists()


def test_wrong_reviewer_id_fails_replay_cassette_load_even_when_rehashed(
    tmp_path: Path,
) -> None:
    _context_value, request, response, wrong = _critic_replay_binding_case(
        tmp_path, "load"
    )
    entry = ReplayEntry.from_response(response, request=request)
    request_by_hash = {request.request_hash: request}
    path = write_replay_cassette(
        [entry],
        tmp_path / "wrong-reviewer-load.jsonl",
        request_by_hash=request_by_hash,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["output"] = wrong
    payload["entry_hash"] = hash_json(
        {key: value for key, value in payload.items() if key != "entry_hash"}
    )
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ValueError, match="invalid replay cassette line"):
        load_replay_cassette(
            path,
            request_by_hash=request_by_hash,
            expected_file_hash=hash_file(path),
        )


def test_wrong_reviewer_id_fails_exact_replay_complete_without_success_cache(
    tmp_path: Path,
) -> None:
    context, request, response, wrong = _critic_replay_binding_case(
        tmp_path, "complete"
    )
    entry = ReplayEntry.from_response(response, request=request)
    replay = ExactReplayTransport(
        [entry],
        request_by_hash={request.request_hash: request},
    )
    # Simulate post-construction in-memory corruption: complete must not trust
    # the validation performed by the constructor.
    object.__setattr__(entry, "output", wrong)
    executor = _executor(tmp_path, replay, name="wrong-reviewer-complete")
    with pytest.raises(StructuredCallFailed, match="transport_terminal_failure"):
        CriticAgent(executor).call(context)
    assert replay.call_count == 1
    assert all(row["event"] != "call_succeeded" for row in executor.ledger.records)


def test_replay_request_registry_rejects_missing_and_extra_requests(
    tmp_path: Path,
) -> None:
    _context_value, request, response, _wrong = _critic_replay_binding_case(
        tmp_path, "registry"
    )
    entry = ReplayEntry.from_response(response, request=request)
    with pytest.raises(ValueError, match="no frozen request"):
        write_replay_cassette(
            [entry],
            tmp_path / "missing-request.jsonl",
            request_by_hash={},
        )

    extra_context, extra_request, _extra_response, _ = _critic_replay_binding_case(
        tmp_path, "registry-extra"
    )
    assert extra_context != _context_value
    with pytest.raises(ValueError, match="must match exactly"):
        write_replay_cassette(
            [entry],
            tmp_path / "extra-request.jsonl",
            request_by_hash={
                request.request_hash: request,
                extra_request.request_hash: extra_request,
            },
        )

    valid_path = write_replay_cassette(
        [entry],
        tmp_path / "registry-load.jsonl",
        request_by_hash={request.request_hash: request},
    )
    valid_hash = hash_file(valid_path)
    with pytest.raises(ValueError, match="invalid replay cassette line"):
        load_replay_cassette(
            valid_path,
            request_by_hash={},
            expected_file_hash=valid_hash,
        )
    with pytest.raises(ValueError, match="must match exactly"):
        load_replay_cassette(
            valid_path,
            request_by_hash={
                request.request_hash: request,
                extra_request.request_hash: extra_request,
            },
            expected_file_hash=valid_hash,
        )


def test_ema30_contract_binds_schema_descriptor_parent_and_provider(
    tmp_path: Path,
) -> None:
    protocol = default_protocol("ema30-e2e")

    def minute_candidate(candidate_id: str, span: int) -> dict[str, object]:
        item = _candidate(candidate_id, "Neg(TsMean(Returns, 5))", ["Returns"])
        item["frequency"] = "minute"
        item["aggregation"] = {
            "method": "mean",
            "window": "close30",
            "smoothing_span": span,
        }
        return item

    ema20 = CandidateDraft.from_dict(minute_candidate("ema20_factor", 20))
    ema30 = CandidateDraft.from_dict(minute_candidate("ema30_factor", 30))
    assert ema20.semantic_hash != ema30.semantic_hash
    parent30 = ParentCatalogEntry.from_draft(ema30, lineage_depth=0)
    assert parent30.smoothing_span == 30
    descriptor_context = SafeResearchContext(
        campaign_hash=hash_json({"campaign": "ema30-descriptor"}),
        protocol_hash=protocol.content_hash,
        generation=0,
        requested_count=1,
        allowed_fields=("Returns",),
        allowed_operators=("Neg", "TsMean"),
        allowed_windows=(5,),
    ).to_dict(candidate_drafts=(ema30,))
    assert descriptor_context["candidate_descriptors"][0]["smoothing_span"] == 30

    scripts = {
        LLMRole.PROPOSER: [
            {"schema_version": "llm-proposer-output/v1", "proposals": [ema30.to_dict()]}
        ],
        LLMRole.CRITIC: [
            {
                "schema_version": "llm-critic-output/v1",
                "reviews": [
                    {
                        "candidate_id": ema30.candidate_id,
                        "verdict": "approve",
                        "reason_codes": ["mechanism_supported"],
                    }
                ],
            }
        ],
        LLMRole.RISK: [
            {
                "schema_version": "llm-risk-output/v1",
                "assessments": [
                    {
                        "candidate_id": ema30.candidate_id,
                        "decision": "allow",
                        "reason_codes": ["no_blocker"],
                    }
                ],
            }
        ],
        LLMRole.ARBITER: [
            {
                "schema_version": "llm-arbiter-output/v1",
                "decisions": [
                    {
                        "candidate_id": ema30.candidate_id,
                        "decision": "select",
                        "reason_codes": ["balanced_mechanism"],
                        "priority": 50,
                    }
                ],
            }
        ],
    }
    provider = LLMPanelProvider(
        protocol=protocol,
        panel=AlphaResearchPanel(
            _executor(tmp_path, FakeTransport(scripts), name="ema30")
        ),
    )
    batch = provider.propose(
        ProposalContext(
            run_id="ema30-run",
            protocol_hash=protocol.content_hash,
            generation=0,
            request_count=1,
        )
    )
    assert batch.candidates[0].aggregation["smoothing_span"] == 30


def test_ledger_pin_head_count_and_bytes_are_one_concurrent_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import llm_alpha_mining.mining.llm.ledger as ledger_module

    path = tmp_path / "pin-race.jsonl"
    first = CallLedger(path)
    second = CallLedger(path)
    protocol = default_protocol("pin-race-regression")
    request_a = ProposerAgent(
        _executor(tmp_path, FakeTransport({}), name="pin-a-request")
    ).build_request(_context(protocol.content_hash).to_dict())
    other_context = SafeResearchContext(
        campaign_hash=hash_json({"campaign": "pin-race-other"}),
        protocol_hash=protocol.content_hash,
        generation=0,
        requested_count=1,
        allowed_fields=("Returns",),
        allowed_operators=("Neg",),
        allowed_windows=(5,),
    ).to_dict()
    request_b = ProposerAgent(
        _executor(tmp_path, FakeTransport({}), name="pin-b-request")
    ).build_request(other_context)
    reservation = {"input_tokens": 10, "output_tokens": 10, "cost_microusd": 10}
    first.reserve_and_start(
        request_a,
        attempt=1,
        reservation=reservation,
        limits=_budget().limits,
    )

    entered_probe = threading.Event()
    append_done = threading.Event()
    original_file_hash = ledger_module._file_hash

    def racing_file_hash(target):
        entered_probe.set()
        assert append_done.wait(5)
        return original_file_hash(target)

    monkeypatch.setattr(ledger_module, "_file_hash", racing_file_hash)

    def append_second() -> None:
        assert entered_probe.wait(5)
        second.reserve_and_start(
            request_b,
            attempt=1,
            reservation=reservation,
            limits=_budget().limits,
        )
        append_done.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(append_second)
        pin = first.pin
        future.result()
    CallLedger(
        path,
        expected_head_hash=pin.head_hash,
        expected_file_hash=pin.file_hash,
    )
