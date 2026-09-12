from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Mapping, cast

import pytest

from llm_alpha_mining.research.agents import (
    HTTPResponse,
    LiveProviderConfig,
    LiveStructuredTransport,
)
from llm_alpha_mining.mining.llm.agents import StructuredAgent, StructuredCallExecutor
from llm_alpha_mining.mining.llm.budget import LLMBudget, LLMBudgetLimits
from llm_alpha_mining.mining.llm.domain import LLMRole
from llm_alpha_mining.mining.llm.ledger import CallLedger
from llm_alpha_mining.mining.llm.policy import LLMPolicy
from llm_alpha_mining.mining.llm.safe_context import SafeResearchContext
from llm_alpha_mining.mining.llm.transport.base import TransportError


def _candidate_output() -> dict[str, object]:
    return {
        "schema_version": "llm-proposer-output/v1",
        "proposals": [
            {
                "candidate_id": "amount_reversal",
                "hypothesis": "A transient liquidity mechanism can predict reversal.",
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
        ],
    }


def _provider_body(*, status_model: str = "gpt-test") -> bytes:
    return json.dumps(
        {
            "id": "provider-request-1",
            "model": status_model,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(_candidate_output())},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
    ).encode()


@dataclass
class FakeHTTPClient:
    responses: list[HTTPResponse]
    calls: list[dict[str, object]] = field(default_factory=list)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> HTTPResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout_seconds": timeout_seconds,
                "maximum_response_bytes": maximum_response_bytes,
            }
        )
        return self.responses.pop(0)


def _policy() -> LLMPolicy:
    return LLMPolicy(
        allowed_model_ids=("live-test/v1",),
        timeout_seconds=2.0,
        max_retries=0,
    )


def _config() -> LiveProviderConfig:
    return LiveProviderConfig(
        base_url="https://api.example.test/v1",
        allowed_hosts=("api.example.test",),
        api_key_env="RESEARCH_LIVE_API_KEY",
        provider_model_id="gpt-test",
        accepted_provider_model_ids=("gpt-test",),
        wire_model_id="live-test/v1",
        input_cost_microusd_per_million=100_000,
        output_cost_microusd_per_million=200_000,
    )


def _build_request(policy: LLMPolicy):
    context = SafeResearchContext(
        campaign_hash="a" * 64,
        protocol_hash="b" * 64,
        generation=0,
        requested_count=1,
        allowed_fields=("Returns", "Close"),
        allowed_operators=("Neg", "TsMean"),
        allowed_windows=(5, 10),
    )
    executor_stub = cast(StructuredCallExecutor, SimpleNamespace(policy=policy))
    return StructuredAgent(LLMRole.PROPOSER, executor_stub).build_request(
        context.to_dict()
    )


def test_live_transport_runs_through_v5_ledger_and_strict_schema(
    tmp_path, monkeypatch
) -> None:
    policy = _policy()
    client = FakeHTTPClient([HTTPResponse(200, {}, _provider_body())])
    transport = LiveStructuredTransport(config=_config(), policy=policy, client=client)
    monkeypatch.setenv("RESEARCH_LIVE_API_KEY", "test-runtime-credential")
    executor = StructuredCallExecutor(
        transport=transport,
        ledger=CallLedger(tmp_path / "live-ledger.jsonl"),
        budget=LLMBudget(
            LLMBudgetLimits(
                max_calls=2,
                max_input_tokens=100_000,
                max_output_tokens=10_000,
                max_total_tokens=110_000,
                max_cost_microusd=1_000_000,
            )
        ),
        policy=policy,
    )
    request = _build_request(policy)
    response = executor.execute(request)
    assert response.request_hash == request.request_hash
    assert response.model_id == "live-test/v1"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 20
    assert response.usage.cost_microusd == 5
    assert response.provider_request_id.startswith("sha256:")
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["url"] == "https://api.example.test/v1/chat/completions"
    assert call["headers"]["Idempotency-Key"] == request.operation_id
    assert b"test-runtime-credential" not in call["body"]
    executor.ledger.verify_against_pin(executor.ledger.pin)


def test_missing_credential_fails_before_any_network_dispatch(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_LIVE_API_KEY", raising=False)
    policy = _policy()
    client = FakeHTTPClient([])
    transport = LiveStructuredTransport(config=_config(), policy=policy, client=client)
    request = _build_request(policy)
    with pytest.raises(TransportError) as captured:
        transport.complete(
            request,
            idempotency_key=request.operation_id,
            timeout_seconds=1.0,
        )
    assert captured.value.definitely_unsent is True
    assert client.calls == []


@pytest.mark.parametrize(
    ("status", "retryable", "dispatch_unknown"),
    [(429, True, False), (500, False, True)],
)
def test_http_failures_preserve_safe_dispatch_state(
    monkeypatch, status: int, retryable: bool, dispatch_unknown: bool
) -> None:
    monkeypatch.setenv("RESEARCH_LIVE_API_KEY", "test-runtime-credential")
    policy = _policy()
    client = FakeHTTPClient([HTTPResponse(status, {}, b"provider detail is ignored")])
    transport = LiveStructuredTransport(config=_config(), policy=policy, client=client)
    request = _build_request(policy)
    with pytest.raises(TransportError) as captured:
        transport.complete(
            request,
            idempotency_key=request.operation_id,
            timeout_seconds=1.0,
        )
    assert captured.value.retryable is retryable
    assert captured.value.dispatch_unknown is dispatch_unknown


def test_live_config_blocks_http_and_non_allowlisted_hosts() -> None:
    with pytest.raises(ValueError, match="HTTPS host allowlist"):
        LiveProviderConfig(
            base_url="http://api.example.test/v1",
            allowed_hosts=("api.example.test",),
            api_key_env="RESEARCH_LIVE_API_KEY",
            provider_model_id="gpt-test",
            accepted_provider_model_ids=("gpt-test",),
            wire_model_id="live-test/v1",
            input_cost_microusd_per_million=1,
            output_cost_microusd_per_million=1,
        )
    with pytest.raises(ValueError, match="HTTPS host allowlist"):
        LiveProviderConfig(
            base_url="https://metadata.internal/v1",
            allowed_hosts=("api.example.test",),
            api_key_env="RESEARCH_LIVE_API_KEY",
            provider_model_id="gpt-test",
            accepted_provider_model_ids=("gpt-test",),
            wire_model_id="live-test/v1",
            input_cost_microusd_per_million=1,
            output_cost_microusd_per_million=1,
        )
