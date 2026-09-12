from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Iterable, Mapping

from llm_alpha_mining.mining.llm.domain import (
    LLMRole,
    StructuredCallRequest,
    StructuredCallResponse,
    Usage,
)
from llm_alpha_mining.mining.llm.transport.base import StructuredTransport


class FakeTransport(StructuredTransport):
    """Deterministic test double with no file, SDK, socket or subprocess access."""

    name = "fake-offline"

    def __init__(
        self,
        scripts: Mapping[LLMRole | str, Iterable[Mapping[str, Any] | BaseException]],
        *,
        usage: Usage | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self._scripts = {LLMRole(role): deque(items) for role, items in scripts.items()}
        self.usage = usage or Usage(
            input_tokens=20, output_tokens=10, cost_microusd=100
        )
        self.delay_seconds = float(delay_seconds)
        self.requests: list[StructuredCallRequest] = []
        self.idempotency_keys: list[str] = []
        self.call_count_by_role: dict[LLMRole, int] = defaultdict(int)
        self.recorded_responses: list[StructuredCallResponse] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def complete(
        self,
        request: StructuredCallRequest,
        *,
        idempotency_key: str,
        timeout_seconds: float,
    ) -> StructuredCallResponse:
        self.requests.append(request)
        self.idempotency_keys.append(idempotency_key)
        self.call_count_by_role[request.role] += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        queue = self._scripts.get(request.role)
        if not queue:
            raise AssertionError(f"no fake response scripted for {request.role.value}")
        item = queue.popleft()
        if isinstance(item, BaseException):
            raise item
        response = StructuredCallResponse(
            request_hash=request.request_hash,
            output=item,
            usage=self.usage,
            model_id="offline-fake/v1",
            provider_request_id=f"fake-{self.call_count}",
        )
        self.recorded_responses.append(response)
        return response
