from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from llm_alpha_mining.mining.llm.domain import (
    StructuredCallRequest,
    StructuredCallResponse,
)


class TransportError(RuntimeError):
    """Typed failure; only definitely-unsent failures may be retried."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        dispatch_unknown: bool = False,
        definitely_unsent: bool = False,
    ) -> None:
        if not isinstance(message, str):
            message = type(message).__name__
        # Provider/SDK exception messages are untrusted metadata and may embed
        # credentials, request bodies or internal endpoints.  Preserve only an
        # irreversible diagnostic correlation value; callers receive a closed
        # generic message and a closed error code from the executor.
        self.diagnostic_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()
        super().__init__("structured transport failure")
        self.retryable = bool(retryable)
        self.dispatch_unknown = bool(dispatch_unknown)
        # Only this explicit contract permits zero usage after entering the
        # dispatch path.  Retryable errors are defined as pre-dispatch.
        self.definitely_unsent = bool(definitely_unsent or retryable)
        if self.retryable and self.dispatch_unknown:
            raise ValueError("an unknown dispatch state must never be marked retryable")
        if self.dispatch_unknown and self.definitely_unsent:
            raise ValueError("unknown dispatch cannot also be definitely unsent")


class TransportTimeout(TransportError):
    def __init__(self, message: str = "structured transport timed out") -> None:
        super().__init__(message, retryable=False, dispatch_unknown=True)


class NetworkDenied(TransportError):
    def __init__(self, message: str = "network access is disabled") -> None:
        super().__init__(message, definitely_unsent=True)


class StructuredTransport(ABC):
    """Vendor-neutral injection point.  Implementations must honor idempotency."""

    name: str

    @abstractmethod
    def complete(
        self,
        request: StructuredCallRequest,
        *,
        idempotency_key: str,
        timeout_seconds: float,
    ) -> StructuredCallResponse:
        raise NotImplementedError


class NetworkDeniedTransport(StructuredTransport):
    """Executable proof that the golden path cannot silently use a network."""

    name = "network-denied"

    def __init__(self) -> None:
        self.call_count = 0

    def complete(
        self,
        request: StructuredCallRequest,
        *,
        idempotency_key: str,
        timeout_seconds: float,
    ) -> StructuredCallResponse:
        self.call_count += 1
        raise NetworkDenied(
            "network access is disabled for the offline LLM control plane"
        )
