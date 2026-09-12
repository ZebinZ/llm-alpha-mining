from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from llm_alpha_mining.mining.llm.domain import (
    LLMRole,
    StructuredCallRequest,
    StructuredCallResponse,
)
from llm_alpha_mining.mining.llm.safe_context import (
    UnsafeLLMContext,
    assert_safe_context,
    assert_safe_response_metadata,
)
from llm_alpha_mining.mining.llm.schemas import StructuredOutputError, parse_role_output


class ResponseRequestBindingError(ValueError):
    """A response does not belong to the exact structured request supplied."""


def request_descriptor_ids(request: StructuredCallRequest) -> tuple[str, ...]:
    """Return the canonical reviewer descriptor-ID set committed by a request.

    Proposer calls do not review an existing descriptor set and therefore bind
    the empty tuple.  Review calls must carry a complete, unique descriptor
    collection.  Sorting makes the persisted binding explicitly set-like while
    the request hash still commits the original descriptor order and payload.
    """

    if type(request) is not StructuredCallRequest:
        raise ResponseRequestBindingError("request must be an exact wire object")
    if request.role is LLMRole.PROPOSER:
        return ()
    descriptors = request.context.get("candidate_descriptors")
    if not isinstance(descriptors, (list, tuple)):
        raise StructuredOutputError("review request has no descriptor set")
    identifiers: list[str] = []
    for item in descriptors:
        if not isinstance(item, Mapping):
            raise StructuredOutputError("review descriptor set is malformed")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise StructuredOutputError("review descriptor set is malformed")
        identifiers.append(candidate_id)
    if len(identifiers) != len(set(identifiers)):
        raise StructuredOutputError("review descriptor IDs must be unique")
    return tuple(sorted(identifiers))


def _validate_role_binding(
    request: StructuredCallRequest,
    parsed: Any,
) -> tuple[str, ...]:
    expected = request_descriptor_ids(request)
    if request.role is LLMRole.PROPOSER:
        requested_count = request.context.get("requested_count")
        if (
            not isinstance(requested_count, int)
            or isinstance(requested_count, bool)
            or requested_count < 1
            or len(parsed) > requested_count
        ):
            raise StructuredOutputError("proposer exceeded requested_count")
        return expected
    actual = tuple(sorted(item.candidate_id for item in parsed))
    if len(actual) != len(set(actual)) or actual != expected:
        raise StructuredOutputError("review candidate coverage mismatch")
    return expected


def validate_response_against_request(
    request: StructuredCallRequest,
    response: StructuredCallResponse,
    *,
    allowed_model_ids: Iterable[str] | None = None,
) -> Any:
    """Authoritatively validate one response against one exact request.

    This is the single schema/DLP/role/request-binding authority used by both
    the executor and exact replay.  It deliberately has no dependency on the
    executor, ledger, or transport packages, preventing validation cycles.
    The parsed role-domain value is returned only after every check passes.
    """

    if type(request) is not StructuredCallRequest:
        raise ResponseRequestBindingError("request must be an exact wire object")
    if type(response) is not StructuredCallResponse:
        raise ResponseRequestBindingError("response must be an exact wire object")
    if response.request_hash != request.request_hash:
        raise ResponseRequestBindingError("response request hash mismatch")
    if allowed_model_ids is not None and response.model_id not in frozenset(
        allowed_model_ids
    ):
        raise UnsafeLLMContext("response model is outside the closed allow-list")
    assert_safe_response_metadata(
        model_id=response.model_id,
        provider_request_id=response.provider_request_id,
    )
    # DLP executes before domain materialization.  The role parser then applies
    # the closed schema, CoT rejection, enums, duplicate checks and invariants.
    assert_safe_context(response.output)
    parsed = parse_role_output(request.role, response.output)
    _validate_role_binding(request, parsed)
    return parsed


__all__ = [
    "ResponseRequestBindingError",
    "request_descriptor_ids",
    "validate_response_against_request",
]
