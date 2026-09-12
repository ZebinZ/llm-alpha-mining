from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from threading import RLock
from typing import Any, Mapping

from factor_production.v5.artifacts.hashing import canonical_json_bytes, hash_json
from factor_production.v5.llm.budget import LLMBudget
from factor_production.v5.llm.domain import (
    LLMContractError,
    LLMRole,
    StructuredCallRequest,
    StructuredCallResponse,
    Usage,
)
from factor_production.v5.llm.ledger import CallLedger, IndeterminateCallError
from factor_production.v5.llm.policy import LLMPolicy
from factor_production.v5.llm.safe_context import (
    UnsafeLLMContext,
    assert_safe_context,
    validate_safe_context_payload,
)
from factor_production.v5.llm.schemas import (
    ROLE_SCHEMAS,
    StructuredOutputError,
    parse_role_output,
    schema_hash,
)
from factor_production.v5.llm.transport.base import (
    StructuredTransport,
    TransportError,
    TransportTimeout,
)
from factor_production.v5.llm.validation import (
    ResponseRequestBindingError,
    validate_response_against_request,
)


class CircuitOpen(RuntimeError):
    pass


class StructuredCallFailed(RuntimeError):
    pass


@dataclass(slots=True)
class CircuitBreaker:
    threshold: int
    failures: int = 0
    is_open: bool = False

    def success(self) -> None:
        self.failures = 0

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.is_open = True


class StructuredCallExecutor:
    """One audited dispatch path shared by every role."""

    def __init__(
        self,
        *,
        transport: StructuredTransport,
        ledger: CallLedger,
        budget: LLMBudget,
        policy: LLMPolicy,
    ) -> None:
        self.transport = transport
        self.ledger = ledger
        self.budget = budget
        self.policy = policy
        self.circuit = CircuitBreaker(policy.circuit_failure_threshold)
        self._lock = RLock()
        # Budget authorization and settlement are ledger transactions.  This
        # binding makes every process read the same persisted counters.
        self.budget.bind(self.ledger)

    @staticmethod
    def _response_hash(response: StructuredCallResponse) -> str:
        # Only this digest may survive a rejected response.  Raw output and
        # provider-controlled metadata are never copied into a failure record.
        return hash_json(response.to_dict())

    @staticmethod
    def _usage_fits(request: StructuredCallRequest, response: StructuredCallResponse, reserved_input: int) -> bool:
        return (
            response.usage.input_tokens <= reserved_input
            and response.usage.output_tokens <= request.max_output_tokens
            and response.usage.cost_microusd <= request.max_cost_microusd
        )

    def _validate_response(self, request: StructuredCallRequest, response: StructuredCallResponse) -> None:
        try:
            validate_response_against_request(
                request,
                response,
                allowed_model_ids=self.policy.allowed_model_ids,
            )
        except ResponseRequestBindingError:
            raise StructuredCallFailed("response_request_hash_mismatch") from None

    def _invoke(self, request: StructuredCallRequest) -> StructuredCallResponse:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(
            self.transport.complete,
            request,
            idempotency_key=request.operation_id,
            timeout_seconds=float(self.policy.timeout_seconds),
        )
        timed_out = False
        try:
            response = future.result(timeout=float(self.policy.timeout_seconds))
        except concurrent.futures.TimeoutError:
            future.cancel()
            timed_out = True
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if timed_out:
            # Raised after leaving the provider exception handler so the raw
            # exception is absent from both __context__ and rendered traceback.
            raise TransportTimeout() from None
        return response

    def execute(self, request: StructuredCallRequest) -> StructuredCallResponse:
        with self._lock:
            if self.circuit.is_open:
                raise CircuitOpen("structured LLM circuit is open")
            existing = self.ledger.lookup(request.operation_id)
            if existing is not None:
                if existing.request_hash != request.request_hash:
                    raise StructuredCallFailed("operation hash collision")
                if existing.state == "succeeded":
                    response = existing.response
                    assert response is not None
                    self._validate_response(request, response)
                    return response
                if existing.state == "indeterminate":
                    raise IndeterminateCallError(
                        "prior dispatch may have reached the provider; duplicate call denied"
                    )
                if existing.state == "terminal_failure":
                    raise StructuredCallFailed("operation previously failed terminally")
                attempt = existing.next_attempt
            else:
                attempt = 1

            maximum_attempt = self.policy.max_retries + 1
            while attempt <= maximum_attempt:
                estimated_input = max(
                    1,
                    math.ceil(
                        (
                            len(canonical_json_bytes(request.to_dict()))
                            + len(self.policy.prompts[request.role].encode("utf-8"))
                            + len(canonical_json_bytes(ROLE_SCHEMAS[request.role]))
                        )
                        / 4
                    ),
                )
                if estimated_input > self.policy.max_input_tokens_per_call:
                    raise StructuredCallFailed("request_exceeds_input_limit")
                reservation = {
                    "input_tokens": self.policy.max_input_tokens_per_call,
                    "output_tokens": request.max_output_tokens,
                    "cost_microusd": request.max_cost_microusd,
                }
                self.ledger.reserve_and_start(
                    request,
                    attempt=attempt,
                    reservation=reservation,
                    limits=self.budget.limits,
                )
                outward_failure: RuntimeError | None = None
                retry_dispatch = False
                try:
                    response = self._invoke(request)
                except TransportError as exc:
                    self.circuit.failure()
                    if exc.dispatch_unknown:
                        self.ledger.indeterminate(
                            request, attempt=attempt, error_code="transport_dispatch_indeterminate"
                        )
                        # The terminal event atomically charges the reservation.
                        outward_failure = IndeterminateCallError(
                            "transport_dispatch_indeterminate"
                        )
                    elif exc.retryable and attempt < maximum_attempt:
                        self.ledger.retryable_failure(
                            request, attempt=attempt, error_code="transport_retryable_failure"
                        )
                        attempt += 1
                        retry_dispatch = True
                    else:
                        self.ledger.terminal_failure(
                            request,
                            attempt=attempt,
                            error_code="transport_terminal_failure",
                            usage=(
                                Usage(0, 0, 0)
                                if exc.definitely_unsent
                                else Usage.from_dict(reservation)
                            ),
                        )
                        outward_failure = StructuredCallFailed(
                            "transport_terminal_failure"
                        )
                except LLMContractError:
                    self.circuit.failure()
                    # Response construction itself rejected provider metadata.
                    # No raw identifier exists and unknown usage is charged at
                    # the full reservation.
                    self.ledger.terminal_failure(
                        request,
                        attempt=attempt,
                        error_code="response_dlp_rejected",
                        usage=Usage.from_dict(reservation),
                    )
                    outward_failure = StructuredCallFailed(
                        "response_dlp_rejected"
                    )
                except Exception:
                    self.circuit.failure()
                    # An untyped exception after dispatch has unknown billing
                    # and delivery state.  Persist no exception text/secret.
                    self.ledger.indeterminate(
                        request, attempt=attempt, error_code="untyped_transport_failure"
                    )
                    outward_failure = IndeterminateCallError(
                        "untyped_transport_failure"
                    )
                if retry_dispatch:
                    continue
                if outward_failure is not None:
                    # Deliberately raised after the provider exception handler:
                    # no SDK message remains in __context__ or the traceback.
                    raise outward_failure from None

                response_hash: str | None = None
                validation_failure: str | None = None
                failure_usage: Usage | None = None
                try:
                    # A transport implementation is not allowed to smuggle an
                    # arbitrary object with an executable ``to_dict`` method
                    # into the persistence path.  Exact wire objects only.
                    if type(response) is not StructuredCallResponse:
                        raise LLMContractError("transport returned a non-wire response")
                    response_hash = self._response_hash(response)
                    if not self._usage_fits(
                        request,
                        response,
                        self.policy.max_input_tokens_per_call,
                    ):
                        validation_failure = "provider_usage_exceeded_reservation"
                        failure_usage = response.usage
                    else:
                        # At this point the immutable Usage object has passed
                        # its type/range checks and is bounded by the reserved
                        # maxima, so it is safe to use as the conservative
                        # charge for a later content-validation rejection.
                        failure_usage = response.usage
                        self._validate_response(request, response)
                except UnsafeLLMContext:
                    validation_failure = "response_dlp_rejected"
                except StructuredOutputError:
                    validation_failure = "structured_output_invalid"
                except StructuredCallFailed as exc:
                    safe_code = str(exc)
                    validation_failure = (
                        safe_code
                        if safe_code in {
                            "response_request_hash_mismatch",
                        }
                        else "malformed_transport_response"
                    )
                except Exception:
                    # This includes response serialization/hashing/property
                    # failures.  Raw provider exception text is intentionally
                    # discarded and never becomes ledger or traceback data.
                    validation_failure = "malformed_transport_response"

                if validation_failure is not None:
                    self.circuit.failure()
                    # Unless an over-reservation Usage object was completely
                    # validated, retain the entire reservation as the
                    # conservative terminal charge.
                    if failure_usage is None:
                        failure_usage = Usage.from_dict(reservation)
                    self.ledger.terminal_failure(
                        request,
                        attempt=attempt,
                        error_code=validation_failure,
                        usage=failure_usage,
                        response_hash=response_hash,
                    )
                    raise StructuredCallFailed(validation_failure) from None

                # The ledger repeats the validation as defense in depth.  Any
                # unexpected failure here is also closed safely and cannot
                # leave a lone active reservation.
                success_failure = False
                try:
                    self.ledger.success(
                        request,
                        attempt=attempt,
                        response=response,
                        allowed_model_ids=self.policy.allowed_model_ids,
                    )
                except Exception:
                    success_failure = True
                if success_failure:
                    self.circuit.failure()
                    self.ledger.terminal_failure(
                        request,
                        attempt=attempt,
                        error_code="malformed_transport_response",
                        usage=Usage.from_dict(reservation),
                        response_hash=response_hash,
                    )
                    raise StructuredCallFailed("malformed_transport_response") from None
                self.circuit.success()
                return response
            raise StructuredCallFailed("finite retry budget exhausted")


class StructuredAgent:
    def __init__(self, role: LLMRole | str, executor: StructuredCallExecutor) -> None:
        self.role = LLMRole(role)
        self.executor = executor

    def build_request(self, context: Mapping[str, Any]) -> StructuredCallRequest:
        assert_safe_context(context)
        validate_safe_context_payload(context)
        policy = self.executor.policy
        operation_seed = {
            "role": self.role.value,
            "context": context,
            "policy_hash": policy.content_hash,
        }
        operation_id = f"llm-{hash_json(operation_seed)[:40]}"
        return StructuredCallRequest(
            operation_id=operation_id,
            role=self.role,
            prompt_template_id=policy.prompt_template_id(self.role),
            prompt_hash=policy.prompt_hash(self.role),
            schema_id=policy.schema_id(self.role),
            schema_hash=schema_hash(self.role),
            policy_hash=policy.content_hash,
            context=context,
            max_output_tokens=policy.max_output_tokens,
            max_cost_microusd=policy.max_cost_microusd_per_call,
        )

    def call(self, context: Mapping[str, Any]) -> Any:
        request = self.build_request(context)
        response = self.executor.execute(request)
        # Domain pass: enums, duplicate IDs and immutable typed objects.
        return parse_role_output(self.role, response.output)


class ProposerAgent(StructuredAgent):
    def __init__(self, executor: StructuredCallExecutor) -> None:
        super().__init__(LLMRole.PROPOSER, executor)


class CriticAgent(StructuredAgent):
    def __init__(self, executor: StructuredCallExecutor) -> None:
        super().__init__(LLMRole.CRITIC, executor)


class RiskAgent(StructuredAgent):
    def __init__(self, executor: StructuredCallExecutor) -> None:
        super().__init__(LLMRole.RISK, executor)


class ArbiterAgent(StructuredAgent):
    def __init__(self, executor: StructuredCallExecutor) -> None:
        super().__init__(LLMRole.ARBITER, executor)
