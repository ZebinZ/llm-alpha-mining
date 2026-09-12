from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from llm_alpha_mining.mining.artifacts.hashing import canonical_json_bytes, hash_json
from llm_alpha_mining.mining.llm.domain import (
    StructuredCallRequest,
    StructuredCallResponse,
    Usage,
)

if TYPE_CHECKING:
    from llm_alpha_mining.mining.llm.budget import BudgetSnapshot, LLMBudgetLimits

try:  # POSIX is the supported unattended execution target.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


GENESIS_HASH = "0" * 64
LEDGER_SCHEMA = "structured-llm-ledger/v2"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{1,95}")
SAFE_ERROR_CODES = frozenset(
    {
        "transport_dispatch_indeterminate",
        "transport_retryable_failure",
        "transport_terminal_failure",
        "untyped_transport_failure",
        "provider_usage_exceeded_reservation",
        "response_dlp_rejected",
        "structured_output_invalid",
        "malformed_transport_response",
        "response_request_hash_mismatch",
    }
)
_LIMIT_FIELDS = {
    "max_calls",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "max_cost_microusd",
}
_USAGE_FIELDS = {"input_tokens", "output_tokens", "cost_microusd"}


class LedgerIntegrityError(ValueError):
    pass


class IndeterminateCallError(RuntimeError):
    """Recovery must not redispatch an operation with unknown provider state."""


@dataclass(frozen=True, slots=True)
class LedgerPin:
    """Values that must be frozen in an artifact manifest outside the ledger."""

    head_hash: str
    file_hash: str
    record_count: int
    schema_version: str = "structured-llm-ledger-pin/v1"

    def __post_init__(self) -> None:
        if (
            _DIGEST.fullmatch(self.head_hash) is None
            or _DIGEST.fullmatch(self.file_hash) is None
        ):
            raise LedgerIntegrityError(
                "ledger pin hashes must be lowercase SHA-256 digests"
            )
        if (
            not isinstance(self.record_count, int)
            or isinstance(self.record_count, bool)
            or self.record_count < 0
        ):
            raise LedgerIntegrityError("ledger pin record_count must be non-negative")
        if self.schema_version != "structured-llm-ledger-pin/v1":
            raise LedgerIntegrityError("unsupported ledger pin schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "head_hash": self.head_hash,
            "file_hash": self.file_hash,
            "record_count": self.record_count,
        }


@dataclass(frozen=True, slots=True)
class LedgerOperation:
    operation_id: str
    state: str
    request_hash: str
    response: StructuredCallResponse | None
    next_attempt: int


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _limits_dict(limits: "LLMBudgetLimits") -> dict[str, int]:
    return {name: int(getattr(limits, name)) for name in sorted(_LIMIT_FIELDS)}


def _zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "cost_microusd": 0}


def _validate_nonnegative_int_mapping(
    value: Any, fields: set[str], name: str
) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise LedgerIntegrityError(f"{name} fields must be exact")
    result = dict(value)
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in result.values()
    ):
        raise LedgerIntegrityError(f"{name} values must be non-negative integers")
    return result


class CallLedger:
    """Cross-process atomic call state, budget journal and hash-chain ledger.

    A non-empty ledger cannot be reopened without an expected head *and* file
    hash supplied by a separately frozen manifest.  This closes the otherwise
    undetectable "edit every row and recompute the chain" attack.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_head_hash: str | None = None,
        expected_file_hash: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._reload()
        if self._records:
            if expected_head_hash is None or expected_file_hash is None:
                raise LedgerIntegrityError(
                    "external expected ledger head and file hash are required"
                )
            if self._records[-1]["record_hash"] != expected_head_hash:
                raise LedgerIntegrityError("external expected ledger head mismatch")
            if _file_hash(self.path) != expected_file_hash:
                raise LedgerIntegrityError(
                    "external expected ledger file hash mismatch"
                )
        elif expected_head_hash not in {None, GENESIS_HASH}:
            raise LedgerIntegrityError("empty ledger does not match expected head")
        elif (
            expected_file_hash is not None
            and _file_hash(self.path) != expected_file_hash
        ):
            raise LedgerIntegrityError("empty ledger does not match expected file hash")

    def _read_handle(self, handle: Any) -> list[dict[str, Any]]:
        handle.seek(0)
        records: list[dict[str, Any]] = []
        previous = GENESIS_HASH
        for line_number, line in enumerate(handle.read().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise LedgerIntegrityError(
                    f"invalid ledger JSON at line {line_number}"
                ) from None
            expected = {
                "schema_version",
                "seq",
                "created_ns",
                "prev_hash",
                "event",
                "operation_id",
                "request_hash",
                "role",
                "attempt",
                "details",
                "record_hash",
            }
            if not isinstance(record, dict) or set(record) != expected:
                raise LedgerIntegrityError(
                    f"invalid ledger fields at line {line_number}"
                )
            supplied_hash = record["record_hash"]
            body = {key: value for key, value in record.items() if key != "record_hash"}
            if record["schema_version"] != LEDGER_SCHEMA:
                raise LedgerIntegrityError(
                    f"unsupported ledger schema at line {line_number}"
                )
            if record["seq"] != len(records) + 1:
                raise LedgerIntegrityError(
                    f"non-contiguous ledger sequence at line {line_number}"
                )
            if record["prev_hash"] != previous:
                raise LedgerIntegrityError(f"broken ledger chain at line {line_number}")
            if supplied_hash != hash_json(body):
                raise LedgerIntegrityError(
                    f"ledger record hash mismatch at line {line_number}"
                )
            if _DIGEST.fullmatch(str(record["request_hash"])) is None:
                raise LedgerIntegrityError(
                    f"invalid request hash at line {line_number}"
                )
            previous = supplied_hash
            records.append(record)
        self._validate_state_machine(records)
        return records

    @staticmethod
    def _validate_state_machine(records: list[dict[str, Any]]) -> None:
        operations: dict[str, dict[str, Any]] = {}
        bound_limits_hash: str | None = None
        for record in records:
            operation_id = record["operation_id"]
            if not isinstance(operation_id, str) or not operation_id:
                raise LedgerIntegrityError("empty operation_id")
            attempt = record["attempt"]
            if (
                not isinstance(attempt, int)
                or isinstance(attempt, bool)
                or attempt <= 0
            ):
                raise LedgerIntegrityError("attempt must be a positive integer")
            state = operations.get(operation_id)
            if state is not None:
                if (
                    state["request_hash"] != record["request_hash"]
                    or state["role"] != record["role"]
                ):
                    raise LedgerIntegrityError(
                        "operation identity changed inside ledger"
                    )
            event = record["event"]
            details = record["details"]
            if not isinstance(details, Mapping):
                raise LedgerIntegrityError("ledger details must be an object")
            if event == "attempt_started":
                expected = {"reservation", "budget_limits", "budget_limits_hash"}
                if set(details) != expected:
                    raise LedgerIntegrityError(
                        "attempt_started details fields must be exact"
                    )
                reservation = _validate_nonnegative_int_mapping(
                    details["reservation"], _USAGE_FIELDS, "reservation"
                )
                limits = _validate_nonnegative_int_mapping(
                    details["budget_limits"], _LIMIT_FIELDS, "budget limits"
                )
                if any(value <= 0 for value in limits.values()):
                    raise LedgerIntegrityError("budget limits must be positive")
                limits_hash = hash_json(limits)
                if details["budget_limits_hash"] != limits_hash:
                    raise LedgerIntegrityError("budget limits hash mismatch")
                if bound_limits_hash is None:
                    bound_limits_hash = limits_hash
                elif bound_limits_hash != limits_hash:
                    raise LedgerIntegrityError("multiple budget policies in one ledger")
                if state is None:
                    if attempt != 1:
                        raise LedgerIntegrityError(
                            "new operation must start at attempt one"
                        )
                elif state["phase"] != "retryable" or attempt != state["attempt"] + 1:
                    raise LedgerIntegrityError("invalid attempt_started transition")
                operations[operation_id] = {
                    "request_hash": record["request_hash"],
                    "role": record["role"],
                    "attempt": attempt,
                    "phase": "active",
                    "reservation": reservation,
                }
                continue
            if (
                state is None
                or state["phase"] != "active"
                or attempt != state["attempt"]
            ):
                raise LedgerIntegrityError(f"invalid terminal transition for {event}")
            if event not in {
                "attempt_retryable_failure",
                "call_indeterminate",
                "call_terminal_failure",
                "call_succeeded",
            }:
                raise LedgerIntegrityError(f"unknown ledger event: {event}")
            if event == "call_succeeded":
                if set(details) != {"charge", "response"}:
                    raise LedgerIntegrityError(
                        "call_succeeded details fields must be exact"
                    )
                response = StructuredCallResponse.from_dict(details["response"])
                if response.request_hash != record["request_hash"]:
                    raise LedgerIntegrityError(
                        "persisted response request hash mismatch"
                    )
                charge = _validate_nonnegative_int_mapping(
                    details["charge"], _USAGE_FIELDS, "charge"
                )
                if charge != response.usage.to_dict():
                    raise LedgerIntegrityError(
                        "success charge differs from response usage"
                    )
                if any(
                    charge[name] > state["reservation"][name] for name in _USAGE_FIELDS
                ):
                    raise LedgerIntegrityError("successful usage exceeds reservation")
                phase = "succeeded"
            else:
                if set(details) != {"charge", "error_code", "response_hash"}:
                    raise LedgerIntegrityError(f"{event} details fields must be exact")
                charge = _validate_nonnegative_int_mapping(
                    details["charge"], _USAGE_FIELDS, "charge"
                )
                if (
                    _ERROR_CODE.fullmatch(str(details["error_code"])) is None
                    or details["error_code"] not in SAFE_ERROR_CODES
                ):
                    raise LedgerIntegrityError("unsafe persisted error code")
                response_hash = details["response_hash"]
                if (
                    response_hash is not None
                    and _DIGEST.fullmatch(str(response_hash)) is None
                ):
                    raise LedgerIntegrityError("response_hash must be null or SHA-256")
                if event == "attempt_retryable_failure" and charge != _zero_usage():
                    raise LedgerIntegrityError(
                        "definitely-unsent retryable failure must have zero charge"
                    )
                if event == "call_indeterminate" and charge != state["reservation"]:
                    raise LedgerIntegrityError(
                        "indeterminate call must conservatively charge reservation"
                    )
                phase = {
                    "attempt_retryable_failure": "retryable",
                    "call_indeterminate": "indeterminate",
                    "call_terminal_failure": "terminal_failure",
                }[event]
            state["phase"] = phase
            state["charge"] = charge

    def _reload(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                records = self._read_handle(handle)
                self._assert_continuation(records)
                self._records = records
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _assert_continuation(self, records: list[dict[str, Any]]) -> None:
        """A live handle accepts append-only growth, never rewind/re-chaining."""

        if not self._records:
            return
        if len(records) < len(self._records):
            raise LedgerIntegrityError("ledger was truncated after verification")
        for index, previous in enumerate(self._records):
            if records[index]["record_hash"] != previous["record_hash"]:
                raise LedgerIntegrityError("verified ledger prefix was rewritten")

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        self._reload()
        return tuple(dict(item) for item in self._records)

    @property
    def head_hash(self) -> str:
        self._reload()
        return self._records[-1]["record_hash"] if self._records else GENESIS_HASH

    @property
    def pin(self) -> LedgerPin:
        # Head/count and the exact file bytes must come from one locked
        # snapshot.  Reload-then-reopen allowed a concurrent writer to produce
        # an impossible pin containing the old logical head and new file hash.
        # The optimistic probe gives a writer already racing this read a chance
        # to finish, then verifies that the authoritative locked snapshot is of
        # that same file generation.  The hash placed in the pin is still
        # computed directly from bytes read inside the lock.
        for _ in range(3):
            optimistic_hash = _file_hash(self.path)
            with self.path.open("rb") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                try:
                    raw = handle.read()
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        raise LedgerIntegrityError(
                            "ledger is not valid UTF-8"
                        ) from None
                    records = self._read_handle(io.StringIO(text))
                    self._assert_continuation(records)
                    file_hash = hashlib.sha256(raw).hexdigest()
                    pin = LedgerPin(
                        head_hash=(
                            records[-1]["record_hash"] if records else GENESIS_HASH
                        ),
                        file_hash=file_hash,
                        record_count=len(records),
                    )
                    self._records = records
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if file_hash == optimistic_hash:
                return pin
        # Even under sustained appends, the last pin remains internally atomic;
        # stabilization is only an additional reopen-friendly guarantee.
        return pin

    def verify_against_pin(self, pin: LedgerPin) -> None:
        self._reload()
        current = self.pin
        if current != pin:
            raise LedgerIntegrityError("ledger differs from externally frozen pin")

    def _append_locked(
        self,
        handle: Any,
        records: list[dict[str, Any]],
        *,
        event: str,
        request: StructuredCallRequest,
        attempt: int,
        details: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        body = {
            "schema_version": LEDGER_SCHEMA,
            "seq": len(records) + 1,
            "created_ns": time.time_ns(),
            "prev_hash": records[-1]["record_hash"] if records else GENESIS_HASH,
            "event": event,
            "operation_id": request.operation_id,
            "request_hash": request.request_hash,
            "role": request.role.value,
            "attempt": attempt,
            "details": dict(details),
        }
        record = {**body, "record_hash": hash_json(body)}
        trial = records + [record]
        self._validate_state_machine(trial)
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._records = trial
        return dict(record)

    @staticmethod
    def _operation_from_records(
        records: list[dict[str, Any]], operation_id: str
    ) -> LedgerOperation | None:
        rows = [row for row in records if row["operation_id"] == operation_id]
        if not rows:
            return None
        latest = rows[-1]
        event = latest["event"]
        response = None
        if event == "call_succeeded":
            response = StructuredCallResponse.from_dict(latest["details"]["response"])
            state = "succeeded"
        elif event == "attempt_retryable_failure":
            state = "retryable_failure"
        elif event in {"call_indeterminate", "attempt_started"}:
            state = "indeterminate"
        elif event == "call_terminal_failure":
            state = "terminal_failure"
        else:  # pragma: no cover - state validator owns this invariant
            raise LedgerIntegrityError(f"unknown ledger event: {event}")
        return LedgerOperation(
            operation_id=operation_id,
            state=state,
            request_hash=latest["request_hash"],
            response=response,
            next_attempt=max(int(row["attempt"]) for row in rows) + 1,
        )

    def lookup(self, operation_id: str) -> LedgerOperation | None:
        self._reload()
        return self._operation_from_records(self._records, operation_id)

    @staticmethod
    def _budget_values(records: list[dict[str, Any]]) -> tuple[int, int, int, int, int]:
        calls = 0
        actual_input = actual_output = actual_cost = 0
        active: dict[tuple[str, int], dict[str, int]] = {}
        for row in records:
            key = (row["operation_id"], row["attempt"])
            if row["event"] == "attempt_started":
                calls += 1
                active[key] = dict(row["details"]["reservation"])
            else:
                active.pop(key, None)
                charge = row["details"]["charge"]
                actual_input += charge["input_tokens"]
                actual_output += charge["output_tokens"]
                actual_cost += charge["cost_microusd"]
        pending_input = sum(item["input_tokens"] for item in active.values())
        pending_output = sum(item["output_tokens"] for item in active.values())
        pending_cost = sum(item["cost_microusd"] for item in active.values())
        return (
            calls,
            actual_input + pending_input,
            actual_output + pending_output,
            actual_cost + pending_cost,
            len(active),
        )

    @staticmethod
    def _enforce_budget(
        records: list[dict[str, Any]], limits: "LLMBudgetLimits"
    ) -> None:
        from llm_alpha_mining.mining.llm.budget import LLMBudgetExceeded

        calls, input_tokens, output_tokens, cost, _ = CallLedger._budget_values(records)
        checks = {
            "calls": (calls, limits.max_calls),
            "input_tokens": (input_tokens, limits.max_input_tokens),
            "output_tokens": (output_tokens, limits.max_output_tokens),
            "total_tokens": (input_tokens + output_tokens, limits.max_total_tokens),
            "cost_microusd": (cost, limits.max_cost_microusd),
        }
        exceeded = [
            name for name, (actual, maximum) in checks.items() if actual > maximum
        ]
        if exceeded:
            raise LLMBudgetExceeded("LLM budget exceeded: " + ",".join(exceeded))

    def budget_snapshot(self, limits: "LLMBudgetLimits") -> "BudgetSnapshot":
        from llm_alpha_mining.mining.llm.budget import BudgetSnapshot

        self._reload()
        calls, input_tokens, output_tokens, cost, pending = self._budget_values(
            self._records
        )
        return BudgetSnapshot(
            calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_microusd=cost,
            pending_reservations=pending,
        )

    def reserve_and_start(
        self,
        request: StructuredCallRequest,
        *,
        attempt: int,
        reservation: Mapping[str, int],
        limits: "LLMBudgetLimits",
    ) -> None:
        reservation_dict = _validate_nonnegative_int_mapping(
            reservation, _USAGE_FIELDS, "reservation"
        )
        limits_dict = _limits_dict(limits)
        with self.path.open("r+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                records = self._read_handle(handle)
                self._assert_continuation(records)
                existing = self._operation_from_records(records, request.operation_id)
                if existing is not None:
                    if existing.request_hash != request.request_hash:
                        raise LedgerIntegrityError(
                            "operation_id reuse with a different request"
                        )
                    if (
                        existing.state != "retryable_failure"
                        or attempt != existing.next_attempt
                    ):
                        raise IndeterminateCallError(
                            f"operation {request.operation_id} is {existing.state}; redispatch denied"
                        )
                elif attempt != 1:
                    raise LedgerIntegrityError(
                        "a new operation must begin at attempt one"
                    )
                details = {
                    "reservation": reservation_dict,
                    "budget_limits": limits_dict,
                    "budget_limits_hash": hash_json(limits_dict),
                }
                body = {
                    "schema_version": LEDGER_SCHEMA,
                    "seq": len(records) + 1,
                    "created_ns": 0,
                    "prev_hash": records[-1]["record_hash"]
                    if records
                    else GENESIS_HASH,
                    "event": "attempt_started",
                    "operation_id": request.operation_id,
                    "request_hash": request.request_hash,
                    "role": request.role.value,
                    "attempt": attempt,
                    "details": details,
                }
                provisional = {**body, "record_hash": hash_json(body)}
                self._validate_state_machine(records + [provisional])
                self._enforce_budget(records + [provisional], limits)
                self._append_locked(
                    handle,
                    records,
                    event="attempt_started",
                    request=request,
                    attempt=attempt,
                    details=details,
                )
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _finalize(
        self,
        request: StructuredCallRequest,
        *,
        attempt: int,
        event: str,
        charge: Usage,
        error_code: str | None = None,
        response_hash: str | None = None,
        response: StructuredCallResponse | None = None,
    ) -> None:
        if event == "call_succeeded":
            if response is None or error_code is not None or response_hash is not None:
                raise LedgerIntegrityError("invalid success finalization arguments")
            details: dict[str, Any] = {
                "charge": charge.to_dict(),
                "response": response.to_dict(),
            }
        else:
            if (
                response is not None
                or error_code is None
                or _ERROR_CODE.fullmatch(error_code) is None
                or error_code not in SAFE_ERROR_CODES
            ):
                raise LedgerIntegrityError(
                    "terminal failure requires a closed safe error code"
                )
            if response_hash is not None and _DIGEST.fullmatch(response_hash) is None:
                raise LedgerIntegrityError("response_hash must be SHA-256")
            details = {
                "charge": charge.to_dict(),
                "error_code": error_code,
                "response_hash": response_hash,
            }
        with self.path.open("r+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                records = self._read_handle(handle)
                self._assert_continuation(records)
                self._append_locked(
                    handle,
                    records,
                    event=event,
                    request=request,
                    attempt=attempt,
                    details=details,
                )
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def retryable_failure(
        self, request: StructuredCallRequest, *, attempt: int, error_code: str
    ) -> None:
        self._finalize(
            request,
            attempt=attempt,
            event="attempt_retryable_failure",
            charge=Usage(0, 0, 0),
            error_code=error_code,
        )

    def indeterminate(
        self, request: StructuredCallRequest, *, attempt: int, error_code: str
    ) -> None:
        self._reload()
        rows = [
            row
            for row in self._records
            if row["operation_id"] == request.operation_id
            and row["attempt"] == attempt
            and row["event"] == "attempt_started"
        ]
        if not rows:
            raise LedgerIntegrityError(
                "indeterminate finalization lacks active reservation"
            )
        reservation = Usage.from_dict(rows[-1]["details"]["reservation"])
        self._finalize(
            request,
            attempt=attempt,
            event="call_indeterminate",
            charge=reservation,
            error_code=error_code,
        )

    def terminal_failure(
        self,
        request: StructuredCallRequest,
        *,
        attempt: int,
        error_code: str,
        usage: Usage | None = None,
        response_hash: str | None = None,
    ) -> None:
        if usage is None:
            # Never silently convert an already-dispatched terminal failure to
            # zero usage.  Rejecting leaves the active reservation in place,
            # so budget recovery remains conservative after a crash/restart.
            raise LedgerIntegrityError(
                "terminal failure requires explicit usage after dispatch"
            )
        self._finalize(
            request,
            attempt=attempt,
            event="call_terminal_failure",
            charge=usage,
            error_code=error_code,
            response_hash=response_hash,
        )

    def success(
        self,
        request: StructuredCallRequest,
        *,
        attempt: int,
        response: StructuredCallResponse,
        allowed_model_ids: tuple[str, ...],
    ) -> None:
        if response.request_hash != request.request_hash:
            raise LedgerIntegrityError("response request hash mismatch")
        # Defense in depth for direct callers: the ledger itself refuses an
        # unvalidated response even if they bypass StructuredCallExecutor.
        from llm_alpha_mining.mining.llm.safe_context import (
            assert_safe_context,
            assert_safe_response_metadata,
        )
        from llm_alpha_mining.mining.llm.schemas import (
            ROLE_SCHEMAS,
            parse_role_output,
            validate_strict_json,
        )

        assert_safe_response_metadata(
            model_id=response.model_id,
            provider_request_id=response.provider_request_id,
        )
        if response.model_id not in set(allowed_model_ids):
            raise LedgerIntegrityError(
                "response model is outside the closed allow-list"
            )
        validate_strict_json(ROLE_SCHEMAS[request.role], response.output)
        parsed = parse_role_output(request.role, response.output)
        assert_safe_context(response.output)
        if request.role.value == "proposer":
            if len(parsed) > int(request.context["requested_count"]):
                raise LedgerIntegrityError("proposer exceeded requested_count")
        else:
            descriptors = request.context.get("candidate_descriptors")
            if not isinstance(descriptors, (list, tuple)):
                raise LedgerIntegrityError("review request has no descriptor set")
            try:
                expected_ids = {item["candidate_id"] for item in descriptors}
                actual_ids = {item.candidate_id for item in parsed}
            except Exception:
                raise LedgerIntegrityError(
                    "review descriptor binding is malformed"
                ) from None
            if (
                len(expected_ids) != len(descriptors)
                or len(actual_ids) != len(parsed)
                or actual_ids != expected_ids
            ):
                raise LedgerIntegrityError("review candidate coverage mismatch")
        self._finalize(
            request,
            attempt=attempt,
            event="call_succeeded",
            charge=response.usage,
            response=response,
        )
