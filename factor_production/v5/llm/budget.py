from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Mapping

from factor_production.v5.llm.domain import Usage

if TYPE_CHECKING:
    from factor_production.v5.llm.ledger import CallLedger


class LLMBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LLMBudgetLimits:
    max_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    max_cost_microusd: int

    def __post_init__(self) -> None:
        for name in (
            "max_calls",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
            "max_cost_microusd",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_microusd: int
    pending_reservations: int


class LLMBudget:
    """Reserve-before-dispatch budget; unknown calls retain their reservation."""

    def __init__(self, limits: LLMBudgetLimits) -> None:
        self.limits = limits
        self._lock = RLock()
        self._calls = 0
        self._input = 0
        self._output = 0
        self._cost = 0
        self._reservations: dict[str, dict[str, int]] = {}
        self._ledger: "CallLedger | None" = None

    @classmethod
    def from_ledger(cls, limits: LLMBudgetLimits, ledger: "CallLedger") -> "LLMBudget":
        """Rebuild actual and conservative pending charges after a restart."""

        budget = cls(limits)
        budget.bind(ledger)
        snapshot = ledger.budget_snapshot(limits)
        checks = {
            "calls": (snapshot.calls, limits.max_calls),
            "input_tokens": (snapshot.input_tokens, limits.max_input_tokens),
            "output_tokens": (snapshot.output_tokens, limits.max_output_tokens),
            "total_tokens": (snapshot.total_tokens, limits.max_total_tokens),
            "cost_microusd": (snapshot.cost_microusd, limits.max_cost_microusd),
        }
        exceeded = [name for name, (actual, limit) in checks.items() if actual > limit]
        if exceeded:
            raise LLMBudgetExceeded("recovered LLM budget already exceeds limits: " + ",".join(exceeded))
        return budget

    def bind(self, ledger: "CallLedger") -> None:
        """Bind snapshots to the cross-process ledger, not process-local counters."""

        with self._lock:
            if self._ledger is not None and self._ledger.path != ledger.path:
                raise ValueError("an LLM budget cannot be rebound to another ledger")
            if self._calls or self._reservations or self._input or self._output or self._cost:
                raise ValueError("a used in-memory budget cannot be rebound")
            self._ledger = ledger

    def _projected(self, extra: Mapping[str, int] | None = None) -> BudgetSnapshot:
        reservations = list(self._reservations.values())
        if extra is not None:
            reservations.append(dict(extra))
        pending_input = sum(item["input_tokens"] for item in reservations)
        pending_output = sum(item["output_tokens"] for item in reservations)
        pending_cost = sum(item["cost_microusd"] for item in reservations)
        input_tokens = self._input + pending_input
        output_tokens = self._output + pending_output
        return BudgetSnapshot(
            calls=self._calls + (1 if extra is not None else 0),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_microusd=self._cost + pending_cost,
            pending_reservations=len(reservations),
        )

    def reserve(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_microusd: int,
    ) -> Mapping[str, int]:
        values = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_microusd": cost_microusd,
        }
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values.values()):
            raise ValueError("budget reservation values must be non-negative integers")
        with self._lock:
            if self._ledger is not None:
                raise RuntimeError("bound budgets reserve atomically through CallLedger")
            if reservation_id in self._reservations:
                raise ValueError(f"duplicate budget reservation: {reservation_id}")
            projected = self._projected(values)
            checks = {
                "calls": (projected.calls, self.limits.max_calls),
                "input_tokens": (projected.input_tokens, self.limits.max_input_tokens),
                "output_tokens": (projected.output_tokens, self.limits.max_output_tokens),
                "total_tokens": (projected.total_tokens, self.limits.max_total_tokens),
                "cost_microusd": (projected.cost_microusd, self.limits.max_cost_microusd),
            }
            exceeded = [name for name, (actual, limit) in checks.items() if actual > limit]
            if exceeded:
                raise LLMBudgetExceeded("LLM budget exceeded: " + ",".join(exceeded))
            self._calls += 1
            self._reservations[reservation_id] = values
            return dict(values)

    def settle(self, reservation_id: str, usage: Usage) -> None:
        with self._lock:
            if self._ledger is not None:
                raise RuntimeError("bound budgets settle atomically through CallLedger")
            reserved = self._reservations[reservation_id]
            if usage.input_tokens > reserved["input_tokens"]:
                raise LLMBudgetExceeded("provider input usage exceeded its reservation")
            if usage.output_tokens > reserved["output_tokens"]:
                raise LLMBudgetExceeded("provider output usage exceeded its reservation")
            if usage.cost_microusd > reserved["cost_microusd"]:
                raise LLMBudgetExceeded("provider cost exceeded its reservation")
            # Pop only after validation: an over-budget provider response stays
            # conservatively reserved and cannot be replayed around the cap.
            self._reservations.pop(reservation_id)
            self._input += usage.input_tokens
            self._output += usage.output_tokens
            self._cost += usage.cost_microusd

    def release_known_failure(self, reservation_id: str) -> None:
        with self._lock:
            if self._ledger is not None:
                raise RuntimeError("bound budgets finalize atomically through CallLedger")
            self._reservations.pop(reservation_id)

    @property
    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            if self._ledger is not None:
                return self._ledger.budget_snapshot(self.limits)
            return self._projected()
