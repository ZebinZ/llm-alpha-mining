from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Iterator, Mapping, cast

from alpha_research.core.hashing import hash_json, require_sha256


MODEL_EXECUTION_CONTEXT_SCHEMA_VERSION = "model-execution-context/v1"
MODEL_FIT_ATTEMPT_RECEIPT_SCHEMA_VERSION = "model-fit-attempt-receipt/v1"
MODEL_FIT_ATTEMPT_TERMINAL_SCHEMA_VERSION = "model-fit-attempt-terminal/v1"
MODEL_FIT_ATTEMPT_SNAPSHOT_SCHEMA_VERSION = "model-fit-attempt-snapshot/v1"


class ModelExecutionGovernanceError(RuntimeError):
    """Base error for fail-closed model execution governance."""


class ModelFitAttemptBudgetExceeded(ModelExecutionGovernanceError):
    """Raised before preprocessing or fitting when no attempt remains."""


class ModelFitAttemptOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ModelExecutionContext:
    """Stable identity binding one controlled set of model executions."""

    execution_id: str
    execution_scope_hash: str
    schema_version: str = MODEL_EXECUTION_CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_EXECUTION_CONTEXT_SCHEMA_VERSION:
            raise ValueError("unsupported model execution context schema")
        _identity(self.execution_id, name="model execution_id")
        _digest(
            self.execution_scope_hash,
            name="model execution_scope_hash",
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "execution_scope_hash": self.execution_scope_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelExecutionContext:
        expected = {
            "schema_version",
            "execution_id",
            "execution_scope_hash",
        }
        if set(value) != expected:
            raise ValueError("ModelExecutionContext wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            execution_id=_text(value["execution_id"], "execution_id"),
            execution_scope_hash=_text(
                value["execution_scope_hash"], "execution_scope_hash"
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelFitAttemptReceipt:
    """Immutable reservation proving one budget unit was consumed."""

    execution_context_hash: str
    attempt_ordinal: int
    fold_attempt_ordinal: int
    fold_id: str
    model_spec_hash: str
    validation_receipt_hash: str
    previous_attempt_receipt_hash: str | None
    receipt_id: str
    schema_version: str = MODEL_FIT_ATTEMPT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_FIT_ATTEMPT_RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported model fit attempt receipt schema")
        _digest(
            self.execution_context_hash,
            name="model fit attempt execution_context_hash",
        )
        _positive_integer(self.attempt_ordinal, name="attempt_ordinal")
        _positive_integer(
            self.fold_attempt_ordinal,
            name="fold_attempt_ordinal",
        )
        _identity(self.fold_id, name="model fit attempt fold_id")
        _digest(self.model_spec_hash, name="model fit attempt model_spec_hash")
        _digest(
            self.validation_receipt_hash,
            name="model fit attempt validation_receipt_hash",
        )
        _optional_digest(
            self.previous_attempt_receipt_hash,
            name="model fit attempt previous_attempt_receipt_hash",
        )
        _digest(self.receipt_id, name="model fit attempt receipt_id")
        if self.receipt_id != self.recomputed_receipt_id:
            raise ValueError("model fit attempt receipt hash differs")

    @property
    def recomputed_receipt_id(self) -> str:
        return cast(str, hash_json(self._identity_payload()))

    @property
    def content_hash(self) -> str:
        return self.receipt_id

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution_context_hash": self.execution_context_hash,
            "attempt_ordinal": self.attempt_ordinal,
            "fold_attempt_ordinal": self.fold_attempt_ordinal,
            "fold_id": self.fold_id,
            "model_spec_hash": self.model_spec_hash,
            "validation_receipt_hash": self.validation_receipt_hash,
            "previous_attempt_receipt_hash": self.previous_attempt_receipt_hash,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "receipt_id": self.receipt_id}

    @classmethod
    def create(
        cls,
        *,
        execution_context_hash: str,
        attempt_ordinal: int,
        fold_attempt_ordinal: int,
        fold_id: str,
        model_spec_hash: str,
        validation_receipt_hash: str,
        previous_attempt_receipt_hash: str | None,
    ) -> ModelFitAttemptReceipt:
        payload: dict[str, object] = {
            "schema_version": MODEL_FIT_ATTEMPT_RECEIPT_SCHEMA_VERSION,
            "execution_context_hash": execution_context_hash,
            "attempt_ordinal": attempt_ordinal,
            "fold_attempt_ordinal": fold_attempt_ordinal,
            "fold_id": fold_id,
            "model_spec_hash": model_spec_hash,
            "validation_receipt_hash": validation_receipt_hash,
            "previous_attempt_receipt_hash": previous_attempt_receipt_hash,
        }
        return cls(
            execution_context_hash=execution_context_hash,
            attempt_ordinal=attempt_ordinal,
            fold_attempt_ordinal=fold_attempt_ordinal,
            fold_id=fold_id,
            model_spec_hash=model_spec_hash,
            validation_receipt_hash=validation_receipt_hash,
            previous_attempt_receipt_hash=previous_attempt_receipt_hash,
            receipt_id=hash_json(payload),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelFitAttemptReceipt:
        expected = {
            "schema_version",
            "execution_context_hash",
            "attempt_ordinal",
            "fold_attempt_ordinal",
            "fold_id",
            "model_spec_hash",
            "validation_receipt_hash",
            "previous_attempt_receipt_hash",
            "receipt_id",
        }
        if set(value) != expected:
            raise ValueError("ModelFitAttemptReceipt wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            execution_context_hash=_text(
                value["execution_context_hash"], "execution_context_hash"
            ),
            attempt_ordinal=_integer(value["attempt_ordinal"], "attempt_ordinal"),
            fold_attempt_ordinal=_integer(
                value["fold_attempt_ordinal"], "fold_attempt_ordinal"
            ),
            fold_id=_text(value["fold_id"], "fold_id"),
            model_spec_hash=_text(value["model_spec_hash"], "model_spec_hash"),
            validation_receipt_hash=_text(
                value["validation_receipt_hash"], "validation_receipt_hash"
            ),
            previous_attempt_receipt_hash=_optional_text(
                value["previous_attempt_receipt_hash"],
                "previous_attempt_receipt_hash",
            ),
            receipt_id=_text(value["receipt_id"], "receipt_id"),
        )


@dataclass(frozen=True, slots=True)
class ModelFitAttemptTerminalReceipt:
    """Immutable terminal event for one previously reserved attempt."""

    execution_context_hash: str
    terminal_ordinal: int
    attempt_ordinal: int
    attempt_receipt_hash: str
    outcome: ModelFitAttemptOutcome | str
    failure_type: str | None
    previous_terminal_receipt_hash: str | None
    receipt_id: str
    schema_version: str = MODEL_FIT_ATTEMPT_TERMINAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_FIT_ATTEMPT_TERMINAL_SCHEMA_VERSION:
            raise ValueError("unsupported model fit attempt terminal schema")
        _digest(
            self.execution_context_hash,
            name="model fit terminal execution_context_hash",
        )
        _positive_integer(self.terminal_ordinal, name="terminal_ordinal")
        _positive_integer(self.attempt_ordinal, name="terminal attempt_ordinal")
        _digest(
            self.attempt_receipt_hash,
            name="model fit terminal attempt_receipt_hash",
        )
        outcome = ModelFitAttemptOutcome(self.outcome)
        object.__setattr__(self, "outcome", outcome)
        if outcome is ModelFitAttemptOutcome.SUCCEEDED:
            if self.failure_type is not None:
                raise ValueError(
                    "successful model fit terminal cannot have failure_type"
                )
        else:
            if self.failure_type is None:
                raise ValueError("failed model fit terminal requires failure_type")
            _identity(self.failure_type, name="model fit terminal failure_type")
        _optional_digest(
            self.previous_terminal_receipt_hash,
            name="model fit terminal previous_terminal_receipt_hash",
        )
        _digest(self.receipt_id, name="model fit terminal receipt_id")
        if self.receipt_id != self.recomputed_receipt_id:
            raise ValueError("model fit attempt terminal hash differs")

    @property
    def recomputed_receipt_id(self) -> str:
        return cast(str, hash_json(self._identity_payload()))

    @property
    def content_hash(self) -> str:
        return self.receipt_id

    def _identity_payload(self) -> dict[str, object]:
        outcome = self.outcome
        if not isinstance(outcome, ModelFitAttemptOutcome):  # pragma: no cover
            raise RuntimeError("model fit terminal outcome was not normalized")
        return {
            "schema_version": self.schema_version,
            "execution_context_hash": self.execution_context_hash,
            "terminal_ordinal": self.terminal_ordinal,
            "attempt_ordinal": self.attempt_ordinal,
            "attempt_receipt_hash": self.attempt_receipt_hash,
            "outcome": outcome.value,
            "failure_type": self.failure_type,
            "previous_terminal_receipt_hash": self.previous_terminal_receipt_hash,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "receipt_id": self.receipt_id}

    @classmethod
    def create(
        cls,
        *,
        execution_context_hash: str,
        terminal_ordinal: int,
        attempt_ordinal: int,
        attempt_receipt_hash: str,
        outcome: ModelFitAttemptOutcome | str,
        failure_type: str | None,
        previous_terminal_receipt_hash: str | None,
    ) -> ModelFitAttemptTerminalReceipt:
        normalized_outcome = ModelFitAttemptOutcome(outcome)
        payload: dict[str, object] = {
            "schema_version": MODEL_FIT_ATTEMPT_TERMINAL_SCHEMA_VERSION,
            "execution_context_hash": execution_context_hash,
            "terminal_ordinal": terminal_ordinal,
            "attempt_ordinal": attempt_ordinal,
            "attempt_receipt_hash": attempt_receipt_hash,
            "outcome": normalized_outcome.value,
            "failure_type": failure_type,
            "previous_terminal_receipt_hash": previous_terminal_receipt_hash,
        }
        return cls(
            execution_context_hash=execution_context_hash,
            terminal_ordinal=terminal_ordinal,
            attempt_ordinal=attempt_ordinal,
            attempt_receipt_hash=attempt_receipt_hash,
            outcome=normalized_outcome,
            failure_type=failure_type,
            previous_terminal_receipt_hash=previous_terminal_receipt_hash,
            receipt_id=hash_json(payload),
        )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> ModelFitAttemptTerminalReceipt:
        expected = {
            "schema_version",
            "execution_context_hash",
            "terminal_ordinal",
            "attempt_ordinal",
            "attempt_receipt_hash",
            "outcome",
            "failure_type",
            "previous_terminal_receipt_hash",
            "receipt_id",
        }
        if set(value) != expected:
            raise ValueError("ModelFitAttemptTerminalReceipt wire fields differ")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            execution_context_hash=_text(
                value["execution_context_hash"], "execution_context_hash"
            ),
            terminal_ordinal=_integer(value["terminal_ordinal"], "terminal_ordinal"),
            attempt_ordinal=_integer(value["attempt_ordinal"], "attempt_ordinal"),
            attempt_receipt_hash=_text(
                value["attempt_receipt_hash"], "attempt_receipt_hash"
            ),
            outcome=_text(value["outcome"], "outcome"),
            failure_type=_optional_text(value["failure_type"], "failure_type"),
            previous_terminal_receipt_hash=_optional_text(
                value["previous_terminal_receipt_hash"],
                "previous_terminal_receipt_hash",
            ),
            receipt_id=_text(value["receipt_id"], "receipt_id"),
        )


@dataclass(frozen=True, slots=True)
class ModelFitAttemptSnapshot:
    """Content-addressed, independently verifiable ledger snapshot."""

    execution_context: ModelExecutionContext
    execution_context_hash: str
    maximum_attempts: int
    attempts: tuple[ModelFitAttemptReceipt, ...]
    terminals: tuple[ModelFitAttemptTerminalReceipt, ...]
    snapshot_id: str
    schema_version: str = MODEL_FIT_ATTEMPT_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_FIT_ATTEMPT_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported model fit attempt snapshot schema")
        if not isinstance(self.execution_context, ModelExecutionContext):
            raise TypeError("model fit attempt snapshot context type differs")
        _digest(
            self.execution_context_hash,
            name="model fit attempt snapshot execution_context_hash",
        )
        if self.execution_context.content_hash != self.execution_context_hash:
            raise ValueError("model fit attempt snapshot context hash differs")
        _nonnegative_integer(self.maximum_attempts, name="maximum_attempts")
        attempts = tuple(self.attempts)
        terminals = tuple(self.terminals)
        if not all(isinstance(item, ModelFitAttemptReceipt) for item in attempts):
            raise TypeError("model fit attempt snapshot attempt type differs")
        if not all(
            isinstance(item, ModelFitAttemptTerminalReceipt) for item in terminals
        ):
            raise TypeError("model fit attempt snapshot terminal type differs")
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "terminals", terminals)
        self._verify_attempts(attempts)
        self._verify_terminals(attempts, terminals)
        if len(attempts) > self.maximum_attempts:
            raise ValueError("model fit attempt snapshot exceeds budget")
        _digest(self.snapshot_id, name="model fit attempt snapshot_id")
        if self.snapshot_id != self.recomputed_snapshot_id:
            raise ValueError("model fit attempt snapshot hash differs")

    @property
    def consumed_attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def remaining_attempt_count(self) -> int:
        return self.maximum_attempts - self.consumed_attempt_count

    @property
    def terminal_attempt_count(self) -> int:
        return len(self.terminals)

    @property
    def open_attempt_count(self) -> int:
        return self.consumed_attempt_count - self.terminal_attempt_count

    @property
    def recomputed_snapshot_id(self) -> str:
        return cast(str, hash_json(self._identity_payload()))

    @property
    def content_hash(self) -> str:
        return self.snapshot_id

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution_context": self.execution_context.to_dict(),
            "execution_context_hash": self.execution_context_hash,
            "maximum_attempts": self.maximum_attempts,
            "attempts": [item.to_dict() for item in self.attempts],
            "terminals": [item.to_dict() for item in self.terminals],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "snapshot_id": self.snapshot_id}

    def _verify_attempts(self, attempts: tuple[ModelFitAttemptReceipt, ...]) -> None:
        previous_hash: str | None = None
        fold_attempt_counts: dict[tuple[str, str, str], int] = {}
        receipt_ids: set[str] = set()
        for expected_ordinal, attempt in enumerate(attempts, start=1):
            if attempt.execution_context_hash != self.execution_context_hash:
                raise ValueError("model fit attempt context binding differs")
            if attempt.attempt_ordinal != expected_ordinal:
                raise ValueError("model fit attempts are out of order")
            if attempt.previous_attempt_receipt_hash != previous_hash:
                raise ValueError("model fit attempt receipt chain differs")
            if attempt.receipt_id in receipt_ids:
                raise ValueError("model fit attempt receipt ids must be unique")
            key = (
                attempt.model_spec_hash,
                attempt.validation_receipt_hash,
                attempt.fold_id,
            )
            expected_fold_ordinal = fold_attempt_counts.get(key, 0) + 1
            if attempt.fold_attempt_ordinal != expected_fold_ordinal:
                raise ValueError("model fold fit attempts are out of order")
            fold_attempt_counts[key] = expected_fold_ordinal
            receipt_ids.add(attempt.receipt_id)
            previous_hash = attempt.receipt_id

    def _verify_terminals(
        self,
        attempts: tuple[ModelFitAttemptReceipt, ...],
        terminals: tuple[ModelFitAttemptTerminalReceipt, ...],
    ) -> None:
        attempts_by_hash = {item.receipt_id: item for item in attempts}
        previous_hash: str | None = None
        terminal_attempts: set[str] = set()
        terminal_ids: set[str] = set()
        for expected_ordinal, terminal in enumerate(terminals, start=1):
            if terminal.execution_context_hash != self.execution_context_hash:
                raise ValueError("model fit terminal context binding differs")
            if terminal.terminal_ordinal != expected_ordinal:
                raise ValueError("model fit terminals are out of order")
            if terminal.previous_terminal_receipt_hash != previous_hash:
                raise ValueError("model fit terminal receipt chain differs")
            attempt = attempts_by_hash.get(terminal.attempt_receipt_hash)
            if attempt is None:
                raise ValueError("model fit terminal references unknown attempt")
            if terminal.attempt_ordinal != attempt.attempt_ordinal:
                raise ValueError("model fit terminal attempt ordinal differs")
            if terminal.attempt_receipt_hash in terminal_attempts:
                raise ValueError("model fit attempt has duplicate terminal events")
            if terminal.receipt_id in terminal_ids:
                raise ValueError("model fit terminal receipt ids must be unique")
            terminal_attempts.add(terminal.attempt_receipt_hash)
            terminal_ids.add(terminal.receipt_id)
            previous_hash = terminal.receipt_id

    @classmethod
    def create(
        cls,
        *,
        execution_context: ModelExecutionContext,
        maximum_attempts: int,
        attempts: tuple[ModelFitAttemptReceipt, ...],
        terminals: tuple[ModelFitAttemptTerminalReceipt, ...],
    ) -> ModelFitAttemptSnapshot:
        payload: dict[str, object] = {
            "schema_version": MODEL_FIT_ATTEMPT_SNAPSHOT_SCHEMA_VERSION,
            "execution_context": execution_context.to_dict(),
            "execution_context_hash": execution_context.content_hash,
            "maximum_attempts": maximum_attempts,
            "attempts": [item.to_dict() for item in attempts],
            "terminals": [item.to_dict() for item in terminals],
        }
        return cls(
            execution_context=execution_context,
            execution_context_hash=execution_context.content_hash,
            maximum_attempts=maximum_attempts,
            attempts=attempts,
            terminals=terminals,
            snapshot_id=hash_json(payload),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelFitAttemptSnapshot:
        expected = {
            "schema_version",
            "execution_context",
            "execution_context_hash",
            "maximum_attempts",
            "attempts",
            "terminals",
            "snapshot_id",
        }
        if set(value) != expected:
            raise ValueError("ModelFitAttemptSnapshot wire fields differ")
        attempt_values = _mapping_tuple(value["attempts"], "attempts")
        terminal_values = _mapping_tuple(value["terminals"], "terminals")
        return cls(
            schema_version=_text(value["schema_version"], "schema_version"),
            execution_context=ModelExecutionContext.from_mapping(
                _mapping(value["execution_context"], "execution_context")
            ),
            execution_context_hash=_text(
                value["execution_context_hash"], "execution_context_hash"
            ),
            maximum_attempts=_integer(value["maximum_attempts"], "maximum_attempts"),
            attempts=tuple(
                ModelFitAttemptReceipt.from_mapping(item) for item in attempt_values
            ),
            terminals=tuple(
                ModelFitAttemptTerminalReceipt.from_mapping(item)
                for item in terminal_values
            ),
            snapshot_id=_text(value["snapshot_id"], "snapshot_id"),
        )


class ModelFitAttemptLedger:
    """Thread-safe append-only controller for actual fold execution attempts."""

    __slots__ = (
        "_attempts",
        "_context",
        "_lock",
        "_maximum_attempts",
        "_terminals",
    )

    def __init__(
        self,
        *,
        context: ModelExecutionContext,
        maximum_attempts: int,
    ) -> None:
        # Constructing a snapshot applies the same strict checks used on reload.
        snapshot = ModelFitAttemptSnapshot.create(
            execution_context=context,
            maximum_attempts=maximum_attempts,
            attempts=(),
            terminals=(),
        )
        self._context = snapshot.execution_context
        self._maximum_attempts = snapshot.maximum_attempts
        self._attempts: list[ModelFitAttemptReceipt] = []
        self._terminals: list[ModelFitAttemptTerminalReceipt] = []
        self._lock = RLock()

    @property
    def context(self) -> ModelExecutionContext:
        return self._context

    @property
    def maximum_attempts(self) -> int:
        return self._maximum_attempts

    def snapshot(self) -> ModelFitAttemptSnapshot:
        with self._lock:
            return ModelFitAttemptSnapshot.create(
                execution_context=self._context,
                maximum_attempts=self._maximum_attempts,
                attempts=tuple(self._attempts),
                terminals=tuple(self._terminals),
            )

    @classmethod
    def from_snapshot(cls, snapshot: ModelFitAttemptSnapshot) -> ModelFitAttemptLedger:
        if not isinstance(snapshot, ModelFitAttemptSnapshot):
            raise TypeError("model fit attempt ledger snapshot type differs")
        verified = ModelFitAttemptSnapshot.from_mapping(snapshot.to_dict())
        ledger = cls(
            context=verified.execution_context,
            maximum_attempts=verified.maximum_attempts,
        )
        ledger._attempts.extend(verified.attempts)
        ledger._terminals.extend(verified.terminals)
        return ledger

    def reserve(
        self,
        *,
        context: ModelExecutionContext,
        model_spec_hash: str,
        validation_receipt_hash: str,
        fold_id: str,
    ) -> ModelFitAttemptReceipt:
        self._require_context(context)
        _digest(model_spec_hash, name="model fit reservation model_spec_hash")
        _digest(
            validation_receipt_hash,
            name="model fit reservation validation_receipt_hash",
        )
        _identity(fold_id, name="model fit reservation fold_id")
        with self._lock:
            if len(self._attempts) >= self._maximum_attempts:
                raise ModelFitAttemptBudgetExceeded(
                    "model_fit_attempt_budget_exhausted"
                )
            matching_attempts = sum(
                item.model_spec_hash == model_spec_hash
                and item.validation_receipt_hash == validation_receipt_hash
                and item.fold_id == fold_id
                for item in self._attempts
            )
            receipt = ModelFitAttemptReceipt.create(
                execution_context_hash=self._context.content_hash,
                attempt_ordinal=len(self._attempts) + 1,
                fold_attempt_ordinal=matching_attempts + 1,
                fold_id=fold_id,
                model_spec_hash=model_spec_hash,
                validation_receipt_hash=validation_receipt_hash,
                previous_attempt_receipt_hash=(
                    self._attempts[-1].receipt_id if self._attempts else None
                ),
            )
            self._attempts.append(receipt)
            return receipt

    def terminalize(
        self,
        attempt: ModelFitAttemptReceipt,
        *,
        outcome: ModelFitAttemptOutcome | str,
        failure_type: str | None = None,
    ) -> ModelFitAttemptTerminalReceipt:
        if not isinstance(attempt, ModelFitAttemptReceipt):
            raise TypeError("model fit terminal attempt type differs")
        normalized_outcome = ModelFitAttemptOutcome(outcome)
        with self._lock:
            stored = next(
                (
                    item
                    for item in self._attempts
                    if item.receipt_id == attempt.receipt_id
                ),
                None,
            )
            if stored is None or stored.to_dict() != attempt.to_dict():
                raise ModelExecutionGovernanceError(
                    "model_fit_attempt_not_reserved_by_ledger"
                )
            if any(
                item.attempt_receipt_hash == attempt.receipt_id
                for item in self._terminals
            ):
                raise ModelExecutionGovernanceError(
                    "model_fit_attempt_already_terminal"
                )
            terminal = ModelFitAttemptTerminalReceipt.create(
                execution_context_hash=self._context.content_hash,
                terminal_ordinal=len(self._terminals) + 1,
                attempt_ordinal=attempt.attempt_ordinal,
                attempt_receipt_hash=attempt.receipt_id,
                outcome=normalized_outcome,
                failure_type=failure_type,
                previous_terminal_receipt_hash=(
                    self._terminals[-1].receipt_id if self._terminals else None
                ),
            )
            self._terminals.append(terminal)
            return terminal

    @contextmanager
    def attempt(
        self,
        *,
        context: ModelExecutionContext,
        model_spec_hash: str,
        validation_receipt_hash: str,
        fold_id: str,
    ) -> Iterator[ModelFitAttemptReceipt]:
        """Reserve before yielding; every yielded execution gets one terminal."""

        receipt = self.reserve(
            context=context,
            model_spec_hash=model_spec_hash,
            validation_receipt_hash=validation_receipt_hash,
            fold_id=fold_id,
        )
        try:
            yield receipt
        except BaseException as exc:
            self.terminalize(
                receipt,
                outcome=ModelFitAttemptOutcome.FAILED,
                failure_type=_exception_type(exc),
            )
            raise
        else:
            self.terminalize(
                receipt,
                outcome=ModelFitAttemptOutcome.SUCCEEDED,
            )

    def _require_context(self, context: ModelExecutionContext) -> None:
        if not isinstance(context, ModelExecutionContext):
            raise TypeError("model fit attempt execution context type differs")
        if (
            context.content_hash != self._context.content_hash
            or context.to_dict() != self._context.to_dict()
        ):
            raise ModelExecutionGovernanceError(
                "model_fit_attempt_execution_context_differs"
            )


def _exception_type(exc: BaseException) -> str:
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def _identity(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty canonical text")
    return value


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return cast(str, require_sha256(value, name=name))


def _optional_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _digest(value, name=name)


def _positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _mapping_tuple(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a mapping sequence")
    return tuple(_mapping(item, name) for item in value)


__all__ = [
    "MODEL_EXECUTION_CONTEXT_SCHEMA_VERSION",
    "MODEL_FIT_ATTEMPT_RECEIPT_SCHEMA_VERSION",
    "MODEL_FIT_ATTEMPT_SNAPSHOT_SCHEMA_VERSION",
    "MODEL_FIT_ATTEMPT_TERMINAL_SCHEMA_VERSION",
    "ModelExecutionContext",
    "ModelExecutionGovernanceError",
    "ModelFitAttemptBudgetExceeded",
    "ModelFitAttemptLedger",
    "ModelFitAttemptOutcome",
    "ModelFitAttemptReceipt",
    "ModelFitAttemptSnapshot",
    "ModelFitAttemptTerminalReceipt",
]
