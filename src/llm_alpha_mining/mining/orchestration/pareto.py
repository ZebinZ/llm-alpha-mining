from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from llm_alpha_mining.mining.artifacts.hashing import hash_json
from llm_alpha_mining.mining.domain.enums import ScoreVisibility, StopReason
from llm_alpha_mining.mining.orchestration.budget import BudgetController
from llm_alpha_mining.mining.orchestration.stopping import HoldoutLeakageError


PARETO_LEDGER_SCHEMA = "pareto-observation-ledger/v1"
PARETO_CHECKPOINT_SCHEMA = "pareto-governance-checkpoint/v1"
_GENESIS_HASH = hash_json({"schema_version": PARETO_LEDGER_SCHEMA, "genesis": True})


class ParetoGovernanceError(ValueError):
    pass


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ParetoGovernanceError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ParetoGovernanceError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ParetoGovernanceError(f"{name} must be a finite number")
    return number


class ObjectiveDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ParetoObjective(str, Enum):
    NEUTRAL_IC_STRENGTH = "neutral_ic_strength"
    PERIOD_STABILITY = "period_stability"
    NET_COST_SPREAD = "net_cost_spread"
    TURNOVER = "turnover"
    COVERAGE = "coverage"
    NOVELTY = "novelty"
    COMPLEXITY = "complexity"
    COMPUTE_COST = "compute_cost"


_EXPECTED_DIRECTIONS: Mapping[ParetoObjective, ObjectiveDirection] = MappingProxyType(
    {
        ParetoObjective.NEUTRAL_IC_STRENGTH: ObjectiveDirection.MAXIMIZE,
        ParetoObjective.PERIOD_STABILITY: ObjectiveDirection.MAXIMIZE,
        ParetoObjective.NET_COST_SPREAD: ObjectiveDirection.MAXIMIZE,
        ParetoObjective.TURNOVER: ObjectiveDirection.MINIMIZE,
        ParetoObjective.COVERAGE: ObjectiveDirection.MAXIMIZE,
        ParetoObjective.NOVELTY: ObjectiveDirection.MAXIMIZE,
        ParetoObjective.COMPLEXITY: ObjectiveDirection.MINIMIZE,
        ParetoObjective.COMPUTE_COST: ObjectiveDirection.MINIMIZE,
    }
)


@dataclass(frozen=True, slots=True)
class ParetoObjectiveSpec:
    name: ParetoObjective
    direction: ObjectiveDirection
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        try:
            name = ParetoObjective(self.name)
            direction = ObjectiveDirection(self.direction)
        except (TypeError, ValueError) as exc:
            raise ParetoGovernanceError(
                "unknown Pareto objective or direction"
            ) from exc
        if direction is not _EXPECTED_DIRECTIONS[name]:
            raise ParetoGovernanceError(
                f"objective {name.value} must use direction "
                f"{_EXPECTED_DIRECTIONS[name].value}"
            )
        epsilon = _finite_number(self.epsilon, f"epsilon for {name.value}")
        if epsilon < 0:
            raise ParetoGovernanceError("objective epsilon cannot be negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "epsilon", epsilon)

    def to_dict(self) -> dict[str, str | float]:
        return {
            "name": self.name.value,
            "direction": self.direction.value,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParetoObjectiveSpec":
        expected = {"name", "direction", "epsilon"}
        if set(value) != expected:
            raise ParetoGovernanceError(
                f"objective spec fields must be exactly {sorted(expected)}"
            )
        try:
            name = ParetoObjective(str(value["name"]))
            direction = ObjectiveDirection(str(value["direction"]))
        except ValueError as exc:
            raise ParetoGovernanceError(
                "unknown Pareto objective or direction"
            ) from exc
        return cls(name=name, direction=direction, epsilon=value["epsilon"])


DEFAULT_PARETO_OBJECTIVES: tuple[ParetoObjectiveSpec, ...] = tuple(
    ParetoObjectiveSpec(name=name, direction=direction, epsilon=0.0)
    for name, direction in _EXPECTED_DIRECTIONS.items()
)


def objective_specs_with_epsilon(
    epsilons: Mapping[ParetoObjective | str, float],
) -> tuple[ParetoObjectiveSpec, ...]:
    """Create the complete typed objective set with preregistered epsilons."""

    if not isinstance(epsilons, Mapping):
        raise ParetoGovernanceError("epsilons must be an objective mapping")
    normalized: dict[ParetoObjective, float] = {}
    for key, value in epsilons.items():
        try:
            objective = (
                key if isinstance(key, ParetoObjective) else ParetoObjective(str(key))
            )
        except ValueError as exc:
            raise ParetoGovernanceError(f"unknown Pareto objective: {key!r}") from exc
        if objective in normalized:
            raise ParetoGovernanceError(f"duplicate epsilon for {objective.value}")
        normalized[objective] = value
    return tuple(
        ParetoObjectiveSpec(
            name=name,
            direction=_EXPECTED_DIRECTIONS[name],
            epsilon=normalized.get(name, 0.0),
        )
        for name in ParetoObjective
    )


def _validated_specs(
    objective_specs: Sequence[ParetoObjectiveSpec],
) -> tuple[ParetoObjectiveSpec, ...]:
    specs = tuple(objective_specs)
    if any(not isinstance(item, ParetoObjectiveSpec) for item in specs):
        raise ParetoGovernanceError(
            "objective_specs must contain only typed ParetoObjectiveSpec values"
        )
    names = [item.name for item in specs]
    expected = set(ParetoObjective)
    if len(names) != len(expected) or set(names) != expected:
        missing = sorted(item.value for item in expected - set(names))
        duplicates = sorted({item.value for item in names if names.count(item) > 1})
        raise ParetoGovernanceError(
            f"Pareto objective set must contain all eight objectives exactly once; "
            f"missing={missing}, duplicates={duplicates}"
        )
    return specs


def objective_specs_hash(
    objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
) -> str:
    specs = _validated_specs(objective_specs)
    return hash_json([item.to_dict() for item in specs])


@dataclass(frozen=True, slots=True)
class ParetoObservation:
    candidate_id: str
    campaign_round: int
    metrics: Mapping[str, float]
    visibility: ScoreVisibility = ScoreVisibility.LOCAL_RESEARCH

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            raise ParetoGovernanceError("candidate_id must not be empty")
        if (
            not isinstance(self.campaign_round, int)
            or isinstance(self.campaign_round, bool)
            or self.campaign_round < 0
        ):
            raise ParetoGovernanceError("campaign_round must be a non-negative integer")
        try:
            visibility = ScoreVisibility(self.visibility)
        except (TypeError, ValueError) as exc:
            raise HoldoutLeakageError(
                "unknown score visibility cannot enter Pareto governance"
            ) from exc
        if visibility is not ScoreVisibility.LOCAL_RESEARCH:
            raise HoldoutLeakageError(
                "external teacher/official visibility cannot enter Pareto "
                "selection, frontier, or stopping"
            )

        if not isinstance(self.metrics, Mapping):
            raise ParetoGovernanceError("Pareto metrics must be an object")
        normalized: dict[str, float] = {}
        for key, value in self.metrics.items():
            name = key.value if isinstance(key, ParetoObjective) else str(key)
            if name in normalized:
                raise ParetoGovernanceError(f"duplicate objective metric: {name}")
            normalized[name] = _finite_number(value, f"metric {name}")
        expected = {item.value for item in ParetoObjective}
        if set(normalized) != expected:
            missing = sorted(expected - set(normalized))
            extra = sorted(set(normalized) - expected)
            raise ParetoGovernanceError(
                f"Pareto metrics must be complete and exact; missing={missing}, "
                f"extra={extra}"
            )
        ordered = {item.value: normalized[item.value] for item in ParetoObjective}
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "metrics", MappingProxyType(ordered))

    @property
    def generation(self) -> int:
        return self.campaign_round

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def value(self, objective: ParetoObjective | str) -> float:
        name = (
            objective.value
            if isinstance(objective, ParetoObjective)
            else str(objective)
        )
        if name not in self.metrics:
            raise ParetoGovernanceError(f"missing objective metric: {name}")
        return self.metrics[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "campaign_round": self.campaign_round,
            "metrics": dict(self.metrics),
            "visibility": self.visibility.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParetoObservation":
        expected = {"candidate_id", "campaign_round", "metrics", "visibility"}
        if set(value) != expected:
            raise ParetoGovernanceError(
                f"Pareto observation fields must be exactly {sorted(expected)}"
            )
        metrics = value["metrics"]
        if not isinstance(metrics, Mapping):
            raise ParetoGovernanceError("Pareto observation metrics must be an object")
        try:
            visibility = ScoreVisibility(str(value["visibility"]))
        except ValueError as exc:
            raise HoldoutLeakageError(
                "unknown score visibility cannot enter Pareto governance"
            ) from exc
        return cls(
            candidate_id=str(value["candidate_id"]),
            campaign_round=value["campaign_round"],
            metrics=metrics,
            visibility=visibility,
        )


@dataclass(frozen=True, slots=True)
class ParetoRoundObservation:
    campaign_round: int
    observations: tuple[ParetoObservation, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.campaign_round, int)
            or isinstance(self.campaign_round, bool)
            or self.campaign_round < 0
        ):
            raise ParetoGovernanceError("campaign_round must be a non-negative integer")
        observations = tuple(self.observations)
        ids = [item.candidate_id for item in observations]
        if len(ids) != len(set(ids)):
            raise ParetoGovernanceError(
                "round contains duplicate candidate observations"
            )
        for item in observations:
            if not isinstance(item, ParetoObservation):
                raise ParetoGovernanceError("round contains an untyped observation")
            if item.campaign_round != self.campaign_round:
                raise ParetoGovernanceError(
                    "candidate observation campaign_round differs from round"
                )
        object.__setattr__(self, "observations", observations)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_round": self.campaign_round,
            "observations": [item.to_dict() for item in self.observations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParetoRoundObservation":
        expected = {"campaign_round", "observations"}
        if set(value) != expected:
            raise ParetoGovernanceError(
                f"Pareto round fields must be exactly {sorted(expected)}"
            )
        raw = value["observations"]
        if not isinstance(raw, list):
            raise ParetoGovernanceError("round observations must be a list")
        if any(not isinstance(item, Mapping) for item in raw):
            raise ParetoGovernanceError("round observation records must be objects")
        return cls(
            campaign_round=value["campaign_round"],
            observations=tuple(ParetoObservation.from_dict(item) for item in raw),
        )


def epsilon_dominates(
    left: ParetoObservation,
    right: ParetoObservation,
    objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
) -> bool:
    """Return whether ``left`` is no-worse and meaningfully better than ``right``.

    A difference inside epsilon is treated as practically equivalent.  At least
    one objective must improve by more than its epsilon; candidate names and
    any scalar score never enter this decision.
    """

    if not isinstance(left, ParetoObservation) or not isinstance(
        right, ParetoObservation
    ):
        raise ParetoGovernanceError(
            "epsilon dominance requires typed local Pareto observations"
        )
    specs = _validated_specs(objective_specs)
    no_worse = True
    meaningfully_better = False
    for spec in specs:
        left_value = left.value(spec.name)
        right_value = right.value(spec.name)
        if spec.direction is ObjectiveDirection.MAXIMIZE:
            no_worse = no_worse and left_value >= right_value - spec.epsilon
            meaningfully_better = (
                meaningfully_better or left_value > right_value + spec.epsilon
            )
        else:
            no_worse = no_worse and left_value <= right_value + spec.epsilon
            meaningfully_better = (
                meaningfully_better or left_value < right_value - spec.epsilon
            )
        if not no_worse:
            return False
    return meaningfully_better


def epsilon_equivalent(
    left: ParetoObservation,
    right: ParetoObservation,
    objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
) -> bool:
    if not isinstance(left, ParetoObservation) or not isinstance(
        right, ParetoObservation
    ):
        raise ParetoGovernanceError(
            "epsilon equivalence requires typed local Pareto observations"
        )
    specs = _validated_specs(objective_specs)
    return all(
        abs(left.value(spec.name) - right.value(spec.name)) <= spec.epsilon
        for spec in specs
    )


def pareto_frontier(
    observations: Iterable[ParetoObservation],
    objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
) -> tuple[ParetoObservation, ...]:
    specs = _validated_specs(objective_specs)
    items = tuple(observations)
    for item in items:
        if not isinstance(item, ParetoObservation):
            raise ParetoGovernanceError(
                "frontier input contains an untyped observation"
            )
    ids = [item.candidate_id for item in items]
    if len(ids) != len(set(ids)):
        raise ParetoGovernanceError("frontier input contains duplicate candidate IDs")
    frontier = [
        item
        for index, item in enumerate(items)
        if not any(
            epsilon_dominates(other, item, specs)
            for other_index, other in enumerate(items)
            if other_index != index
        )
    ]
    return tuple(
        sorted(frontier, key=lambda item: (item.campaign_round, item.candidate_id))
    )


@dataclass(frozen=True, slots=True)
class FrontierSnapshot:
    scope: str
    through_round: int
    members: tuple[ParetoObservation, ...]
    objective_spec_hash: str

    def __post_init__(self) -> None:
        if self.scope not in {"round", "campaign"}:
            raise ParetoGovernanceError("frontier scope must be round or campaign")
        if (
            not isinstance(self.through_round, int)
            or isinstance(self.through_round, bool)
            or self.through_round < 0
        ):
            raise ParetoGovernanceError(
                "frontier through_round must be a non-negative integer"
            )
        _assert_sha256(self.objective_spec_hash, "objective_spec_hash")
        members = tuple(self.members)
        if any(not isinstance(item, ParetoObservation) for item in members):
            raise ParetoGovernanceError("frontier contains an untyped observation")
        ids = [item.candidate_id for item in members]
        if len(ids) != len(set(ids)):
            raise ParetoGovernanceError("frontier contains duplicate candidate IDs")
        object.__setattr__(self, "members", members)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.members)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "through_round": self.through_round,
            "objective_spec_hash": self.objective_spec_hash,
            "members": [item.to_dict() for item in self.members],
        }


def build_round_frontier(
    round_observation: ParetoRoundObservation,
    objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
) -> FrontierSnapshot:
    if not isinstance(round_observation, ParetoRoundObservation):
        raise ParetoGovernanceError("round frontier requires a typed Pareto round")
    specs = _validated_specs(objective_specs)
    return FrontierSnapshot(
        scope="round",
        through_round=round_observation.campaign_round,
        members=pareto_frontier(round_observation.observations, specs),
        objective_spec_hash=objective_specs_hash(specs),
    )


def build_campaign_frontier(
    observations: (
        Iterable[ParetoObservation | ParetoRoundObservation] | "ParetoObservationLedger"
    ),
    objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
) -> FrontierSnapshot:
    specs = _validated_specs(objective_specs)
    if isinstance(observations, ParetoObservationLedger):
        items = observations.observations
        through_round = observations.last_campaign_round
    else:
        raw_items = tuple(observations)
        flattened: list[ParetoObservation] = []
        raw_rounds: list[int] = []
        for item in raw_items:
            if isinstance(item, ParetoObservation):
                flattened.append(item)
                raw_rounds.append(item.campaign_round)
            elif isinstance(item, ParetoRoundObservation):
                flattened.extend(item.observations)
                raw_rounds.append(item.campaign_round)
            else:
                raise ParetoGovernanceError(
                    "campaign frontier input contains an untyped observation"
                )
        items = tuple(flattened)
        through_round = max(raw_rounds, default=0)
    return FrontierSnapshot(
        scope="campaign",
        through_round=through_round if through_round is not None else 0,
        members=pareto_frontier(items, specs),
        objective_spec_hash=objective_specs_hash(specs),
    )


def frontier_progressed(
    previous: FrontierSnapshot | None,
    current: FrontierSnapshot,
    objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
) -> bool:
    """Detect a new non-dominated trade-off beyond preregistered epsilons."""

    specs = _validated_specs(objective_specs)
    expected_hash = objective_specs_hash(specs)
    if current.objective_spec_hash != expected_hash or (
        previous is not None and previous.objective_spec_hash != expected_hash
    ):
        raise ParetoGovernanceError("frontier objective specification hash mismatch")
    if previous is None or not previous.members:
        return bool(current.members)
    previous_ids = set(previous.candidate_ids)
    for candidate in current.members:
        if candidate.candidate_id in previous_ids:
            continue
        if any(epsilon_equivalent(candidate, old, specs) for old in previous.members):
            continue
        if not any(
            epsilon_dominates(old, candidate, specs) for old in previous.members
        ):
            return True
    return False


@dataclass(frozen=True, slots=True)
class ShortlistRejection:
    candidate_id: str
    reason: str
    correlated_with: str | None = None
    absolute_correlation: float | None = None

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            raise ParetoGovernanceError("shortlist rejection candidate_id is empty")
        if self.reason not in {
            "correlation_limit",
            "missing_correlation",
            "capacity",
        }:
            raise ParetoGovernanceError("unknown shortlist rejection reason")
        if self.absolute_correlation is not None:
            absolute = _finite_number(self.absolute_correlation, "absolute_correlation")
            if not 0 <= absolute <= 1:
                raise ParetoGovernanceError(
                    "absolute_correlation must be between zero and one"
                )
            object.__setattr__(self, "absolute_correlation", absolute)
        object.__setattr__(self, "candidate_id", candidate_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "correlated_with": self.correlated_with,
            "absolute_correlation": self.absolute_correlation,
        }


@dataclass(frozen=True, slots=True)
class DiverseShortlist:
    selected: tuple[ParetoObservation, ...]
    rejected: tuple[ShortlistRejection, ...]
    max_abs_correlation: float
    capacity: int

    def __post_init__(self) -> None:
        selected = tuple(self.selected)
        rejected = tuple(self.rejected)
        if any(not isinstance(item, ParetoObservation) for item in selected):
            raise ParetoGovernanceError("shortlist contains an untyped observation")
        if any(not isinstance(item, ShortlistRejection) for item in rejected):
            raise ParetoGovernanceError("shortlist contains an untyped rejection")
        if (
            not isinstance(self.capacity, int)
            or isinstance(self.capacity, bool)
            or self.capacity < 0
            or len(selected) > self.capacity
        ):
            raise ParetoGovernanceError("invalid shortlist capacity")
        limit = _finite_number(self.max_abs_correlation, "max_abs_correlation")
        if not 0 <= limit <= 1:
            raise ParetoGovernanceError(
                "max_abs_correlation must be between zero and one"
            )
        selected_ids = [item.candidate_id for item in selected]
        rejected_ids = [item.candidate_id for item in rejected]
        if len(selected_ids) != len(set(selected_ids)) or len(rejected_ids) != len(
            set(rejected_ids)
        ):
            raise ParetoGovernanceError("shortlist IDs must be unique")
        if set(selected_ids) & set(rejected_ids):
            raise ParetoGovernanceError(
                "candidate cannot be both selected and rejected"
            )
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "rejected", rejected)
        object.__setattr__(self, "max_abs_correlation", limit)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [item.to_dict() for item in self.selected],
            "rejected": [item.to_dict() for item in self.rejected],
            "max_abs_correlation": self.max_abs_correlation,
            "capacity": self.capacity,
        }


def select_diverse_shortlist(
    observations: Iterable[ParetoObservation] | FrontierSnapshot,
    correlations: Any,
    *,
    capacity: int,
    max_abs_correlation: float,
    objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
) -> DiverseShortlist:
    """Select a correlation-constrained shortlist without a weighted scalar.

    Objective-specific best queues are visited round-robin.  Missing pairwise
    correlation is fail-closed for the later candidate and is reported rather
    than silently treated as zero.
    """

    specs = _validated_specs(objective_specs)
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ParetoGovernanceError("shortlist capacity must be a non-negative integer")
    correlation_limit = _finite_number(max_abs_correlation, "max_abs_correlation")
    if not 0 <= correlation_limit <= 1:
        raise ParetoGovernanceError("max_abs_correlation must be between zero and one")
    if isinstance(observations, FrontierSnapshot):
        if observations.objective_spec_hash != objective_specs_hash(specs):
            raise ParetoGovernanceError(
                "shortlist frontier objective specification hash mismatch"
            )
        source = observations.members
    else:
        source = tuple(observations)
    frontier = pareto_frontier(source, specs)
    if capacity == 0 or not frontier:
        return DiverseShortlist(
            selected=(),
            rejected=tuple(
                ShortlistRejection(item.candidate_id, "capacity") for item in frontier
            ),
            max_abs_correlation=correlation_limit,
            capacity=capacity,
        )

    queues: dict[ParetoObjective, tuple[ParetoObservation, ...]] = {}
    for spec in specs:
        reverse = spec.direction is ObjectiveDirection.MAXIMIZE
        queues[spec.name] = tuple(
            sorted(
                frontier,
                key=lambda item: (
                    -item.value(spec.name) if reverse else item.value(spec.name),
                    item.candidate_id,
                ),
            )
        )

    remaining = {item.candidate_id: item for item in frontier}
    selected: list[ParetoObservation] = []
    rejected: list[ShortlistRejection] = []
    while remaining and len(selected) < capacity:
        considered_in_pass = False
        for spec in specs:
            candidate = next(
                (item for item in queues[spec.name] if item.candidate_id in remaining),
                None,
            )
            if candidate is None:
                continue
            considered_in_pass = True
            del remaining[candidate.candidate_id]
            rejection = _correlation_rejection(
                candidate,
                selected,
                correlations,
                correlation_limit,
            )
            if rejection is None:
                selected.append(candidate)
                if len(selected) >= capacity:
                    break
            else:
                rejected.append(rejection)
        if not considered_in_pass:  # pragma: no cover - defensive invariant.
            raise ParetoGovernanceError("shortlist queue made no progress")
    rejected.extend(
        ShortlistRejection(candidate_id, "capacity")
        for candidate_id in sorted(remaining)
    )
    return DiverseShortlist(
        selected=tuple(selected),
        rejected=tuple(rejected),
        max_abs_correlation=correlation_limit,
        capacity=capacity,
    )


def _correlation_rejection(
    candidate: ParetoObservation,
    selected: Sequence[ParetoObservation],
    correlations: Any,
    limit: float,
) -> ShortlistRejection | None:
    for existing in selected:
        correlation = _lookup_correlation(
            correlations, candidate.candidate_id, existing.candidate_id
        )
        if correlation is None:
            return ShortlistRejection(
                candidate.candidate_id,
                "missing_correlation",
                correlated_with=existing.candidate_id,
            )
        absolute = abs(correlation)
        if absolute > limit:
            return ShortlistRejection(
                candidate.candidate_id,
                "correlation_limit",
                correlated_with=existing.candidate_id,
                absolute_correlation=absolute,
            )
    return None


def _lookup_correlation(correlations: Any, left: str, right: str) -> float | None:
    if left == right:
        return 1.0
    raw_values: list[Any] = []
    if hasattr(correlations, "loc"):
        for first, second in ((left, right), (right, left)):
            try:
                raw_values.append(correlations.loc[first, second])
            except (KeyError, IndexError, TypeError):
                pass
    elif isinstance(correlations, Mapping):
        for first, second in ((left, right), (right, left)):
            if (first, second) in correlations:
                raw_values.append(correlations[(first, second)])
                continue
            row = correlations.get(first)
            if isinstance(row, Mapping) and second in row:
                raw_values.append(row[second])
    if not raw_values:
        return None
    values: list[float] = []
    for raw in raw_values:
        if raw is None:
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ParetoGovernanceError(
                f"correlation {left}/{right} must be numeric or missing"
            ) from exc
        if math.isnan(number):
            continue
        if not math.isfinite(number):
            raise ParetoGovernanceError(
                f"correlation {left}/{right} must be finite when present"
            )
        if abs(number) > 1.0 + 1e-12:
            raise ParetoGovernanceError(
                f"correlation {left}/{right} is outside [-1, 1]"
            )
        values.append(max(-1.0, min(1.0, number)))
    if not values:
        return None
    if max(values) - min(values) > 1e-10:
        raise ParetoGovernanceError(
            f"correlation matrix is asymmetric for {left}/{right}"
        )
    return values[0]


@dataclass(frozen=True, slots=True)
class ParetoLedgerEntry:
    sequence: int
    round_observation: ParetoRoundObservation
    previous_hash: str
    entry_hash: str

    @classmethod
    def create(
        cls,
        sequence: int,
        round_observation: ParetoRoundObservation,
        previous_hash: str,
    ) -> "ParetoLedgerEntry":
        payload = {
            "sequence": sequence,
            "round_observation": round_observation.to_dict(),
            "previous_hash": previous_hash,
        }
        return cls(
            sequence=sequence,
            round_observation=round_observation,
            previous_hash=previous_hash,
            entry_hash=hash_json(payload),
        )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 0
        ):
            raise ParetoGovernanceError(
                "ledger sequence must be a non-negative integer"
            )
        if not isinstance(self.round_observation, ParetoRoundObservation):
            raise ParetoGovernanceError("ledger entry requires a typed Pareto round")
        _assert_sha256(self.previous_hash, "previous_hash")
        _assert_sha256(self.entry_hash, "entry_hash")
        expected = hash_json(
            {
                "sequence": self.sequence,
                "round_observation": self.round_observation.to_dict(),
                "previous_hash": self.previous_hash,
            }
        )
        if self.entry_hash != expected:
            raise ParetoGovernanceError("Pareto ledger entry hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "round_observation": self.round_observation.to_dict(),
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParetoLedgerEntry":
        expected = {"sequence", "round_observation", "previous_hash", "entry_hash"}
        if set(value) != expected:
            raise ParetoGovernanceError(
                f"Pareto ledger entry fields must be exactly {sorted(expected)}"
            )
        round_payload = value["round_observation"]
        if not isinstance(round_payload, Mapping):
            raise ParetoGovernanceError("ledger round_observation must be an object")
        return cls(
            sequence=value["sequence"],
            round_observation=ParetoRoundObservation.from_dict(round_payload),
            previous_hash=str(value["previous_hash"]),
            entry_hash=str(value["entry_hash"]),
        )


class ParetoObservationLedger:
    """Append-only, hash-chained local research observations by campaign round."""

    def __init__(self, entries: Sequence[ParetoLedgerEntry] = ()) -> None:
        self._entries: list[ParetoLedgerEntry] = []
        self._candidate_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, ParetoLedgerEntry):
                raise ParetoGovernanceError("Pareto ledger contains an untyped entry")
            self._append_verified(entry)

    @property
    def entries(self) -> tuple[ParetoLedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def rounds(self) -> tuple[ParetoRoundObservation, ...]:
        return tuple(item.round_observation for item in self._entries)

    @property
    def observations(self) -> tuple[ParetoObservation, ...]:
        return tuple(
            observation
            for entry in self._entries
            for observation in entry.round_observation.observations
        )

    @property
    def last_campaign_round(self) -> int | None:
        return (
            self._entries[-1].round_observation.campaign_round
            if self._entries
            else None
        )

    @property
    def content_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else _GENESIS_HASH

    def append(self, observation: ParetoRoundObservation) -> ParetoLedgerEntry:
        previous_hash = self.content_hash
        entry = ParetoLedgerEntry.create(len(self._entries), observation, previous_hash)
        self._append_verified(entry)
        return entry

    def _append_verified(self, entry: ParetoLedgerEntry) -> None:
        if entry.sequence != len(self._entries):
            raise ParetoGovernanceError("Pareto ledger sequence is not append-only")
        if entry.previous_hash != self.content_hash:
            raise ParetoGovernanceError("Pareto ledger hash chain is broken")
        campaign_round = entry.round_observation.campaign_round
        if (
            self.last_campaign_round is not None
            and campaign_round <= self.last_campaign_round
        ):
            raise ParetoGovernanceError(
                "Pareto ledger rounds must be strictly increasing"
            )
        new_ids = {item.candidate_id for item in entry.round_observation.observations}
        duplicates = sorted(new_ids & self._candidate_ids)
        if duplicates:
            raise ParetoGovernanceError(
                f"candidate observations are append-only and cannot be replaced: {duplicates}"
            )
        self._entries.append(entry)
        self._candidate_ids.update(new_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PARETO_LEDGER_SCHEMA,
            "entries": [item.to_dict() for item in self._entries],
            "ledger_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParetoObservationLedger":
        expected = {"schema_version", "entries", "ledger_hash"}
        if set(value) != expected:
            raise ParetoGovernanceError(
                f"Pareto ledger fields must be exactly {sorted(expected)}"
            )
        if value["schema_version"] != PARETO_LEDGER_SCHEMA:
            raise ParetoGovernanceError("unsupported Pareto ledger schema")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list):
            raise ParetoGovernanceError("Pareto ledger entries must be a list")
        if any(not isinstance(item, Mapping) for item in raw_entries):
            raise ParetoGovernanceError("Pareto ledger entry records must be objects")
        ledger = cls(tuple(ParetoLedgerEntry.from_dict(item) for item in raw_entries))
        if ledger.content_hash != value["ledger_hash"]:
            raise ParetoGovernanceError("Pareto ledger terminal hash mismatch")
        return ledger


@dataclass(frozen=True, slots=True)
class ParetoStoppingConfig:
    patience_rounds: int = 2

    def __post_init__(self) -> None:
        if (
            not isinstance(self.patience_rounds, int)
            or isinstance(self.patience_rounds, bool)
            or self.patience_rounds <= 0
        ):
            raise ParetoGovernanceError("patience_rounds must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {"patience_rounds": self.patience_rounds}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParetoStoppingConfig":
        if set(value) != {"patience_rounds"}:
            raise ParetoGovernanceError(
                "Pareto stopping config fields must be exactly patience_rounds"
            )
        return cls(patience_rounds=value["patience_rounds"])


@dataclass(frozen=True, slots=True)
class ParetoStoppingDecision:
    should_stop: bool
    reason: StopReason
    detail: str
    frontier_progressed: bool
    stalled_rounds: int
    frontier: FrontierSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.should_stop, bool) or not isinstance(
            self.frontier_progressed, bool
        ):
            raise ParetoGovernanceError("stopping flags must be booleans")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ParetoGovernanceError("stopping detail must not be empty")
        try:
            reason = StopReason(self.reason)
        except (TypeError, ValueError) as exc:
            raise ParetoGovernanceError("unknown Pareto stopping reason") from exc
        allowed_terminal = {
            StopReason.CANDIDATE_BUDGET,
            StopReason.PROVIDER_CALL_BUDGET,
            StopReason.EVALUATION_BUDGET,
            StopReason.GENERATION_BUDGET,
            StopReason.WALL_TIME_BUDGET,
            StopReason.PATIENCE_EXHAUSTED,
        }
        if self.should_stop and reason not in allowed_terminal:
            raise ParetoGovernanceError(
                "Pareto stopping accepts only hard-budget or frontier-patience reasons"
            )
        if not self.should_stop and reason is not StopReason.CONTINUE:
            raise ParetoGovernanceError(
                "a non-terminal Pareto decision must use continue"
            )
        if (
            not isinstance(self.stalled_rounds, int)
            or isinstance(self.stalled_rounds, bool)
            or self.stalled_rounds < 0
        ):
            raise ParetoGovernanceError("stalled_rounds cannot be negative")
        if not isinstance(self.frontier, FrontierSnapshot):
            raise ParetoGovernanceError("decision requires a typed frontier snapshot")
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "reason": self.reason.value,
            "detail": self.detail,
            "frontier_progressed": self.frontier_progressed,
            "stalled_rounds": self.stalled_rounds,
            "frontier": self.frontier.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ParetoCheckpoint:
    objective_spec_hash: str
    ledger_hash: str
    round_count: int
    observation_count: int
    last_campaign_round: int | None
    frontier_candidate_ids: tuple[str, ...]
    frontier_hash: str
    stalled_rounds: int
    patience_rounds: int
    stop_reason: StopReason

    def __post_init__(self) -> None:
        _assert_sha256(self.objective_spec_hash, "objective_spec_hash")
        _assert_sha256(self.ledger_hash, "ledger_hash")
        _assert_sha256(self.frontier_hash, "frontier_hash")
        for name in ("round_count", "observation_count", "stalled_rounds"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ParetoGovernanceError(f"checkpoint {name} cannot be negative")
        if (
            not isinstance(self.patience_rounds, int)
            or isinstance(self.patience_rounds, bool)
            or self.patience_rounds <= 0
        ):
            raise ParetoGovernanceError("checkpoint patience_rounds must be positive")
        if self.last_campaign_round is not None and (
            not isinstance(self.last_campaign_round, int)
            or isinstance(self.last_campaign_round, bool)
            or self.last_campaign_round < 0
        ):
            raise ParetoGovernanceError(
                "checkpoint last_campaign_round must be a non-negative integer"
            )
        ids = tuple(str(item) for item in self.frontier_candidate_ids)
        if any(not item.strip() for item in ids):
            raise ParetoGovernanceError("checkpoint frontier IDs must not be empty")
        if len(ids) != len(set(ids)):
            raise ParetoGovernanceError("checkpoint frontier IDs must be unique")
        if len(ids) > self.observation_count:
            raise ParetoGovernanceError(
                "checkpoint frontier count exceeds observation count"
            )
        if (self.round_count == 0) != (self.last_campaign_round is None):
            raise ParetoGovernanceError(
                "checkpoint round_count and last_campaign_round are inconsistent"
            )
        object.__setattr__(self, "frontier_candidate_ids", ids)
        try:
            reason = StopReason(self.stop_reason)
        except (TypeError, ValueError) as exc:
            raise ParetoGovernanceError("unknown checkpoint stop reason") from exc
        hard_reasons = {
            StopReason.CANDIDATE_BUDGET,
            StopReason.PROVIDER_CALL_BUDGET,
            StopReason.EVALUATION_BUDGET,
            StopReason.GENERATION_BUDGET,
            StopReason.WALL_TIME_BUDGET,
        }
        if reason is StopReason.MANUAL or reason not in {
            StopReason.CONTINUE,
            StopReason.PATIENCE_EXHAUSTED,
            *hard_reasons,
        }:
            raise ParetoGovernanceError(
                "checkpoint stop reason must be frontier patience or hard budget"
            )
        if (
            reason is StopReason.CONTINUE
            and self.stalled_rounds >= self.patience_rounds
        ):
            raise ParetoGovernanceError("continue checkpoint exceeds Pareto patience")
        if (
            reason is StopReason.PATIENCE_EXHAUSTED
            and self.stalled_rounds < self.patience_rounds
        ):
            raise ParetoGovernanceError(
                "patience checkpoint has insufficient stalled rounds"
            )
        object.__setattr__(self, "stop_reason", reason)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PARETO_CHECKPOINT_SCHEMA,
            "objective_spec_hash": self.objective_spec_hash,
            "ledger_hash": self.ledger_hash,
            "round_count": self.round_count,
            "observation_count": self.observation_count,
            "last_campaign_round": self.last_campaign_round,
            "frontier_candidate_ids": list(self.frontier_candidate_ids),
            "frontier_hash": self.frontier_hash,
            "stalled_rounds": self.stalled_rounds,
            "patience_rounds": self.patience_rounds,
            "stop_reason": self.stop_reason.value,
        }

    @property
    def checkpoint_hash(self) -> str:
        return hash_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "checkpoint_hash": self.checkpoint_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParetoCheckpoint":
        expected = {
            "schema_version",
            "objective_spec_hash",
            "ledger_hash",
            "round_count",
            "observation_count",
            "last_campaign_round",
            "frontier_candidate_ids",
            "frontier_hash",
            "stalled_rounds",
            "patience_rounds",
            "stop_reason",
            "checkpoint_hash",
        }
        if set(value) != expected:
            raise ParetoGovernanceError(
                f"Pareto checkpoint fields must be exactly {sorted(expected)}"
            )
        if value["schema_version"] != PARETO_CHECKPOINT_SCHEMA:
            raise ParetoGovernanceError("unsupported Pareto checkpoint schema")
        raw_ids = value["frontier_candidate_ids"]
        if not isinstance(raw_ids, list):
            raise ParetoGovernanceError("checkpoint frontier IDs must be a list")
        checkpoint = cls(
            objective_spec_hash=str(value["objective_spec_hash"]),
            ledger_hash=str(value["ledger_hash"]),
            round_count=value["round_count"],
            observation_count=value["observation_count"],
            last_campaign_round=value["last_campaign_round"],
            frontier_candidate_ids=tuple(str(item) for item in raw_ids),
            frontier_hash=str(value["frontier_hash"]),
            stalled_rounds=value["stalled_rounds"],
            patience_rounds=value["patience_rounds"],
            stop_reason=value["stop_reason"],
        )
        if checkpoint.checkpoint_hash != value["checkpoint_hash"]:
            raise ParetoGovernanceError("Pareto checkpoint hash mismatch")
        return checkpoint

    def verify_against(
        self,
        ledger: ParetoObservationLedger,
        objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
    ) -> None:
        specs = _validated_specs(objective_specs)
        frontier, derived_stalled = _replay_frontier_state(ledger, specs)
        expected = {
            "objective_spec_hash": objective_specs_hash(specs),
            "ledger_hash": ledger.content_hash,
            "round_count": len(ledger.entries),
            "observation_count": len(ledger.observations),
            "last_campaign_round": ledger.last_campaign_round,
            "frontier_candidate_ids": frontier.candidate_ids,
            "frontier_hash": frontier.content_hash,
            "stalled_rounds": derived_stalled,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ParetoGovernanceError(f"Pareto checkpoint {name} mismatch")


def _replay_frontier_state(
    ledger: ParetoObservationLedger,
    objective_specs: Sequence[ParetoObjectiveSpec],
) -> tuple[FrontierSnapshot, int]:
    replay = ParetoObservationLedger()
    previous: FrontierSnapshot | None = None
    stalled = 0
    for round_observation in ledger.rounds:
        replay.append(round_observation)
        current = build_campaign_frontier(replay, objective_specs)
        if frontier_progressed(previous, current, objective_specs):
            stalled = 0
        else:
            stalled += 1
        previous = current
    return (
        previous or build_campaign_frontier(replay, objective_specs),
        stalled,
    )


class ParetoStopController:
    """Stop only on Pareto-frontier patience or preregistered hard budgets."""

    def __init__(
        self,
        config: ParetoStoppingConfig,
        objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
    ) -> None:
        if not isinstance(config, ParetoStoppingConfig):
            raise ParetoGovernanceError(
                "Pareto controller requires ParetoStoppingConfig"
            )
        self.config = config
        self.objective_specs = _validated_specs(objective_specs)
        self._ledger = ParetoObservationLedger()
        self._frontier: FrontierSnapshot | None = None
        self._stalled_rounds = 0
        self._decision: ParetoStoppingDecision | None = None

    @property
    def ledger(self) -> ParetoObservationLedger:
        # Return a verified copy so callers cannot mutate controller state.
        return ParetoObservationLedger.from_dict(self._ledger.to_dict())

    @property
    def frontier(self) -> FrontierSnapshot | None:
        return self._frontier

    @property
    def decision(self) -> ParetoStoppingDecision | None:
        return self._decision

    def observe(
        self,
        campaign_round: int,
        observations: Sequence[ParetoObservation],
        budget: BudgetController,
    ) -> ParetoStoppingDecision:
        return self.observe_round(
            ParetoRoundObservation(campaign_round, tuple(observations)), budget
        )

    def observe_round(
        self,
        round_observation: ParetoRoundObservation,
        budget: BudgetController,
    ) -> ParetoStoppingDecision:
        if self._decision is not None and self._decision.should_stop:
            raise ParetoGovernanceError("Pareto controller is already stopped")
        if not isinstance(round_observation, ParetoRoundObservation):
            raise ParetoGovernanceError("stopping requires a typed Pareto round")

        previous = self._frontier
        self._ledger.append(round_observation)
        current = build_campaign_frontier(self._ledger, self.objective_specs)
        progressed = frontier_progressed(previous, current, self.objective_specs)
        self._frontier = current
        if progressed:
            self._stalled_rounds = 0
        else:
            self._stalled_rounds += 1

        budget_reason = budget.first_exhausted_reason()
        if budget_reason is not StopReason.CONTINUE:
            decision = ParetoStoppingDecision(
                True,
                budget_reason,
                "a preregistered hard budget was exhausted",
                progressed,
                self._stalled_rounds,
                current,
            )
        elif self._stalled_rounds >= self.config.patience_rounds:
            decision = ParetoStoppingDecision(
                True,
                StopReason.PATIENCE_EXHAUSTED,
                f"no Pareto-frontier progress for {self._stalled_rounds} rounds",
                progressed,
                self._stalled_rounds,
                current,
            )
        else:
            detail = (
                "local Pareto frontier progressed"
                if progressed
                else "within preregistered Pareto-frontier patience"
            )
            decision = ParetoStoppingDecision(
                False,
                StopReason.CONTINUE,
                detail,
                progressed,
                self._stalled_rounds,
                current,
            )
        self._decision = decision
        return decision

    def checkpoint(self) -> ParetoCheckpoint:
        frontier = self._frontier or build_campaign_frontier(
            self._ledger, self.objective_specs
        )
        return ParetoCheckpoint(
            objective_spec_hash=objective_specs_hash(self.objective_specs),
            ledger_hash=self._ledger.content_hash,
            round_count=len(self._ledger.entries),
            observation_count=len(self._ledger.observations),
            last_campaign_round=self._ledger.last_campaign_round,
            frontier_candidate_ids=frontier.candidate_ids,
            frontier_hash=frontier.content_hash,
            stalled_rounds=self._stalled_rounds,
            patience_rounds=self.config.patience_rounds,
            stop_reason=(
                self._decision.reason
                if self._decision is not None and self._decision.should_stop
                else StopReason.CONTINUE
            ),
        )

    @classmethod
    def from_checkpoint(
        cls,
        config: ParetoStoppingConfig,
        checkpoint: ParetoCheckpoint,
        ledger: ParetoObservationLedger,
        objective_specs: Sequence[ParetoObjectiveSpec] = DEFAULT_PARETO_OBJECTIVES,
    ) -> "ParetoStopController":
        specs = _validated_specs(objective_specs)
        checkpoint.verify_against(ledger, specs)
        if checkpoint.patience_rounds != config.patience_rounds:
            raise ParetoGovernanceError("checkpoint patience configuration mismatch")
        controller = cls(config, specs)
        controller._ledger = ParetoObservationLedger.from_dict(ledger.to_dict())
        controller._frontier = build_campaign_frontier(controller._ledger, specs)
        controller._stalled_rounds = checkpoint.stalled_rounds
        if checkpoint.stop_reason is not StopReason.CONTINUE:
            controller._decision = ParetoStoppingDecision(
                True,
                checkpoint.stop_reason,
                "restored terminal Pareto checkpoint",
                False,
                checkpoint.stalled_rounds,
                controller._frontier,
            )
        return controller


def _assert_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ParetoGovernanceError(f"{name} must be a lowercase SHA-256 digest")
