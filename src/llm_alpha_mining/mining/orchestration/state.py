from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TypeVar

from llm_alpha_mining.mining.domain.enums import CandidateState, RunState


class InvalidStateTransition(ValueError):
    pass


CANDIDATE_TRANSITIONS: Mapping[CandidateState, frozenset[CandidateState]] = {
    CandidateState.DRAFT: frozenset({CandidateState.PROPOSED, CandidateState.FAILED}),
    CandidateState.PROPOSED: frozenset(
        {CandidateState.VALIDATED, CandidateState.REJECTED, CandidateState.FAILED}
    ),
    CandidateState.VALIDATED: frozenset(
        {CandidateState.EVALUATING, CandidateState.REJECTED, CandidateState.FAILED}
    ),
    CandidateState.EVALUATING: frozenset(
        {CandidateState.EVALUATED, CandidateState.FAILED}
    ),
    CandidateState.EVALUATED: frozenset(
        {CandidateState.ELIGIBLE, CandidateState.REJECTED, CandidateState.FAILED}
    ),
    CandidateState.ELIGIBLE: frozenset(
        {CandidateState.SELECTED, CandidateState.REJECTED, CandidateState.FAILED}
    ),
    CandidateState.SELECTED: frozenset(
        {CandidateState.EXPORTED, CandidateState.REJECTED, CandidateState.FAILED}
    ),
    CandidateState.REJECTED: frozenset(),
    CandidateState.EXPORTED: frozenset(),
    CandidateState.FAILED: frozenset(),
}


RUN_TRANSITIONS: Mapping[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.INITIALIZED, RunState.FAILED}),
    RunState.INITIALIZED: frozenset({RunState.RUNNING, RunState.FAILED}),
    RunState.RUNNING: frozenset(
        {RunState.STOPPING, RunState.COMPLETED, RunState.FAILED}
    ),
    RunState.STOPPING: frozenset({RunState.COMPLETED, RunState.FAILED}),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
}


StateT = TypeVar("StateT", bound=Enum)


def assert_transition(
    current: StateT,
    target: StateT,
    transitions: Mapping[StateT, frozenset[StateT]],
) -> None:
    if target not in transitions[current]:
        raise InvalidStateTransition(
            f"illegal state transition: {current.value} -> {target.value}"
        )


def assert_candidate_transition(
    current: CandidateState, target: CandidateState
) -> None:
    assert_transition(current, target, CANDIDATE_TRANSITIONS)


def assert_run_transition(current: RunState, target: RunState) -> None:
    assert_transition(current, target, RUN_TRANSITIONS)
