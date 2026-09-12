from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_alpha_mining.mining.artifacts import ArtifactManifest, ArtifactStore
from llm_alpha_mining.mining.cli import (
    initialize_workspace,
    verify_workspace,
    workspace_status,
)
from llm_alpha_mining.mining.domain.enums import (
    CandidateState,
    RunState,
    ScoreVisibility,
)
from llm_alpha_mining.mining.domain.models import CandidateSpecError, CandidateSpec
from llm_alpha_mining.mining.orchestration.budget import (
    BudgetController,
    BudgetExceeded,
    BudgetLimits,
)
from llm_alpha_mining.mining.orchestration.repository import SQLiteRepository
from llm_alpha_mining.mining.orchestration.state import InvalidStateTransition
from llm_alpha_mining.mining.orchestration.stopping import (
    HoldoutLeakageError,
    ObjectiveObservation,
    StopController,
    StoppingConfig,
)
from llm_alpha_mining.mining.protocol import (
    ProtocolError,
    MiningProtocol,
    default_protocol,
)
from llm_alpha_mining.mining.providers import (
    MockProvider,
    ProposalContext,
    ReplayProvider,
)
from llm_alpha_mining.mining.scorers import (
    FileDropScorer,
    HoldoutIsolationError,
    MockScorer,
)
from llm_alpha_mining.mining.scorers.file_drop import DROP_RESULT_SCHEMA, FileDropError


def _candidate(
    protocol_hash: str,
    *,
    candidate_id: str = "g0_low_vol",
    generation: int = 0,
    parent_ids: tuple[str, ...] = (),
) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=candidate_id,
        hypothesis="short-horizon uncertainty is followed by reversal",
        expression="CsRank(TsStd(Returns, 10))",
        direction=-1,
        frequency="daily",
        generation=generation,
        family="uncertainty",
        required_fields=("Returns",),
        protocol_hash=protocol_hash,
        provider="mock",
        parent_ids=parent_ids,
        aggregation={"interday": "last"},
        parameters={"window": 10},
        tags=("low_vol",),
    )


def _limits(**overrides: int) -> BudgetLimits:
    values = {
        "max_candidates": 10,
        "max_generations": 3,
        "max_provider_calls": 3,
        "max_evaluations": 10,
        "max_wall_seconds": 60,
    }
    values.update(overrides)
    return BudgetLimits(**values)


def test_candidate_spec_is_single_strict_immutable_and_content_hashed() -> None:
    protocol = default_protocol("candidate-contract")
    first = _candidate(protocol.content_hash)
    second = CandidateSpec.from_dict(
        {key: value for key, value in reversed(list(first.to_dict().items()))}
    )

    assert first == second
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64
    with pytest.raises(TypeError):
        first.parameters["window"] = 20  # type: ignore[index]
    with pytest.raises(CandidateSpecError, match="unknown CandidateSpec fields"):
        CandidateSpec.from_dict({**first.to_dict(), "local_score": 1.0})
    with pytest.raises(CandidateSpecError, match="external holdout field is forbidden"):
        CandidateSpec.from_dict(
            {**first.to_dict(), "parameters": {"teacher": {"score": 99}}}
        )


def test_protocol_rejects_teacher_feedback_into_search() -> None:
    value = default_protocol("isolation").to_dict()
    value["external_evaluation"]["may_influence_search"] = True

    with pytest.raises(ProtocolError, match="must not influence"):
        MiningProtocol.from_dict(value)
    with pytest.raises(HoldoutLeakageError, match="external holdout scores"):
        ObjectiveObservation(
            generation=0,
            value=0.5,
            metric="teacher_score",
            visibility=ScoreVisibility.EXTERNAL_HOLDOUT,
        )


def test_sqlite_repository_enforces_protocol_and_state_machine(tmp_path: Path) -> None:
    protocol = default_protocol("repository")
    candidate = _candidate(protocol.content_hash)
    with SQLiteRepository(tmp_path / "control.sqlite3") as repository:
        repository.create_run("run", protocol.content_hash)
        repository.transition_run("run", RunState.INITIALIZED, reason="protocol frozen")
        record = repository.add_candidate("run", candidate)
        assert record.state is CandidateState.DRAFT
        assert repository.verify_integrity("run") == ()

        record = repository.transition_candidate(
            "run",
            candidate.candidate_id,
            CandidateState.PROPOSED,
            reason="provider returned",
        )
        assert record.state is CandidateState.PROPOSED
        with pytest.raises(InvalidStateTransition, match="illegal state transition"):
            repository.transition_candidate(
                "run",
                candidate.candidate_id,
                CandidateState.SELECTED,
                reason="skip all gates",
            )


def test_artifact_manifest_detects_post_freeze_tampering(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    record = store.put_bytes(
        "protocol", b'{"frozen":true}\n', media_type="application/json"
    )
    manifest = ArtifactManifest("test-manifest", [record])
    manifest_path = manifest.write(tmp_path / "manifest.json")

    loaded = ArtifactManifest.load(manifest_path)
    assert loaded.verify(tmp_path).ok
    (tmp_path / record.location).write_bytes(b"tampered")
    verification = loaded.verify(tmp_path)
    assert not verification.ok
    assert any("hash_mismatch" in error for error in verification.errors)


def test_mock_and_replay_providers_are_deterministic(tmp_path: Path) -> None:
    protocol = default_protocol("providers")
    first = _candidate(protocol.content_hash)
    second = _candidate(protocol.content_hash, candidate_id="g0_second")
    context = ProposalContext(
        run_id="run",
        protocol_hash=protocol.content_hash,
        generation=0,
        request_count=1,
    )

    mock = MockProvider((first, second))
    assert mock.propose(context).candidates == (first,)
    assert mock.propose(context).candidates == (second,)

    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        "\n".join(
            json.dumps({"sha256": item.content_hash, "spec": item.to_dict()})
            for item in (first, second)
        )
        + "\n",
        encoding="utf-8",
    )
    replay = ReplayProvider(replay_path)
    assert replay.propose(context).candidates == (first,)
    assert replay.propose(context).candidates == (second,)


def test_file_drop_results_are_hash_bound_and_cannot_become_feedback(
    tmp_path: Path,
) -> None:
    protocol = default_protocol("file-drop")
    candidate = _candidate(protocol.content_hash)
    scorer = FileDropScorer()
    request_path = tmp_path / "request.json"
    request = scorer.export_request(
        (candidate,), request_path, protocol_hash=protocol.content_hash
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": DROP_RESULT_SCHEMA,
                "batch_id": request.batch_id,
                "request_hash": request.request_hash,
                "scorer_id": "teacher_private_system",
                "results": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "spec_hash": candidate.content_hash,
                        "metrics": {"private_score": 0.75},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = scorer.import_results(result_path, request_path)
    assert result.visibility is ScoreVisibility.EXTERNAL_HOLDOUT
    with pytest.raises(HoldoutIsolationError, match="cannot be converted"):
        result.to_local_feedback("private_score")

    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["results"][0]["spec_hash"] = "0" * 64
    result_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(FileDropError, match="spec hash mismatch"):
        scorer.import_results(result_path, request_path)


def test_local_mock_scores_can_drive_feedback_but_external_scorer_cannot_inline() -> (
    None
):
    protocol = default_protocol("scores")
    candidate = _candidate(protocol.content_hash)
    local = MockScorer({candidate.candidate_id: {"robust_score": 0.12}}).score(
        (candidate,)
    )

    feedback = local.to_local_feedback("robust_score")
    assert feedback[0].value == pytest.approx(0.12)
    assert feedback[0].visibility is ScoreVisibility.LOCAL_RESEARCH
    with pytest.raises(HoldoutIsolationError, match="not callable"):
        FileDropScorer().score((candidate,))


def test_budget_is_reserved_fail_closed_and_stopping_uses_local_metric() -> None:
    budget = BudgetController(_limits(max_candidates=1))
    budget.reserve(candidates=1)
    with pytest.raises(BudgetExceeded):
        budget.reserve(candidates=1)

    stopping_budget = BudgetController(_limits())
    controller = StopController(
        StoppingConfig(patience_generations=2, min_improvement=0.01)
    )
    first = controller.observe(
        ObjectiveObservation(0, 0.10, "local_score"), stopping_budget
    )
    second = controller.observe(
        ObjectiveObservation(1, 0.105, "local_score"), stopping_budget
    )
    third = controller.observe(
        ObjectiveObservation(2, 0.106, "local_score"), stopping_budget
    )
    assert not first.should_stop
    assert not second.should_stop
    assert third.should_stop
    assert third.reason.value == "patience_exhausted"


def test_cli_init_status_verify_and_tamper_detection(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    initialized = initialize_workspace(workspace, run_id="cli-run")
    status = workspace_status(workspace)
    verified = verify_workspace(workspace)

    assert initialized["run_state"] == "initialized"
    assert status["run_id"] == "cli-run"
    assert verified["ok"]
    protocol = json.loads((workspace / "protocol.json").read_text(encoding="utf-8"))
    protocol["random_seed"] += 1
    (workspace / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    verified = verify_workspace(workspace)
    assert not verified["ok"]
    assert any("protocol_hash" in error for error in verified["errors"])
