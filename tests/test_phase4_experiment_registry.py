from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from alpha_research.experiments import (
    ApprovalGrant,
    ApprovalRevocation,
    DataPartition,
    ExperimentProfile,
    ExperimentReceipt,
    ExperimentRegistry,
    ExperimentRegistryConflict,
    ExperimentSpec,
    ResourceBudget,
    RetryPolicy,
)


NOW = datetime(2026, 7, 19, 8, 0, tzinfo=timezone.utc)


def _hash(character: str) -> str:
    return character * 64


def _budget(*, llm: bool = False) -> ResourceBudget:
    return ResourceBudget(
        maximum_attempts=20,
        maximum_wall_seconds=3600.0,
        maximum_cpu_seconds=7200.0,
        maximum_peak_memory_bytes=2_000_000_000,
        maximum_disk_write_bytes=5_000_000_000,
        maximum_parallel_tasks=2,
        per_stage_timeout_seconds=600.0,
        maximum_llm_calls=10 if llm else 0,
        maximum_llm_tokens=20_000 if llm else 0,
        maximum_llm_cost_microusd=1_000_000 if llm else 0,
    )


def _retry() -> RetryPolicy:
    return RetryPolicy(
        maximum_attempts_per_stage=3,
        initial_backoff_seconds=1.0,
        maximum_backoff_seconds=30.0,
        backoff_multiplier=2.0,
        jitter_fraction=0.2,
        retryable_failure_codes=("provider_429", "worker_lost"),
    )


def _spec(**changes) -> ExperimentSpec:
    values = {
        "experiment_id": "phase4-registry-test",
        "version": "1",
        "profile": ExperimentProfile.RESEARCH,
        "data_partitions": {
            DataPartition.DISCOVERY: _hash("a"),
            DataPartition.TRAIN: _hash("b"),
            DataPartition.VALIDATION: _hash("c"),
            DataPartition.TEST: _hash("d"),
        },
        "agent_visible_partitions": (
            DataPartition.DISCOVERY,
            DataPartition.TRAIN,
        ),
        "factor_spec_hashes": (_hash("e"),),
        "label_spec_hash": _hash("f"),
        "validation_spec_hash": _hash("1"),
        "evaluation_spec_hash": _hash("2"),
        "model_spec_hash": _hash("3"),
        "portfolio_spec_hash": _hash("4"),
        "cost_model_hash": _hash("5"),
        "robustness_spec_hash": _hash("6"),
        "code_snapshot_hash": _hash("7"),
        "environment_hash": _hash("8"),
        "llm_policy_hash": None,
        "llm_transport_hash": None,
        "stages": (
            "data_quality",
            "factor_generation",
            "label_building",
            "validation_split",
            "model_training",
            "factor_evaluation",
            "portfolio_construction",
            "backtest",
            "robustness",
            "report",
        ),
        "random_seed": 17,
        "resource_budget": _budget(),
        "retry_policy": _retry(),
    }
    values.update(changes)
    return ExperimentSpec(**values)


def _component_type(role: str) -> str:
    return role.split(":", 1)[0]


def _register_components(registry: ExperimentRegistry, spec: ExperimentSpec) -> None:
    for role, digest in spec.component_bindings():
        registry.register_component(
            digest,
            component_type=_component_type(role),
            descriptor={"content_hash": digest, "name": role},
            registered_at=NOW,
        )


def _component_batch(
    spec: ExperimentSpec,
) -> tuple[tuple[str, str, dict[str, object]], ...]:
    return tuple(
        (
            digest,
            _component_type(role),
            {"content_hash": digest, "name": role},
        )
        for role, digest in spec.component_bindings()
    )


def test_experiment_spec_blocks_agent_holdout_and_missing_stage_contracts() -> None:
    with pytest.raises(ValueError, match="protected partitions:test"):
        _spec(agent_visible_partitions=(DataPartition.TEST,))
    with pytest.raises(ValueError, match="stage lacks its contract:model_training"):
        _spec(model_spec_hash=None)
    with pytest.raises(ValueError, match="requires a protected test/holdout"):
        _spec(
            profile=ExperimentProfile.PRODUCTION_CANDIDATE,
            data_partitions={DataPartition.DISCOVERY: _hash("a")},
            agent_visible_partitions=(DataPartition.DISCOVERY,),
        )
    with pytest.raises(ValueError, match="LLM budget requires"):
        _spec(resource_budget=_budget(llm=True))
    with pytest.raises(ValueError, match="bound LLM transport"):
        _spec(llm_policy_hash=_hash("9"), resource_budget=_budget(llm=True))
    with pytest.raises(ValueError, match="must bind distinct content hashes"):
        _spec(
            data_partitions={
                DataPartition.DISCOVERY: _hash("a"),
                DataPartition.TRAIN: _hash("a"),
            },
            agent_visible_partitions=(
                DataPartition.DISCOVERY,
                DataPartition.TRAIN,
            ),
        )

    llm = _spec(
        llm_policy_hash=_hash("9"),
        llm_transport_hash=_hash("8"),
        resource_budget=_budget(llm=True),
    )
    assert llm.agent_visible_partitions == (
        DataPartition.DISCOVERY,
        DataPartition.TRAIN,
    )
    assert _retry().delay_seconds(2, jitter_unit=0.5) == pytest.approx(4.0)


def test_registry_is_immutable_and_binds_every_component(tmp_path) -> None:
    spec = _spec()
    with ExperimentRegistry(tmp_path / "experiments.sqlite3") as registry:
        with pytest.raises(ExperimentRegistryConflict, match="not registered"):
            registry.register_experiment(spec, registered_at=NOW)
        _register_components(registry, spec)
        first = registry.register_experiment(spec, registered_at=NOW)
        second = registry.register_experiment(spec, registered_at=NOW)
        assert first == second
        assert first.status == "registered"
        with pytest.raises(ExperimentRegistryConflict, match="immutable component"):
            registry.register_component(
                _hash("a"),
                component_type="data",
                descriptor={"content_hash": _hash("a"), "name": "changed"},
                registered_at=NOW,
            )
        with pytest.raises(ExperimentRegistryConflict, match="another immutable spec"):
            registry.register_experiment(
                replace(spec, random_seed=18), registered_at=NOW
            )
        running = registry.set_status(
            spec.content_hash,
            "running",
            reason_code="worker_dispatched",
            at=NOW + timedelta(seconds=1),
        )
        assert running.status == "running"
        with pytest.raises(ValueError, match="only created by sealing"):
            registry.set_status(
                spec.content_hash,
                "completed",
                reason_code="bad_path",
                at=NOW + timedelta(seconds=2),
            )
        assert registry.verify_integrity() == ()


def test_bulk_component_and_experiment_registration_is_exact_and_idempotent(
    tmp_path,
) -> None:
    spec = _spec()
    batch = _component_batch(spec)
    with ExperimentRegistry(tmp_path / "bulk.sqlite3") as registry:
        first = registry.register_components_and_experiment(
            spec,
            components=batch,
            registered_at=NOW,
        )
        second = registry.register_components_and_experiment(
            spec,
            components=batch,
            registered_at=NOW,
        )
        assert first == second
        assert first.experiment_spec_hash == spec.content_hash
        assert registry.connection.execute(
            "SELECT COUNT(*) FROM components"
        ).fetchone()[0] == len(spec.component_bindings())
        assert registry.connection.execute(
            "SELECT COUNT(*) FROM experiments"
        ).fetchone()[0] == 1

        with pytest.raises(ValueError, match="exactly cover"):
            registry.register_components_and_experiment(
                replace(spec, version="missing-component"),
                components=batch[:-1],
                registered_at=NOW,
            )


def test_bulk_component_conflict_rolls_back_every_new_row(tmp_path) -> None:
    spec = _spec()
    batch = _component_batch(spec)
    conflict = next(item for item in batch if item[0] == _hash("f"))
    with ExperimentRegistry(tmp_path / "bulk-rollback.sqlite3") as registry:
        registry.register_component(
            conflict[0],
            component_type=conflict[1],
            descriptor={"content_hash": conflict[0], "name": "conflict"},
            registered_at=NOW,
        )
        with pytest.raises(ExperimentRegistryConflict, match="immutable component"):
            registry.register_components_and_experiment(
                spec,
                components=batch,
                registered_at=NOW,
            )
        assert registry.connection.execute(
            "SELECT COUNT(*) FROM components"
        ).fetchone()[0] == 1
        assert registry.connection.execute(
            "SELECT COUNT(*) FROM experiments"
        ).fetchone()[0] == 0


def test_load_experiment_spec_reopens_with_complete_local_integrity(tmp_path) -> None:
    path = tmp_path / "experiments.sqlite3"
    spec = _spec()
    with ExperimentRegistry(path) as registry:
        _register_components(registry, spec)
        registry.register_experiment(spec, registered_at=NOW)
        assert registry.load_experiment_spec(spec.content_hash) == spec

    with ExperimentRegistry(path) as reopened:
        assert reopened.load_experiment_spec(spec.content_hash) == spec


@pytest.mark.parametrize(
    ("tamper_case", "message"),
    (
        ("noncanonical_spec_json", "spec is not canonical"),
        ("immutable_column", "immutable columns differ"),
        ("missing_component_row", "component bindings differ"),
        ("component_role", "component bindings differ"),
        ("component_hash", "component bindings differ"),
        ("component_type", "component type differs:model"),
    ),
)
def test_load_experiment_spec_fails_closed_after_persisted_tamper(
    tmp_path,
    tamper_case: str,
    message: str,
) -> None:
    path = tmp_path / f"{tamper_case}.sqlite3"
    spec = _spec()
    with ExperimentRegistry(path) as registry:
        _register_components(registry, spec)
        registry.register_experiment(spec, registered_at=NOW)
        assert registry.load_experiment_spec(spec.content_hash) == spec

    with sqlite3.connect(path) as connection:
        if tamper_case == "noncanonical_spec_json":
            connection.execute(
                """UPDATE experiments SET spec_json=spec_json || ' '
                   WHERE experiment_spec_hash=?""",
                (spec.content_hash,),
            )
        elif tamper_case == "immutable_column":
            connection.execute(
                """UPDATE experiments SET profile='production_candidate'
                   WHERE experiment_spec_hash=?""",
                (spec.content_hash,),
            )
        elif tamper_case == "missing_component_row":
            connection.execute(
                """DELETE FROM experiment_components
                   WHERE experiment_spec_hash=? AND role='model'""",
                (spec.content_hash,),
            )
        elif tamper_case == "component_role":
            connection.execute(
                """UPDATE experiment_components SET role='model:renamed'
                   WHERE experiment_spec_hash=? AND role='model'""",
                (spec.content_hash,),
            )
        elif tamper_case == "component_hash":
            connection.execute(
                """UPDATE experiment_components SET component_hash=?
                   WHERE experiment_spec_hash=? AND role='model'""",
                (spec.label_spec_hash, spec.content_hash),
            )
        elif tamper_case == "component_type":
            connection.execute(
                "UPDATE components SET component_type='factor' WHERE component_hash=?",
                (spec.model_spec_hash,),
            )
        else:  # pragma: no cover
            raise AssertionError(f"unknown tamper case:{tamper_case}")

    with ExperimentRegistry(path) as reopened:
        with pytest.raises(ExperimentRegistryConflict, match=message):
            reopened.load_experiment_spec(spec.content_hash)


def test_lineage_and_frozen_receipt_are_queryable_and_cycle_safe(tmp_path) -> None:
    spec = _spec()
    artifact_hash = _hash("9")
    with ExperimentRegistry(tmp_path / "experiments.sqlite3") as registry:
        _register_components(registry, spec)
        registry.register_experiment(spec, registered_at=NOW)
        registered = registry.register_artifact(
            spec.content_hash,
            artifact_hash=artifact_hash,
            logical_name="factor_metrics",
            kind="metrics",
            location="artifacts/metrics.json",
            media_type="application/json",
            size_bytes=123,
            descriptor={
                "artifact_hash": artifact_hash,
                "logical_name": "factor_metrics",
                "kind": "metrics",
                "location": "artifacts/metrics.json",
                "media_type": "application/json",
                "size_bytes": 123,
            },
            created_at=NOW + timedelta(seconds=1),
        )
        assert dict(
            registry.load_artifact_descriptor_by_logical_name(
                spec.content_hash,
                "factor_metrics",
            )
        ) == {
            "artifact_hash": artifact_hash,
            "logical_name": "factor_metrics",
            "kind": "metrics",
            "location": "artifacts/metrics.json",
            "media_type": "application/json",
            "size_bytes": 123,
        }
        assert registered.artifact_hash == artifact_hash
        with pytest.raises(KeyError, match="logical name is not registered"):
            registry.load_artifact_descriptor_by_logical_name(
                spec.content_hash,
                "missing_metrics",
            )
        parent = spec.factor_spec_hashes[0]
        edge = registry.add_lineage(
            spec.content_hash,
            child_hash=artifact_hash,
            parent_hash=parent,
            relationship="evaluated_from",
            recorded_at=NOW + timedelta(seconds=2),
        )
        assert edge.parent_hash == parent
        assert registry.ancestors(artifact_hash) == (edge,)
        with pytest.raises(ExperimentRegistryConflict, match="create a cycle"):
            registry.add_lineage(
                spec.content_hash,
                child_hash=parent,
                parent_hash=artifact_hash,
                relationship="bad_cycle",
                recorded_at=NOW + timedelta(seconds=3),
            )

        receipt = ExperimentReceipt(
            experiment_spec_hash=spec.content_hash,
            terminal_status="completed",
            component_bindings=dict(spec.component_bindings()),
            stage_result_hashes={"factor_evaluation": artifact_hash},
            artifact_hashes=(artifact_hash,),
            metrics_artifact_hash=artifact_hash,
            started_at=NOW.isoformat(),
            finished_at=(NOW + timedelta(seconds=4)).isoformat(),
            random_seed=spec.random_seed,
            code_snapshot_hash=spec.code_snapshot_hash,
            environment_hash=spec.environment_hash,
            production_ready=False,
        )
        receipt_hash = registry.seal_receipt(
            receipt, sealed_at=NOW + timedelta(seconds=5)
        )
        assert receipt_hash == receipt.content_hash
        assert registry.load_receipt(spec.content_hash) == receipt
        assert registry.get_experiment(spec.content_hash).status == "completed"
        with pytest.raises(ExperimentRegistryConflict, match="already terminal"):
            registry.seal_receipt(receipt, sealed_at=NOW + timedelta(seconds=6))
        assert registry.verify_integrity() == ()


def test_receipt_fails_closed_on_binding_or_production_claim(tmp_path) -> None:
    spec = _spec()
    artifact_hash = _hash("9")
    with pytest.raises(ValueError, match="requires human approval"):
        ExperimentReceipt(
            experiment_spec_hash=spec.content_hash,
            terminal_status="completed",
            component_bindings=dict(spec.component_bindings()),
            stage_result_hashes={},
            artifact_hashes=(),
            metrics_artifact_hash=None,
            started_at=NOW.isoformat(),
            finished_at=NOW.isoformat(),
            random_seed=spec.random_seed,
            code_snapshot_hash=spec.code_snapshot_hash,
            environment_hash=spec.environment_hash,
            production_ready=True,
        )
    with ExperimentRegistry(tmp_path / "experiments.sqlite3") as registry:
        _register_components(registry, spec)
        registry.register_experiment(spec, registered_at=NOW)
        receipt = ExperimentReceipt(
            experiment_spec_hash=spec.content_hash,
            terminal_status="failed",
            component_bindings={**dict(spec.component_bindings()), "label": _hash("0")},
            stage_result_hashes={},
            artifact_hashes=(),
            metrics_artifact_hash=None,
            started_at=NOW.isoformat(),
            finished_at=NOW.isoformat(),
            random_seed=spec.random_seed,
            code_snapshot_hash=spec.code_snapshot_hash,
            environment_hash=spec.environment_hash,
            production_ready=False,
        )
        with pytest.raises(
            ExperimentRegistryConflict, match="component bindings differ"
        ):
            registry.seal_receipt(receipt, sealed_at=NOW)
        with pytest.raises(ExperimentRegistryConflict, match="unknown artifacts"):
            unknown = replace(
                receipt,
                component_bindings=dict(spec.component_bindings()),
                artifact_hashes=(artifact_hash,),
            )
            registry.seal_receipt(unknown, sealed_at=NOW)


def test_production_receipt_requires_active_scoped_signed_approval(tmp_path) -> None:
    spec = _spec(profile=ExperimentProfile.PRODUCTION_CANDIDATE)
    metrics_hash = _hash("9")
    with ExperimentRegistry(tmp_path / "experiments.sqlite3") as registry:
        _register_components(registry, spec)
        registry.register_experiment(spec, registered_at=NOW)
        registry.register_artifact(
            spec.content_hash,
            artifact_hash=metrics_hash,
            logical_name="production_metrics",
            kind="metrics",
            location="artifacts/production_metrics.json",
            media_type="application/json",
            size_bytes=50,
            descriptor={
                "artifact_hash": metrics_hash,
                "logical_name": "production_metrics",
                "kind": "metrics",
                "location": "artifacts/production_metrics.json",
                "media_type": "application/json",
                "size_bytes": 50,
            },
            created_at=NOW,
        )
        unauthorized = ApprovalGrant(
            experiment_spec_hash=spec.content_hash,
            action="publish",
            scope="production_release",
            artifact_hash=metrics_hash,
            actor="intern",
            actor_role="research_intern",
            signature_hash=_hash("a"),
            nonce="approval-nonce-unauthorized",
            issued_at=NOW.isoformat(),
            expires_at=(NOW + timedelta(hours=1)).isoformat(),
        )
        registry.register_approval(unauthorized)
        invalid_receipt = ExperimentReceipt(
            experiment_spec_hash=spec.content_hash,
            terminal_status="completed",
            component_bindings=dict(spec.component_bindings()),
            stage_result_hashes={"report": metrics_hash},
            artifact_hashes=(metrics_hash,),
            metrics_artifact_hash=metrics_hash,
            started_at=NOW.isoformat(),
            finished_at=(NOW + timedelta(minutes=1)).isoformat(),
            random_seed=spec.random_seed,
            code_snapshot_hash=spec.code_snapshot_hash,
            environment_hash=spec.environment_hash,
            production_ready=True,
            approval_hash=unauthorized.content_hash,
        )
        with pytest.raises(ExperimentRegistryConflict, match="role is unauthorized"):
            registry.seal_receipt(invalid_receipt, sealed_at=NOW + timedelta(minutes=2))

        revoked = ApprovalGrant(
            experiment_spec_hash=spec.content_hash,
            action="publish",
            scope="production_release",
            artifact_hash=metrics_hash,
            actor="risk-owner",
            actor_role="risk_officer",
            signature_hash=_hash("b"),
            nonce="approval-nonce-revoked",
            issued_at=NOW.isoformat(),
            expires_at=(NOW + timedelta(hours=1)).isoformat(),
        )
        registry.register_approval(revoked)
        registry.revoke_approval(
            ApprovalRevocation(
                approval_hash=revoked.content_hash,
                revoked_at=(NOW + timedelta(minutes=1)).isoformat(),
                revoked_by="risk-owner",
                reason_code="evidence_changed",
            )
        )
        with pytest.raises(ExperimentRegistryConflict, match="revoked"):
            registry.require_approval(
                revoked.content_hash,
                experiment_spec_hash=spec.content_hash,
                action="publish",
                scope="production_release",
                artifact_hash=metrics_hash,
                at=NOW + timedelta(minutes=2),
                allowed_roles=frozenset({"risk_officer"}),
            )

        valid = replace(
            revoked,
            signature_hash=_hash("c"),
            nonce="approval-nonce-valid",
        )
        registry.register_approval(valid)
        final_receipt = replace(invalid_receipt, approval_hash=valid.content_hash)
        assert (
            registry.seal_receipt(final_receipt, sealed_at=NOW + timedelta(minutes=3))
            == final_receipt.content_hash
        )
        assert registry.load_receipt(spec.content_hash).production_ready is True
        assert registry.verify_integrity() == ()
