from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from factor_production.v5.artifacts import ArtifactManifest
from factor_production.v5.artifacts.hashing import hash_file, hash_json
from factor_production.v5.domain.enums import CandidateState
from factor_production.v5.orchestration.budget import BudgetController, BudgetUsage
from factor_production.v5.orchestration.campaign import CampaignConfig, CampaignEngine
from factor_production.v5.orchestration.hypotheses import HypothesisRegistry
from factor_production.v5.orchestration.repository import (
    CandidateRecord,
    EventRecord,
    SQLiteRepository,
)
from factor_production.v5.orchestration.stopping import StopController
from factor_production.v5.protocol import load_protocol
from factor_production.v5.providers.base import SanitizedFeedback
from factor_production.v5.providers.replay import ReplayProvider


CAMPAIGN_CHECKPOINT_SCHEMA = "campaign-checkpoint/v1"
CAMPAIGN_CHECKPOINT_EVENT = "campaign_checkpoint"
WORKSPACE_SCHEMA = "alpha-mining-workspace/v5"


class WorkspaceCampaignError(RuntimeError):
    pass


def _load_workspace_descriptor(root: Path) -> dict[str, str]:
    path = root / "workspace.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspaceCampaignError(f"cannot load workspace descriptor {path}: {exc}") from exc
    expected = {
        "schema_version",
        "run_id",
        "protocol_hash",
        "database",
        "protocol",
        "manifest",
        "descriptor_hash",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise WorkspaceCampaignError("workspace descriptor has an invalid schema")
    if value["schema_version"] != WORKSPACE_SCHEMA:
        raise WorkspaceCampaignError(f"unsupported workspace schema: {value['schema_version']!r}")
    core = {key: value[key] for key in expected - {"descriptor_hash"}}
    if value["descriptor_hash"] != hash_json(core):
        raise WorkspaceCampaignError("workspace descriptor content hash mismatch")
    for name in ("database", "protocol", "manifest"):
        location = Path(value[name])
        if location.is_absolute() or ".." in location.parts:
            raise WorkspaceCampaignError(f"unsafe workspace {name} path: {value[name]!r}")
    return value


def _verify_workspace(root: Path, descriptor: dict[str, str]) -> None:
    errors: list[str] = []
    try:
        protocol = load_protocol(root / descriptor["protocol"])
        if protocol.content_hash != descriptor["protocol_hash"]:
            errors.append("protocol_hash_mismatch")
    except Exception as exc:
        errors.append(f"protocol:{exc}")
    try:
        manifest = ArtifactManifest.load(root / descriptor["manifest"])
        verification = manifest.verify(root)
        errors.extend(verification.errors)
        if manifest.metadata.get("protocol_hash") != descriptor["protocol_hash"]:
            errors.append("manifest_protocol_hash_mismatch")
    except Exception as exc:
        errors.append(f"manifest:{exc}")
    try:
        with SQLiteRepository(root / descriptor["database"]) as repository:
            run = repository.get_run(descriptor["run_id"])
            if run.protocol_hash != descriptor["protocol_hash"]:
                errors.append("repository_protocol_hash_mismatch")
            errors.extend(repository.verify_integrity(descriptor["run_id"]))
    except Exception as exc:
        errors.append(f"repository:{exc}")
    if errors:
        raise WorkspaceCampaignError(f"workspace failed immutable verification: {errors}")


@dataclass(frozen=True, slots=True)
class CampaignCheckpoint:
    run_id: str
    protocol_hash: str
    next_campaign_round: int
    registry_hash: str
    budget_usage: BudgetUsage
    registration_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CAMPAIGN_CHECKPOINT_SCHEMA,
            "run_id": self.run_id,
            "protocol_hash": self.protocol_hash,
            "next_campaign_round": self.next_campaign_round,
            "registry_hash": self.registry_hash,
            "budget_usage": self.budget_usage.to_dict(),
            "registration_hash": self.registration_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CampaignCheckpoint":
        expected = {
            "schema_version",
            "run_id",
            "protocol_hash",
            "next_campaign_round",
            "registry_hash",
            "budget_usage",
            "registration_hash",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise WorkspaceCampaignError("campaign checkpoint has an invalid schema")
        if value["schema_version"] != CAMPAIGN_CHECKPOINT_SCHEMA:
            raise WorkspaceCampaignError(
                f"unsupported campaign checkpoint schema: {value['schema_version']!r}"
            )
        if not isinstance(value["run_id"], str) or not value["run_id"].strip():
            raise WorkspaceCampaignError("checkpoint run_id must be a non-empty string")
        for name in ("protocol_hash", "registry_hash", "registration_hash"):
            digest = value[name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise WorkspaceCampaignError(f"checkpoint {name} must be a SHA-256 digest")
        next_round = value["next_campaign_round"]
        if not isinstance(next_round, int) or isinstance(next_round, bool) or next_round < 0:
            raise WorkspaceCampaignError(
                "checkpoint next_campaign_round must be a non-negative integer"
            )
        usage_payload = value["budget_usage"]
        if not isinstance(usage_payload, dict):
            raise WorkspaceCampaignError("checkpoint budget_usage must be an object")
        return cls(
            run_id=value["run_id"],
            protocol_hash=value["protocol_hash"],
            next_campaign_round=next_round,
            registry_hash=value["registry_hash"],
            budget_usage=BudgetUsage.from_dict(usage_payload),
            registration_hash=value["registration_hash"],
        )


@dataclass(frozen=True, slots=True)
class ReplayRegistrationResult:
    workspace: str
    run_id: str
    campaign_round: int
    prior_candidate_count: int
    registered_candidate_count: int
    registered_candidate_hashes: tuple[str, ...]
    sanitized_feedback_hashes: tuple[str, ...]
    registry_hash: str
    checkpoint_event_id: int
    budget_usage: BudgetUsage
    idempotent_recovery: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "workspace": self.workspace,
            "run_id": self.run_id,
            "campaign_round": self.campaign_round,
            "prior_candidate_count": self.prior_candidate_count,
            "registered_candidate_count": self.registered_candidate_count,
            "registered_candidate_hashes": list(self.registered_candidate_hashes),
            "sanitized_feedback_hashes": list(self.sanitized_feedback_hashes),
            "registry_hash": self.registry_hash,
            "checkpoint_event_id": self.checkpoint_event_id,
            "budget_usage": self.budget_usage.to_dict(),
            "idempotent_recovery": self.idempotent_recovery,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceBranchBinding:
    source_workspace: str
    destination_workspace: str
    run_id: str
    protocol_hash: str
    source_registry_hash: str
    source_candidate_count: int


def validate_workspace_branch(
    source_workspace: str | Path,
    destination_workspace: str | Path,
) -> WorkspaceBranchBinding:
    """Prove a campaign branch shares provenance but not its mutable database."""

    source = Path(source_workspace).resolve()
    destination = Path(destination_workspace).resolve()
    if source == destination:
        raise WorkspaceCampaignError("source and destination workspace must differ")
    source_descriptor = _load_workspace_descriptor(source)
    destination_descriptor = _load_workspace_descriptor(destination)
    _verify_workspace(source, source_descriptor)
    _verify_workspace(destination, destination_descriptor)
    for name in ("run_id", "protocol_hash"):
        if source_descriptor[name] != destination_descriptor[name]:
            raise WorkspaceCampaignError(f"workspace branch {name} mismatch")
    source_database = source / source_descriptor["database"]
    destination_database = destination / destination_descriptor["database"]
    source_stat = source_database.stat()
    destination_stat = destination_database.stat()
    if (source_stat.st_dev, source_stat.st_ino) == (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        raise WorkspaceCampaignError(
            "workspace branch control database is hard-linked to its source"
        )
    with SQLiteRepository(source_database) as source_repository:
        source_records = source_repository.list_candidates(source_descriptor["run_id"])
        source_registry = _registry_from_records(
            source_descriptor["protocol_hash"],
            source_records,
        )
        source_hashes = {
            record.candidate_id: record.spec_hash for record in source_records
        }
    with SQLiteRepository(destination_database) as destination_repository:
        destination_hashes = {
            record.candidate_id: record.spec_hash
            for record in destination_repository.list_candidates(destination_descriptor["run_id"])
        }
    mismatched = {
        candidate_id: (spec_hash, destination_hashes.get(candidate_id))
        for candidate_id, spec_hash in source_hashes.items()
        if destination_hashes.get(candidate_id) != spec_hash
    }
    if mismatched:
        raise WorkspaceCampaignError(
            f"workspace branch does not preserve source candidate hashes: {mismatched}"
        )
    return WorkspaceBranchBinding(
        source_workspace=str(source),
        destination_workspace=str(destination),
        run_id=source_descriptor["run_id"],
        protocol_hash=source_descriptor["protocol_hash"],
        source_registry_hash=source_registry.content_hash,
        source_candidate_count=len(source_records),
    )


@dataclass(frozen=True, slots=True)
class SanitizedFeedbackBatch:
    campaign_id: str
    source_campaign_round: int
    source_snapshot_sha256: str
    items: tuple[SanitizedFeedback, ...]
    file_sha256: str


def load_sanitized_feedback_batch(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> SanitizedFeedbackBatch:
    """Load the frozen batch format and verify its file hash sidecar."""

    source = Path(path)
    if expected_sha256 is None:
        sidecar = source.with_suffix(".sha256")
        try:
            parts = sidecar.read_text(encoding="utf-8").strip().split()
        except OSError as exc:
            raise WorkspaceCampaignError(
                "sanitized feedback requires expected_sha256 or a .sha256 sidecar"
            ) from exc
        if not parts:
            raise WorkspaceCampaignError("sanitized feedback hash sidecar is empty")
        expected_sha256 = parts[0]
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise WorkspaceCampaignError("sanitized feedback expected hash is invalid")
    actual_sha256 = hash_file(source)
    if actual_sha256 != expected_sha256:
        raise WorkspaceCampaignError(
            f"sanitized feedback file hash mismatch: {expected_sha256}!={actual_sha256}"
        )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspaceCampaignError(f"cannot load sanitized feedback batch: {exc}") from exc
    expected = {
        "schema_version",
        "campaign_id",
        "source_campaign_round",
        "source_snapshot_sha256",
        "visibility",
        "contains_raw_metric_values",
        "contains_external_teacher_feedback",
        "items",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise WorkspaceCampaignError("sanitized feedback batch has an invalid schema")
    if value["schema_version"] != "sanitized-feedback-batch/v1":
        raise WorkspaceCampaignError("unsupported sanitized feedback batch schema")
    if value["visibility"] != "adaptive_local_research_only":
        raise WorkspaceCampaignError("sanitized feedback visibility is not local research")
    if value["contains_raw_metric_values"] is not False:
        raise WorkspaceCampaignError("sanitized feedback batch contains raw metric values")
    if value["contains_external_teacher_feedback"] is not False:
        raise WorkspaceCampaignError("sanitized feedback batch contains external feedback")
    source_round = value["source_campaign_round"]
    if not isinstance(source_round, int) or isinstance(source_round, bool) or source_round < 0:
        raise WorkspaceCampaignError("source_campaign_round must be non-negative")
    campaign_id = value["campaign_id"]
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise WorkspaceCampaignError("sanitized feedback campaign_id must not be empty")
    snapshot_hash = value["source_snapshot_sha256"]
    if (
        not isinstance(snapshot_hash, str)
        or len(snapshot_hash) != 64
        or any(character not in "0123456789abcdef" for character in snapshot_hash)
    ):
        raise WorkspaceCampaignError("source_snapshot_sha256 is invalid")
    raw_items = value["items"]
    if not isinstance(raw_items, list) or not raw_items:
        raise WorkspaceCampaignError("sanitized feedback items must be a non-empty list")
    try:
        items = tuple(SanitizedFeedback.from_dict(item) for item in raw_items)
    except Exception as exc:
        raise WorkspaceCampaignError(f"invalid sanitized feedback item: {exc}") from exc
    ids = [item.candidate_id for item in items]
    if len(ids) != len(set(ids)):
        raise WorkspaceCampaignError("sanitized feedback batch contains duplicate candidates")
    if any(item.campaign_round != source_round for item in items):
        raise WorkspaceCampaignError("feedback item campaign round differs from batch source round")
    return SanitizedFeedbackBatch(
        campaign_id=campaign_id.strip(),
        source_campaign_round=source_round,
        source_snapshot_sha256=snapshot_hash,
        items=items,
        file_sha256=actual_sha256,
    )


def load_sanitized_feedback_jsonl(path: str | Path) -> tuple[SanitizedFeedback, ...]:
    """Load only hash-wrapped, closed-vocabulary feedback records."""

    source = Path(path)
    feedback: list[SanitizedFeedback] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            if not isinstance(payload, dict) or set(payload) != {"sha256", "feedback"}:
                raise ValueError("record fields must be sha256 and feedback")
            if not isinstance(payload["feedback"], dict):
                raise ValueError("feedback root must be an object")
            item = SanitizedFeedback.from_dict(payload["feedback"])
            if payload["sha256"] != item.content_hash:
                raise ValueError(
                    f"feedback hash mismatch: expected {payload['sha256']}, got {item.content_hash}"
                )
            if item.candidate_id in seen_ids:
                raise ValueError(f"duplicate feedback candidate: {item.candidate_id}")
        except Exception as exc:
            raise WorkspaceCampaignError(
                f"invalid sanitized feedback at {source}:{line_number}: {exc}"
            ) from exc
        seen_ids.add(item.candidate_id)
        feedback.append(item)
    return tuple(feedback)


def _registry_from_records(
    protocol_hash: str,
    records: Iterable[CandidateRecord],
) -> HypothesisRegistry:
    registry = HypothesisRegistry(protocol_hash)
    registry.register_many(record.spec for record in records)
    return registry


def validate_sanitized_feedback_bindings(
    registry: HypothesisRegistry,
    campaign_round: int,
    feedback: Iterable[SanitizedFeedback],
) -> tuple[SanitizedFeedback, ...]:
    """Verify provenance without exposing raw evaluation metrics to providers."""

    items = tuple(feedback)
    if any(not isinstance(item, SanitizedFeedback) for item in items):
        raise WorkspaceCampaignError("feedback accepts SanitizedFeedback objects only")
    ids = [item.candidate_id for item in items]
    if len(ids) != len(set(ids)):
        raise WorkspaceCampaignError("feedback contains duplicate candidate IDs")
    for item in items:
        if item.campaign_round >= campaign_round:
            raise WorkspaceCampaignError("feedback must come from a prior campaign round")
        try:
            entry = registry.get(item.candidate_id)
        except KeyError as exc:
            raise WorkspaceCampaignError(
                f"feedback references an unknown candidate: {item.candidate_id}"
            ) from exc
        if item.candidate_hash != entry.spec_hash:
            raise WorkspaceCampaignError(
                f"feedback hash does not bind to candidate: {item.candidate_id}"
            )
    return items


def _latest_checkpoint(repository: SQLiteRepository, run_id: str) -> tuple[EventRecord, CampaignCheckpoint] | None:
    events = repository.list_events(run_id, event_type=CAMPAIGN_CHECKPOINT_EVENT)
    if not events:
        return None
    event = events[-1]
    checkpoint = CampaignCheckpoint.from_dict(event.payload)
    if checkpoint.run_id != run_id:
        raise WorkspaceCampaignError("checkpoint run_id mismatch")
    return event, checkpoint


def _infer_usage(records: Iterable[CandidateRecord]) -> BudgetUsage:
    items = tuple(records)
    rounds = {record.spec.campaign_round for record in items}
    evaluated_states = {
        CandidateState.EVALUATED,
        CandidateState.ELIGIBLE,
        CandidateState.SELECTED,
        CandidateState.EXPORTED,
    }
    return BudgetUsage(
        candidates=len(items),
        generations_started=len(rounds),
        provider_calls=len(rounds),
        evaluations=sum(record.state in evaluated_states for record in items),
        wall_seconds=0.0,
    )


def _resolved_usage(
    inferred: BudgetUsage,
    latest: tuple[EventRecord, CampaignCheckpoint] | None,
) -> BudgetUsage:
    if latest is None:
        return inferred
    _, checkpoint = latest
    previous = checkpoint.budget_usage
    if previous.candidates > inferred.candidates:
        raise WorkspaceCampaignError("checkpoint candidate usage exceeds repository ledger")
    return BudgetUsage(
        candidates=inferred.candidates,
        generations_started=max(previous.generations_started, inferred.generations_started),
        provider_calls=max(previous.provider_calls, inferred.provider_calls),
        evaluations=max(previous.evaluations, inferred.evaluations),
        wall_seconds=previous.wall_seconds,
    )


def _registration_hash(
    campaign_round: int,
    candidate_hashes: Iterable[str],
    feedback_hashes: Iterable[str],
) -> str:
    return hash_json(
        {
            "campaign_round": campaign_round,
            "candidate_hashes": sorted(candidate_hashes),
            "feedback_hashes": sorted(feedback_hashes),
        }
    )


def _record_checkpoint(
    repository: SQLiteRepository,
    *,
    run_id: str,
    protocol_hash: str,
    registry: HypothesisRegistry,
    next_campaign_round: int,
    budget_usage: BudgetUsage,
    registration_hash: str,
) -> tuple[EventRecord, CampaignCheckpoint]:
    checkpoint = CampaignCheckpoint(
        run_id=run_id,
        protocol_hash=protocol_hash,
        next_campaign_round=next_campaign_round,
        registry_hash=registry.content_hash,
        budget_usage=budget_usage,
        registration_hash=registration_hash,
    )
    event = repository.record_event(run_id, CAMPAIGN_CHECKPOINT_EVENT, checkpoint.to_dict())
    return event, checkpoint


def load_campaign_checkpoint(workspace: str | Path) -> CampaignCheckpoint | None:
    """Load and bind the latest persisted campaign budget/round checkpoint."""

    root = Path(workspace).resolve()
    descriptor = _load_workspace_descriptor(root)
    _verify_workspace(root, descriptor)
    with SQLiteRepository(root / descriptor["database"]) as repository:
        latest = _latest_checkpoint(repository, descriptor["run_id"])
        if latest is None:
            return None
        _, checkpoint = latest
        registry = _registry_from_records(
            descriptor["protocol_hash"],
            repository.list_candidates(descriptor["run_id"]),
        )
        if checkpoint.protocol_hash != descriptor["protocol_hash"]:
            raise WorkspaceCampaignError("checkpoint protocol hash mismatch")
        if checkpoint.registry_hash != registry.content_hash:
            raise WorkspaceCampaignError("checkpoint registry hash is stale or invalid")
        if checkpoint.next_campaign_round != registry.next_campaign_round:
            raise WorkspaceCampaignError("checkpoint next campaign round is inconsistent")
        if checkpoint.budget_usage.candidates != len(registry):
            raise WorkspaceCampaignError("checkpoint candidate budget is inconsistent")
        return checkpoint


def register_replay_generation(
    workspace: str | Path,
    replay_path: str | Path,
    *,
    feedback: Iterable[SanitizedFeedback],
    campaign_round: int = 1,
    expected_prior_count: int = 9,
    expected_new_count: int = 22,
    provider_name: str = "g1_replay_control_plane",
    source_workspace: str | Path | None = None,
) -> ReplayRegistrationResult:
    """Register one preflighted replay generation without running evaluation.

    The function is intentionally suitable for the actual research runner: it
    mutates only the copied workspace control database, is idempotent after a
    crash, and records both the round event and a budget/registry checkpoint.
    """

    if campaign_round <= 0:
        raise ValueError("replay campaign_round must be positive")
    if expected_prior_count < 0 or expected_new_count <= 0:
        raise ValueError("expected candidate counts are invalid")
    root = Path(workspace).resolve()
    if source_workspace is not None:
        validate_workspace_branch(source_workspace, root)
    descriptor = _load_workspace_descriptor(root)
    _verify_workspace(root, descriptor)
    protocol = load_protocol(root / descriptor["protocol"])
    if protocol.content_hash != descriptor["protocol_hash"]:
        raise WorkspaceCampaignError("workspace protocol hash mismatch")
    provider = ReplayProvider(replay_path, name=provider_name)
    proposed = provider.candidates
    if len(proposed) != expected_new_count:
        raise WorkspaceCampaignError(
            f"replay candidate count mismatch: {len(proposed)}!={expected_new_count}"
        )
    if any(not candidate.has_explicit_lineage for candidate in proposed):
        raise WorkspaceCampaignError("every replay proposal must declare explicit lineage")
    if any(candidate.campaign_round != campaign_round for candidate in proposed):
        raise WorkspaceCampaignError("replay proposal campaign_round mismatch")

    with SQLiteRepository(root / descriptor["database"]) as repository:
        run_id = descriptor["run_id"]
        records = repository.list_candidates(run_id)
        prior = [record for record in records if record.spec.campaign_round < campaign_round]
        current = [record for record in records if record.spec.campaign_round == campaign_round]
        future = [record for record in records if record.spec.campaign_round > campaign_round]
        if len(prior) != expected_prior_count:
            raise WorkspaceCampaignError(
                f"prior candidate count mismatch: {len(prior)}!={expected_prior_count}"
            )
        if future:
            raise WorkspaceCampaignError("repository already contains a later campaign round")

        base_registry = _registry_from_records(descriptor["protocol_hash"], prior)
        sanitized = validate_sanitized_feedback_bindings(
            base_registry,
            campaign_round,
            feedback,
        )
        feedback_hashes = tuple(item.content_hash for item in sanitized)
        replay_hashes = tuple(candidate.content_hash for candidate in proposed)
        registration_hash = _registration_hash(
            campaign_round,
            replay_hashes,
            feedback_hashes,
        )

        # Validate the whole proposed ledger before the first SQLite write.
        staged_registry = _registry_from_records(descriptor["protocol_hash"], prior)
        staged_registry.register_many(proposed)

        latest = _latest_checkpoint(repository, run_id)
        if current:
            existing = {record.candidate_id: record.spec_hash for record in current}
            expected = {candidate.candidate_id: candidate.content_hash for candidate in proposed}
            if existing != expected:
                raise WorkspaceCampaignError(
                    "campaign round is partially registered or differs from replay"
                )
            all_records = repository.list_candidates(run_id)
            full_registry = _registry_from_records(descriptor["protocol_hash"], all_records)
            inferred = _infer_usage(all_records)
            usage = _resolved_usage(inferred, latest)
            if (
                latest is not None
                and latest[1].registry_hash == full_registry.content_hash
                and latest[1].registration_hash == registration_hash
                and latest[1].budget_usage == usage
            ):
                checkpoint_event, checkpoint = latest
            else:
                checkpoint_event, checkpoint = _record_checkpoint(
                    repository,
                    run_id=run_id,
                    protocol_hash=descriptor["protocol_hash"],
                    registry=full_registry,
                    next_campaign_round=campaign_round + 1,
                    budget_usage=usage,
                    registration_hash=registration_hash,
                )
            integrity = repository.verify_integrity(run_id)
            if integrity:
                raise WorkspaceCampaignError(f"repository integrity failed: {integrity}")
            return ReplayRegistrationResult(
                workspace=str(root),
                run_id=run_id,
                campaign_round=campaign_round,
                prior_candidate_count=len(prior),
                registered_candidate_count=len(current),
                registered_candidate_hashes=tuple(sorted(existing.values())),
                sanitized_feedback_hashes=tuple(sorted(feedback_hashes)),
                registry_hash=checkpoint.registry_hash,
                checkpoint_event_id=checkpoint_event.event_id,
                budget_usage=checkpoint.budget_usage,
                idempotent_recovery=True,
            )

        inferred = _infer_usage(records)
        usage = _resolved_usage(inferred, latest)
        engine = CampaignEngine(
            run_id=run_id,
            protocol_hash=descriptor["protocol_hash"],
            provider=provider,
            repository=repository,
            budget=BudgetController(protocol.budget, usage),
            stopping=StopController(protocol.stopping),
            config=CampaignConfig(proposals_per_round=expected_new_count),
            registry=base_registry,
        )
        # Keep this explicit even though propose_round repeats it: no provider
        # call or candidate write occurs before feedback provenance is bound.
        engine.validate_feedback(campaign_round, sanitized)
        result = engine.propose_round(campaign_round, feedback=sanitized)
        if len(result.accepted) != expected_new_count or result.rejected:
            raise WorkspaceCampaignError(
                f"replay registration was not exact: accepted={len(result.accepted)}, "
                f"rejected={len(result.rejected)}"
            )
        checkpoint_event, checkpoint = _record_checkpoint(
            repository,
            run_id=run_id,
            protocol_hash=descriptor["protocol_hash"],
            registry=engine.registry,
            next_campaign_round=engine.next_campaign_round,
            budget_usage=engine.budget.usage,
            registration_hash=registration_hash,
        )
        integrity = repository.verify_integrity(run_id)
        if integrity:
            raise WorkspaceCampaignError(f"repository integrity failed: {integrity}")
        return ReplayRegistrationResult(
            workspace=str(root),
            run_id=run_id,
            campaign_round=campaign_round,
            prior_candidate_count=len(prior),
            registered_candidate_count=len(result.accepted),
            registered_candidate_hashes=tuple(
                sorted(candidate.content_hash for candidate in result.accepted)
            ),
            sanitized_feedback_hashes=tuple(sorted(feedback_hashes)),
            registry_hash=checkpoint.registry_hash,
            checkpoint_event_id=checkpoint_event.event_id,
            budget_usage=checkpoint.budget_usage,
            idempotent_recovery=False,
        )
