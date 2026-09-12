from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from factor_production.v5.artifacts.hashing import canonical_json_bytes, hash_bytes, hash_file, hash_json
from factor_production.v5.domain.enums import ScoreVisibility
from factor_production.v5.domain.models import CandidateSpecV5
from factor_production.v5.scorers.base import (
    CandidateScorer,
    HoldoutIsolationError,
    ScoreBatch,
    ScoreObservation,
)


DROP_REQUEST_SCHEMA = "external-score-request/v1"
DROP_RESULT_SCHEMA = "external-score-result/v1"


class FileDropError(ValueError):
    pass


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True, slots=True)
class DropRequest:
    batch_id: str
    request_hash: str
    file_hash: str
    path: Path
    candidate_hashes: Mapping[str, str]


class FileDropScorer(CandidateScorer):
    """One-way boundary for teacher/official holdout evaluation.

    Exported requests contain frozen candidate hashes.  Imported results remain
    EXTERNAL_HOLDOUT ScoreBatches, whose conversion to proposal feedback is
    rejected by :meth:`ScoreBatch.to_local_feedback`.  This class intentionally
    has no inline scoring implementation.
    """

    visibility = ScoreVisibility.EXTERNAL_HOLDOUT

    def __init__(self, *, name: str = "external_file_drop") -> None:
        self.name = name

    def score(self, candidates: Sequence[CandidateSpecV5]) -> ScoreBatch:
        raise HoldoutIsolationError(
            "external scoring is not callable from the mining loop; use export_request/import_results"
        )

    def export_request(
        self,
        candidates: Sequence[CandidateSpecV5],
        path: str | Path,
        *,
        protocol_hash: str,
    ) -> DropRequest:
        if not candidates:
            raise FileDropError("cannot export an empty external score request")
        ids = [candidate.candidate_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise FileDropError("external request contains duplicate candidate IDs")
        for candidate in candidates:
            if candidate.protocol_hash != protocol_hash:
                raise FileDropError(
                    f"candidate {candidate.candidate_id} belongs to a different protocol"
                )
        core = {
            "schema_version": DROP_REQUEST_SCHEMA,
            "protocol_hash": protocol_hash,
            "scorer": self.name,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "spec_hash": candidate.content_hash,
                    "spec": candidate.to_dict(),
                }
                for candidate in candidates
            ],
        }
        request_hash = hash_json(core)
        batch_id = f"holdout-{request_hash[:20]}"
        payload = {**core, "batch_id": batch_id, "request_hash": request_hash}
        destination = Path(path)
        data = canonical_json_bytes(payload) + b"\n"
        _atomic_write(destination, data)
        file_hash = hash_bytes(data)
        _atomic_write(
            destination.with_suffix(destination.suffix + ".sha256"),
            f"{file_hash}  {destination.name}\n".encode("utf-8"),
        )
        return DropRequest(
            batch_id=batch_id,
            request_hash=request_hash,
            file_hash=file_hash,
            path=destination,
            candidate_hashes={candidate.candidate_id: candidate.content_hash for candidate in candidates},
        )

    def import_results(
        self,
        result_path: str | Path,
        request_path: str | Path,
        *,
        require_complete: bool = True,
    ) -> ScoreBatch:
        request = self._load_request(Path(request_path))
        try:
            result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileDropError(f"cannot load external result: {exc}") from exc
        if not isinstance(result, dict):
            raise FileDropError("external result root must be an object")
        allowed = {"schema_version", "batch_id", "request_hash", "scorer_id", "results"}
        if set(result) != allowed:
            raise FileDropError(
                f"external result fields must be exactly {sorted(allowed)}; got {sorted(result)}"
            )
        if result["schema_version"] != DROP_RESULT_SCHEMA:
            raise FileDropError(f"unsupported external result schema: {result['schema_version']!r}")
        if result["batch_id"] != request["batch_id"]:
            raise FileDropError("external result batch_id does not match request")
        if result["request_hash"] != request["request_hash"]:
            raise FileDropError("external result request_hash does not match request")
        scorer_id = result["scorer_id"]
        if not isinstance(scorer_id, str) or not scorer_id.strip():
            raise FileDropError("scorer_id must be a non-empty string")
        expected = {
            item["candidate_id"]: (item["spec_hash"], item["spec"]["generation"])
            for item in request["candidates"]
        }
        observations: list[ScoreObservation] = []
        seen: set[str] = set()
        if not isinstance(result["results"], list):
            raise FileDropError("external results must be a list")
        for index, item in enumerate(result["results"]):
            if not isinstance(item, dict) or set(item) != {"candidate_id", "spec_hash", "metrics"}:
                raise FileDropError(f"external result item {index} has invalid fields")
            candidate_id = item["candidate_id"]
            if candidate_id in seen:
                raise FileDropError(f"duplicate external result for {candidate_id}")
            if candidate_id not in expected:
                raise FileDropError(f"unknown external result candidate: {candidate_id}")
            expected_hash, generation = expected[candidate_id]
            if item["spec_hash"] != expected_hash:
                raise FileDropError(f"spec hash mismatch for external result {candidate_id}")
            if not isinstance(item["metrics"], dict):
                raise FileDropError(f"metrics must be an object for {candidate_id}")
            observations.append(
                ScoreObservation(
                    candidate_id=candidate_id,
                    generation=generation,
                    metrics=item["metrics"],
                    scorer=scorer_id,
                    visibility=self.visibility,
                )
            )
            seen.add(candidate_id)
        if require_complete and seen != set(expected):
            missing = sorted(set(expected) - seen)
            raise FileDropError(f"external result is incomplete; missing {missing}")
        return ScoreBatch(
            batch_id=request["batch_id"],
            scorer=scorer_id,
            visibility=self.visibility,
            observations=tuple(observations),
        )

    def _load_request(self, path: Path) -> dict[str, Any]:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.is_file():
            raise FileDropError(f"request hash sidecar is missing: {sidecar}")
        expected_file_hash = sidecar.read_text(encoding="utf-8").split()[0]
        actual_file_hash = hash_file(path)
        if actual_file_hash != expected_file_hash:
            raise FileDropError(
                f"external request file hash mismatch: expected {expected_file_hash}, got {actual_file_hash}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        allowed = {
            "schema_version",
            "protocol_hash",
            "scorer",
            "candidates",
            "batch_id",
            "request_hash",
        }
        if not isinstance(payload, dict) or set(payload) != allowed:
            raise FileDropError("external request structure is invalid")
        if payload["schema_version"] != DROP_REQUEST_SCHEMA:
            raise FileDropError("external request schema is invalid")
        core = {key: payload[key] for key in ("schema_version", "protocol_hash", "scorer", "candidates")}
        actual_request_hash = hash_json(core)
        if payload["request_hash"] != actual_request_hash:
            raise FileDropError("external request semantic hash mismatch")
        if payload["batch_id"] != f"holdout-{actual_request_hash[:20]}":
            raise FileDropError("external request batch ID is not derived from request hash")
        for item in payload["candidates"]:
            if set(item) != {"candidate_id", "spec_hash", "spec"}:
                raise FileDropError("external request candidate record is invalid")
            spec = CandidateSpecV5.from_dict(item["spec"])
            if item["candidate_id"] != spec.candidate_id or item["spec_hash"] != spec.content_hash:
                raise FileDropError(f"external request candidate hash mismatch: {item['candidate_id']}")
        return payload

