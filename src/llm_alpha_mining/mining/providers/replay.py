from __future__ import annotations

import json
from pathlib import Path

from llm_alpha_mining.mining.domain.models import CandidateSpec
from llm_alpha_mining.mining.providers.base import (
    CandidateProvider,
    ProposalBatch,
    ProposalContext,
)


class ReplayError(ValueError):
    pass


class ReplayProvider(CandidateProvider):
    """Replay immutable candidate records from a JSONL proposal log."""

    def __init__(self, path: str | Path, *, name: str = "replay") -> None:
        self.path = Path(path)
        self.name = name
        self._candidates = self._load()
        self._emitted: set[str] = set()

    @property
    def candidates(self) -> tuple[CandidateSpec, ...]:
        """Immutable replay contents for fail-closed campaign preflight."""

        return self._candidates

    def _load(self) -> tuple[CandidateSpec, ...]:
        candidates: list[CandidateSpec] = []
        ids: set[str] = set()
        for line_number, raw_line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ReplayError("line root is not an object")
                expected_hash = payload.get("sha256")
                spec_payload = payload.get("spec", payload)
                if "sha256" in spec_payload and "spec" not in payload:
                    spec_payload = dict(spec_payload)
                    spec_payload.pop("sha256")
                spec = CandidateSpec.from_dict(spec_payload)
            except Exception as exc:
                raise ReplayError(
                    f"invalid replay record at {self.path}:{line_number}: {exc}"
                ) from exc
            if expected_hash is not None and expected_hash != spec.content_hash:
                raise ReplayError(
                    f"candidate hash mismatch at {self.path}:{line_number}: "
                    f"expected {expected_hash}, got {spec.content_hash}"
                )
            if spec.candidate_id in ids:
                raise ReplayError(
                    f"duplicate candidate id at {self.path}:{line_number}: {spec.candidate_id}"
                )
            ids.add(spec.candidate_id)
            candidates.append(spec)
        return tuple(candidates)

    def propose(self, context: ProposalContext) -> ProposalBatch:
        matching = [
            candidate
            for candidate in self._candidates
            if candidate.candidate_id not in self._emitted
            and candidate.protocol_hash == context.protocol_hash
            and candidate.generation == context.generation
        ]
        selected = tuple(matching[: context.request_count])
        self._emitted.update(candidate.candidate_id for candidate in selected)
        return ProposalBatch(
            self.name, selected, exhausted=len(selected) == len(matching)
        )
