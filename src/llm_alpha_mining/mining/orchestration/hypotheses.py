from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from llm_alpha_mining.mining.artifacts.hashing import hash_json
from llm_alpha_mining.mining.domain.enums import ProposalKind
from llm_alpha_mining.mining.domain.models import CandidateSpec


HYPOTHESIS_REGISTRY_SCHEMA = "hypothesis-registry/v1"


class HypothesisRegistryError(ValueError):
    """A proposal cannot enter the append-only campaign hypothesis ledger."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}:{detail}")
        self.code = code
        self.detail = detail


def _normalized_hypothesis(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _expression_structure(expression: str) -> str:
    # Import lazily to avoid an orchestration-package initialization cycle.
    # The panel and authoritative campaign registry must use exactly one
    # associative/commutative canonicalizer for semantic de-duplication.
    from llm_alpha_mining.mining.llm.schemas import (
        StructuredOutputError,
        canonical_expression_ast,
    )

    try:
        return canonical_expression_ast(expression)
    except StructuredOutputError:
        raise HypothesisRegistryError(
            "invalid_expression_syntax", "invalid expression"
        ) from None


def semantic_signal_hash(spec: CandidateSpec) -> str:
    """Hash executable signal semantics, ignoring names, prose and sign fishing.

    ``direction`` is deliberately excluded: resubmitting the same signal with
    its sign flipped is still a duplicate hypothesis.  Aggregation is included
    because it changes the actual daily signal produced by a minute formula.
    """

    wire = spec.to_dict()
    from llm_alpha_mining.mining.llm.schemas import canonical_aggregation

    return hash_json(
        {
            "expression_ast": _expression_structure(spec.expression),
            "frequency": spec.frequency.value,
            "aggregation": canonical_aggregation(
                spec.frequency.value,
                wire["aggregation"],
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class HypothesisEntry:
    candidate_id: str
    spec_hash: str
    semantic_hash: str
    hypothesis_hash: str
    campaign_round: int
    lineage_depth: int
    proposal_kind: ProposalKind
    parent_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "spec_hash": self.spec_hash,
            "semantic_hash": self.semantic_hash,
            "hypothesis_hash": self.hypothesis_hash,
            "campaign_round": self.campaign_round,
            "lineage_depth": self.lineage_depth,
            "proposal_kind": self.proposal_kind.value,
            "parent_ids": list(self.parent_ids),
        }


class HypothesisRegistry:
    """Campaign-wide append-only lineage and semantic de-duplication ledger."""

    def __init__(self, protocol_hash: str) -> None:
        if (
            not isinstance(protocol_hash, str)
            or len(protocol_hash) != 64
            or any(character not in "0123456789abcdef" for character in protocol_hash)
        ):
            raise ValueError("protocol_hash must be a lowercase SHA-256 digest")
        self.protocol_hash = protocol_hash
        self._entries: dict[str, HypothesisEntry] = {}
        self._spec_hashes: set[str] = set()
        self._semantic_hashes: set[str] = set()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, candidate_id: str) -> bool:
        return candidate_id in self._entries

    @property
    def entries(self) -> tuple[HypothesisEntry, ...]:
        return tuple(
            sorted(
                self._entries.values(),
                key=lambda item: (item.campaign_round, item.candidate_id),
            )
        )

    @property
    def candidate_hashes(self) -> tuple[str, ...]:
        return tuple(entry.spec_hash for entry in self.entries)

    @property
    def next_campaign_round(self) -> int:
        if not self._entries:
            return 0
        return max(entry.campaign_round for entry in self._entries.values()) + 1

    def get(self, candidate_id: str) -> HypothesisEntry:
        try:
            return self._entries[candidate_id]
        except KeyError:
            raise KeyError("unknown hypothesis candidate") from None

    def preflight(self, spec: CandidateSpec) -> HypothesisEntry:
        if spec.protocol_hash != self.protocol_hash:
            raise HypothesisRegistryError(
                "protocol_mismatch",
                f"{spec.protocol_hash}!={self.protocol_hash}",
            )
        if spec.candidate_id in self._entries:
            raise HypothesisRegistryError("duplicate_candidate_id", spec.candidate_id)
        if spec.content_hash in self._spec_hashes:
            raise HypothesisRegistryError("duplicate_spec_hash", spec.content_hash)
        semantic_hash = semantic_signal_hash(spec)
        if semantic_hash in self._semantic_hashes:
            raise HypothesisRegistryError("duplicate_semantic_signal", semantic_hash)

        if spec.proposal_kind is ProposalKind.ROOT:
            resolved_depth = 0
        else:
            missing = [
                parent_id
                for parent_id in spec.parent_ids
                if parent_id not in self._entries
            ]
            if missing:
                raise HypothesisRegistryError(
                    "missing_parent",
                    ",".join(sorted(missing)),
                )
            parent_entries = [self._entries[parent_id] for parent_id in spec.parent_ids]
            if any(
                parent.campaign_round >= spec.campaign_round
                for parent in parent_entries
            ):
                raise HypothesisRegistryError(
                    "parent_not_from_prior_round",
                    spec.candidate_id,
                )
            resolved_depth = 1 + max(parent.lineage_depth for parent in parent_entries)
            if spec.has_explicit_lineage and spec.lineage_depth != resolved_depth:
                raise HypothesisRegistryError(
                    "lineage_depth_mismatch",
                    f"declared={spec.lineage_depth},expected={resolved_depth}",
                )

        return HypothesisEntry(
            candidate_id=spec.candidate_id,
            spec_hash=spec.content_hash,
            semantic_hash=semantic_hash,
            hypothesis_hash=hash_json(
                {
                    "family": spec.family.casefold(),
                    "hypothesis": _normalized_hypothesis(spec.hypothesis),
                }
            ),
            campaign_round=spec.campaign_round,
            lineage_depth=resolved_depth,
            proposal_kind=spec.proposal_kind,
            parent_ids=spec.parent_ids,
        )

    def register(self, spec: CandidateSpec) -> HypothesisEntry:
        entry = self.preflight(spec)
        self._entries[entry.candidate_id] = entry
        self._spec_hashes.add(entry.spec_hash)
        self._semantic_hashes.add(entry.semantic_hash)
        return entry

    def register_many(
        self, candidates: Iterable[CandidateSpec]
    ) -> tuple[HypothesisEntry, ...]:
        registered: list[HypothesisEntry] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                item.campaign_round,
                item.lineage_depth,
                item.candidate_id,
            ),
        ):
            registered.append(self.register(candidate))
        return tuple(registered)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": HYPOTHESIS_REGISTRY_SCHEMA,
            "protocol_hash": self.protocol_hash,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())
