from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from factor_production.v5.artifacts.hashing import hash_json
from factor_production.v5.domain.enums import (
    FeedbackDisposition,
    FeedbackReason,
    ScoreVisibility,
)
from factor_production.v5.domain.models import CandidateSpecV5


@dataclass(frozen=True, slots=True)
class LocalFeedback:
    """Narrow feedback type accepted by a proposal provider.

    There is deliberately no generic metadata field and no constructor argument
    for visibility: an ExternalScoreBatch cannot be cast into this type without
    crossing an explicit, tested isolation guard.
    """

    candidate_id: str
    metric: str
    value: float
    generation: int

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.metric.strip():
            raise ValueError("candidate_id and metric must not be empty")
        if not math.isfinite(self.value):
            raise ValueError("feedback value must be finite")
        if self.generation < 0:
            raise ValueError("generation cannot be negative")

    @property
    def visibility(self) -> ScoreVisibility:
        return ScoreVisibility.LOCAL_RESEARCH


@dataclass(frozen=True, slots=True)
class SanitizedFeedback:
    """Hash-bound, coarse local feedback safe for an adaptive proposer.

    It intentionally carries neither raw metric values nor free-form metadata.
    The closed reason vocabulary prevents private/teacher score names from
    crossing back into the research loop.
    """

    candidate_id: str
    candidate_hash: str
    campaign_round: int
    disposition: FeedbackDisposition | str
    reason_codes: tuple[FeedbackReason | str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        if (
            not isinstance(self.candidate_hash, str)
            or len(self.candidate_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.candidate_hash)
        ):
            raise ValueError("candidate_hash must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.campaign_round, int)
            or isinstance(self.campaign_round, bool)
            or self.campaign_round < 0
        ):
            raise ValueError("campaign_round must be a non-negative integer")
        try:
            disposition = FeedbackDisposition(self.disposition)
        except ValueError as exc:
            raise ValueError(f"unsupported feedback disposition: {self.disposition!r}") from exc
        if not isinstance(self.reason_codes, (tuple, list)) or not self.reason_codes:
            raise ValueError("reason_codes must be a non-empty tuple or list")
        try:
            reasons = tuple(FeedbackReason(value) for value in self.reason_codes)
        except ValueError as exc:
            raise ValueError("feedback contains an unsupported reason code") from exc
        if len(reasons) != len(set(reasons)):
            raise ValueError("reason_codes must not contain duplicates")
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "reason_codes", reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "campaign_round": self.campaign_round,
            "disposition": self.disposition.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SanitizedFeedback":
        expected = {
            "candidate_id",
            "candidate_hash",
            "campaign_round",
            "disposition",
            "reason_codes",
        }
        if not isinstance(value, dict) or set(value) != expected:
            actual = sorted(value) if isinstance(value, dict) else type(value).__name__
            raise ValueError(
                f"sanitized feedback fields must be exactly {sorted(expected)}; got {actual}"
            )
        return cls(
            candidate_id=value["candidate_id"],
            candidate_hash=value["candidate_hash"],
            campaign_round=value["campaign_round"],
            disposition=value["disposition"],
            reason_codes=value["reason_codes"],
        )

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProposalContext:
    run_id: str
    protocol_hash: str
    generation: int
    request_count: int
    prior_candidate_hashes: tuple[str, ...] = ()
    local_feedback: tuple[LocalFeedback, ...] = ()
    sanitized_feedback: tuple[SanitizedFeedback, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.protocol_hash.strip():
            raise ValueError("run_id and protocol_hash must not be empty")
        if self.generation < 0:
            raise ValueError("generation cannot be negative")
        if self.request_count <= 0:
            raise ValueError("request_count must be positive")
        if len(self.prior_candidate_hashes) != len(set(self.prior_candidate_hashes)):
            raise ValueError("prior_candidate_hashes must be unique")
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for digest in self.prior_candidate_hashes
        ):
            raise ValueError("prior_candidate_hashes must be lowercase SHA-256 digests")
        if any(not isinstance(item, LocalFeedback) for item in self.local_feedback):
            raise ValueError("provider context accepts LocalFeedback objects only")
        if any(not isinstance(item, SanitizedFeedback) for item in self.sanitized_feedback):
            raise ValueError("sanitized_feedback accepts SanitizedFeedback objects only")
        feedback_ids = [item.candidate_id for item in self.sanitized_feedback]
        if len(feedback_ids) != len(set(feedback_ids)):
            raise ValueError("sanitized_feedback must contain at most one item per candidate")


@dataclass(frozen=True, slots=True)
class ProposalBatch:
    provider: str
    candidates: tuple[CandidateSpecV5, ...]
    exhausted: bool = False

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("proposal batch contains duplicate candidate IDs")
        if any(not isinstance(candidate, CandidateSpecV5) for candidate in self.candidates):
            raise ValueError("proposal batch accepts CandidateSpecV5 objects only")


class CandidateProvider(ABC):
    name: str

    @abstractmethod
    def propose(self, context: ProposalContext) -> ProposalBatch:
        raise NotImplementedError
