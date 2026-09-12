from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from factor_production.v5.artifacts.hashing import (
    canonical_json_bytes,
    hash_file,
    hash_json,
)
from factor_production.v5.domain.enums import ScoreVisibility
from factor_production.v5.orchestration.stopping import HoldoutLeakageError


CAMPAIGN_CLOSE_FDR_REQUEST_SCHEMA = "campaign-close-fdr-request/v1"
CAMPAIGN_CLOSE_FDR_REPORT_SCHEMA = "campaign-close-fdr-report/v1"
CAMPAIGN_CLOSE_FDR_FAMILY_DEFINITION = (
    "all_unique_frozen_hypotheses_bound_to_one_execution_aware_semantics"
)
CAMPAIGN_CLOSE_FDR_ROLE = "campaign_close_local_multiple_testing_diagnostic_only"
CAMPAIGN_CLOSE_FDR_INTERPRETATION = (
    "This local in-sample campaign-close diagnostic is not external out-of-sample "
    "evidence, production eligibility, factor admission, or a prediction of any "
    "hidden teacher/official score."
)
ATTEMPT_COLLAPSE_POLICY = (
    "one_hypothesis_one_test_latest_terminal_attempt_by_attempt_number"
)
FAILED_TEST_POLICY = (
    "missing_failed_expired_or_nonfinite_primary_test_is_included_as_p_equals_1"
)

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_FORBIDDEN_INPUT_KEY_TOKENS = (
    "teacher",
    "official",
    "hidden_score",
    "holdout_score",
    "external_score",
    "admission_score",
)


class CampaignCloseFDRError(ValueError):
    """The campaign-close family or its content-addressed report is invalid."""


class TerminalAttemptStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise CampaignCloseFDRError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise CampaignCloseFDRError(f"{name} contains unsupported characters")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CampaignCloseFDRError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    result = _require_nonnegative_int(value, name)
    if result == 0:
        raise CampaignCloseFDRError(f"{name} must be a positive integer")
    return result


def _strict_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise CampaignCloseFDRError(
            f"{name} fields must be exact; missing={missing}, extra={extra}"
        )


def _reject_external_input(value: Any, *, path: str = "$") -> None:
    """Reject hidden/official fields before an allowlist projection can drop them."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CampaignCloseFDRError(f"non-string input key at {path}")
            normalized = key.casefold()
            if any(token in normalized for token in _FORBIDDEN_INPUT_KEY_TOKENS):
                raise HoldoutLeakageError(
                    f"external teacher/official field is forbidden at {path}.{key}"
                )
            _reject_external_input(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_external_input(item, path=f"{path}[{index}]")


def _normalize_optional_p(value: Any, name: str) -> float | None:
    """Normalize non-finite source statistics to the fail-closed missing state."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise CampaignCloseFDRError(f"{name} must be a probability or null")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CampaignCloseFDRError(f"{name} must be a probability or null") from exc
    if not math.isfinite(number):
        return None
    if number < 0.0 or number > 1.0:
        raise CampaignCloseFDRError(f"{name} must be between zero and one")
    return number


@dataclass(frozen=True, slots=True)
class FrozenHypothesis:
    """One unique, formula-frozen member of the campaign-wide test family."""

    candidate_id: str
    spec_hash: str
    semantic_hash: str
    campaign_round: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _require_id(self.candidate_id, "candidate_id")
        )
        object.__setattr__(self, "spec_hash", _require_digest(self.spec_hash, "spec_hash"))
        object.__setattr__(
            self,
            "semantic_hash",
            _require_digest(self.semantic_hash, "semantic_hash"),
        )
        object.__setattr__(
            self,
            "campaign_round",
            _require_nonnegative_int(self.campaign_round, "campaign_round"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "spec_hash": self.spec_hash,
            "semantic_hash": self.semantic_hash,
            "campaign_round": self.campaign_round,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenHypothesis":
        if not isinstance(value, Mapping):
            raise CampaignCloseFDRError("frozen hypothesis must be an object")
        _reject_external_input(value)
        _strict_fields(
            value,
            {"candidate_id", "spec_hash", "semantic_hash", "campaign_round"},
            "frozen hypothesis",
        )
        return cls(
            candidate_id=value["candidate_id"],
            spec_hash=value["spec_hash"],
            semantic_hash=value["semantic_hash"],
            campaign_round=value["campaign_round"],
        )


@dataclass(frozen=True, slots=True)
class PrimaryTestAttempt:
    """Terminal local attempt carrying only the two preregistered primary tests."""

    attempt_id: int
    attempt_number: int
    candidate_id: str
    spec_hash: str
    status: TerminalAttemptStatus | str
    raw_primary_p: float | None
    neutral_primary_p: float | None
    visibility: ScoreVisibility | str = ScoreVisibility.LOCAL_RESEARCH

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _require_positive_int(self.attempt_id, "attempt_id"))
        object.__setattr__(
            self,
            "attempt_number",
            _require_positive_int(self.attempt_number, "attempt_number"),
        )
        object.__setattr__(
            self, "candidate_id", _require_id(self.candidate_id, "candidate_id")
        )
        object.__setattr__(self, "spec_hash", _require_digest(self.spec_hash, "spec_hash"))
        try:
            status = TerminalAttemptStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise CampaignCloseFDRError(
                "attempt status must be terminal: succeeded, failed, or expired"
            ) from exc
        object.__setattr__(self, "status", status)
        try:
            visibility = ScoreVisibility(self.visibility)
        except (TypeError, ValueError) as exc:
            raise HoldoutLeakageError(
                "unknown score visibility is forbidden at campaign close"
            ) from exc
        if visibility is not ScoreVisibility.LOCAL_RESEARCH:
            raise HoldoutLeakageError(
                "external teacher/official visibility is forbidden at campaign close"
            )
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(
            self,
            "raw_primary_p",
            _normalize_optional_p(self.raw_primary_p, "raw_primary_p"),
        )
        object.__setattr__(
            self,
            "neutral_primary_p",
            _normalize_optional_p(self.neutral_primary_p, "neutral_primary_p"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "attempt_number": self.attempt_number,
            "candidate_id": self.candidate_id,
            "spec_hash": self.spec_hash,
            "status": self.status.value,
            "raw_primary_p": self.raw_primary_p,
            "neutral_primary_p": self.neutral_primary_p,
            "visibility": self.visibility.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrimaryTestAttempt":
        if not isinstance(value, Mapping):
            raise CampaignCloseFDRError("primary-test attempt must be an object")
        _reject_external_input(value)
        _strict_fields(
            value,
            {
                "attempt_id",
                "attempt_number",
                "candidate_id",
                "spec_hash",
                "status",
                "raw_primary_p",
                "neutral_primary_p",
                "visibility",
            },
            "primary-test attempt",
        )
        visibility = value["visibility"]
        if visibility != ScoreVisibility.LOCAL_RESEARCH.value:
            raise HoldoutLeakageError(
                "external teacher/official visibility is forbidden at campaign close"
            )
        return cls(
            attempt_id=value["attempt_id"],
            attempt_number=value["attempt_number"],
            candidate_id=value["candidate_id"],
            spec_hash=value["spec_hash"],
            status=value["status"],
            raw_primary_p=value["raw_primary_p"],
            neutral_primary_p=value["neutral_primary_p"],
            visibility=visibility,
        )


@dataclass(frozen=True, slots=True)
class CampaignCloseFDRRequest:
    """Strict, hashable input binding one complete campaign-close family."""

    campaign_id: str
    protocol_hash: str
    execution_semantics_hash: str
    registry_content_hash: str
    hypotheses: tuple[FrozenHypothesis, ...] | Sequence[FrozenHypothesis]
    attempts: tuple[PrimaryTestAttempt, ...] | Sequence[PrimaryTestAttempt]
    campaign_closed: bool = True
    schema_version: str = CAMPAIGN_CLOSE_FDR_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_CLOSE_FDR_REQUEST_SCHEMA:
            raise CampaignCloseFDRError("unsupported campaign-close FDR request schema")
        object.__setattr__(self, "campaign_id", _require_id(self.campaign_id, "campaign_id"))
        for name in (
            "protocol_hash",
            "execution_semantics_hash",
            "registry_content_hash",
        ):
            object.__setattr__(self, name, _require_digest(getattr(self, name), name))
        if self.campaign_closed is not True:
            raise CampaignCloseFDRError(
                "campaign-wide FDR can only be frozen after campaign_closed=true"
            )
        hypotheses = tuple(self.hypotheses)
        attempts = tuple(self.attempts)
        if not hypotheses or any(
            not isinstance(item, FrozenHypothesis) for item in hypotheses
        ):
            raise CampaignCloseFDRError(
                "hypotheses must be a non-empty sequence of FrozenHypothesis values"
            )
        if any(not isinstance(item, PrimaryTestAttempt) for item in attempts):
            raise CampaignCloseFDRError(
                "attempts must contain only PrimaryTestAttempt values"
            )

        candidate_ids = [item.candidate_id for item in hypotheses]
        spec_hashes = [item.spec_hash for item in hypotheses]
        semantic_hashes = [item.semantic_hash for item in hypotheses]
        _require_unique(candidate_ids, "candidate_id")
        _require_unique(spec_hashes, "spec_hash")
        _require_unique(semantic_hashes, "semantic_hash")
        hypothesis_by_id = {item.candidate_id: item for item in hypotheses}

        attempt_ids = [item.attempt_id for item in attempts]
        _require_unique(attempt_ids, "attempt_id")
        attempt_numbers: set[tuple[str, int]] = set()
        for attempt in attempts:
            hypothesis = hypothesis_by_id.get(attempt.candidate_id)
            if hypothesis is None:
                raise CampaignCloseFDRError(
                    f"attempt references unknown frozen hypothesis: {attempt.candidate_id}"
                )
            if attempt.spec_hash != hypothesis.spec_hash:
                raise CampaignCloseFDRError(
                    f"attempt spec_hash mismatch for {attempt.candidate_id}"
                )
            key = (attempt.candidate_id, attempt.attempt_number)
            if key in attempt_numbers:
                raise CampaignCloseFDRError(
                    "attempt_number must be unique within each hypothesis"
                )
            attempt_numbers.add(key)

        object.__setattr__(
            self,
            "hypotheses",
            tuple(sorted(hypotheses, key=lambda item: item.candidate_id)),
        )
        object.__setattr__(
            self,
            "attempts",
            tuple(
                sorted(
                    attempts,
                    key=lambda item: (
                        item.candidate_id,
                        item.attempt_number,
                        item.attempt_id,
                    ),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "protocol_hash": self.protocol_hash,
            "execution_semantics_hash": self.execution_semantics_hash,
            "registry_content_hash": self.registry_content_hash,
            "campaign_closed": self.campaign_closed,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "attempts": [item.to_dict() for item in self.attempts],
        }

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignCloseFDRRequest":
        if not isinstance(value, Mapping):
            raise CampaignCloseFDRError("campaign-close FDR request must be an object")
        _reject_external_input(value)
        _strict_fields(
            value,
            {
                "schema_version",
                "campaign_id",
                "protocol_hash",
                "execution_semantics_hash",
                "registry_content_hash",
                "campaign_closed",
                "hypotheses",
                "attempts",
            },
            "campaign-close FDR request",
        )
        if not isinstance(value["hypotheses"], list):
            raise CampaignCloseFDRError("hypotheses must be a JSON list")
        if not isinstance(value["attempts"], list):
            raise CampaignCloseFDRError("attempts must be a JSON list")
        return cls(
            schema_version=value["schema_version"],
            campaign_id=value["campaign_id"],
            protocol_hash=value["protocol_hash"],
            execution_semantics_hash=value["execution_semantics_hash"],
            registry_content_hash=value["registry_content_hash"],
            campaign_closed=value["campaign_closed"],
            hypotheses=tuple(FrozenHypothesis.from_dict(item) for item in value["hypotheses"]),
            attempts=tuple(PrimaryTestAttempt.from_dict(item) for item in value["attempts"]),
        )


def _require_unique(values: Sequence[Any], name: str) -> None:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise CampaignCloseFDRError(
            f"frozen family has duplicate {name}: {sorted(duplicates)!r}"
        )


def benjamini_hochberg_complete_family(p_values: Sequence[float]) -> tuple[float, ...]:
    """BH adjustment for a complete family; missing tests must already be p=1."""

    normalized: list[float] = []
    for index, value in enumerate(p_values):
        if isinstance(value, bool):
            raise CampaignCloseFDRError(f"p_values[{index}] must be a probability")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise CampaignCloseFDRError(
                f"p_values[{index}] must be a probability"
            ) from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise CampaignCloseFDRError(
                "complete-family BH requires finite p-values between zero and one"
            )
        normalized.append(number)
    if not normalized:
        raise CampaignCloseFDRError("complete-family BH requires at least one hypothesis")

    ordered = sorted(enumerate(normalized), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    adjusted = [1.0] * family_size
    running = 1.0
    for offset in range(family_size - 1, -1, -1):
        original_index, p_value = ordered[offset]
        rank = offset + 1
        running = min(running, p_value * family_size / rank)
        adjusted[original_index] = min(max(float(running), 0.0), 1.0)
    return tuple(adjusted)


@dataclass(frozen=True, slots=True)
class CampaignCloseHypothesisResult:
    candidate_id: str
    spec_hash: str
    semantic_hash: str
    attempt_count: int
    authoritative_attempt_id: int | None
    authoritative_attempt_number: int | None
    authoritative_status: str
    raw_primary_p: float
    raw_bh_q: float
    raw_p_imputed_one: bool
    neutral_primary_p: float
    neutral_bh_q: float
    neutral_p_imputed_one: bool

    def __post_init__(self) -> None:
        _require_id(self.candidate_id, "candidate_id")
        _require_digest(self.spec_hash, "spec_hash")
        _require_digest(self.semantic_hash, "semantic_hash")
        _require_nonnegative_int(self.attempt_count, "attempt_count")
        for name in ("raw_primary_p", "raw_bh_q", "neutral_primary_p", "neutral_bh_q"):
            number = _normalize_optional_p(getattr(self, name), name)
            if number is None:
                raise CampaignCloseFDRError(f"{name} must be finite in an FDR report")
        if not isinstance(self.raw_p_imputed_one, bool) or not isinstance(
            self.neutral_p_imputed_one, bool
        ):
            raise CampaignCloseFDRError("imputation flags must be booleans")
        allowed_statuses = {
            "missing",
            TerminalAttemptStatus.SUCCEEDED.value,
            TerminalAttemptStatus.FAILED.value,
            TerminalAttemptStatus.EXPIRED.value,
        }
        if self.authoritative_status not in allowed_statuses:
            raise CampaignCloseFDRError("unknown authoritative attempt status")
        if self.authoritative_status == "missing":
            if (
                self.attempt_count != 0
                or self.authoritative_attempt_id is not None
                or self.authoritative_attempt_number is not None
            ):
                raise CampaignCloseFDRError(
                    "missing evaluation cannot name an authoritative attempt"
                )
        else:
            if self.attempt_count <= 0:
                raise CampaignCloseFDRError(
                    "terminal evaluation must have at least one attempt"
                )
            _require_positive_int(
                self.authoritative_attempt_id, "authoritative_attempt_id"
            )
            _require_positive_int(
                self.authoritative_attempt_number, "authoritative_attempt_number"
            )
        if self.authoritative_status in {
            "missing",
            TerminalAttemptStatus.FAILED.value,
            TerminalAttemptStatus.EXPIRED.value,
        } and (
            self.raw_primary_p != 1.0
            or self.neutral_primary_p != 1.0
            or not self.raw_p_imputed_one
            or not self.neutral_p_imputed_one
        ):
            raise CampaignCloseFDRError(
                "missing/failed/expired results must be represented by p=1"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "spec_hash": self.spec_hash,
            "semantic_hash": self.semantic_hash,
            "attempt_count": self.attempt_count,
            "authoritative_attempt_id": self.authoritative_attempt_id,
            "authoritative_attempt_number": self.authoritative_attempt_number,
            "authoritative_status": self.authoritative_status,
            "raw_primary_p": self.raw_primary_p,
            "raw_bh_q": self.raw_bh_q,
            "raw_p_imputed_one": self.raw_p_imputed_one,
            "neutral_primary_p": self.neutral_primary_p,
            "neutral_bh_q": self.neutral_bh_q,
            "neutral_p_imputed_one": self.neutral_p_imputed_one,
        }


@dataclass(frozen=True, slots=True)
class CampaignCloseFDRReport:
    campaign_id: str
    protocol_hash: str
    execution_semantics_hash: str
    registry_content_hash: str
    request_sha256: str
    family_size: int
    total_attempt_count: int
    failed_attempt_count: int
    failed_hypothesis_count: int
    missing_evaluation_count: int
    invalid_raw_primary_test_count: int
    invalid_neutral_primary_test_count: int
    hypotheses_with_retries_count: int
    results: tuple[CampaignCloseHypothesisResult, ...]

    def __post_init__(self) -> None:
        _require_id(self.campaign_id, "campaign_id")
        for name in (
            "protocol_hash",
            "execution_semantics_hash",
            "registry_content_hash",
            "request_sha256",
        ):
            _require_digest(getattr(self, name), name)
        for name in (
            "family_size",
            "total_attempt_count",
            "failed_attempt_count",
            "failed_hypothesis_count",
            "missing_evaluation_count",
            "invalid_raw_primary_test_count",
            "invalid_neutral_primary_test_count",
            "hypotheses_with_retries_count",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        if self.family_size <= 0 or self.family_size != len(self.results):
            raise CampaignCloseFDRError("family_size must equal the non-empty result count")
        if any(
            not isinstance(item, CampaignCloseHypothesisResult)
            for item in self.results
        ):
            raise CampaignCloseFDRError(
                "results must contain only CampaignCloseHypothesisResult values"
            )
        if tuple(sorted(self.results, key=lambda item: item.candidate_id)) != self.results:
            raise CampaignCloseFDRError("FDR report results must be sorted by candidate_id")
        _require_unique([item.candidate_id for item in self.results], "result candidate_id")
        if self.total_attempt_count != sum(item.attempt_count for item in self.results):
            raise CampaignCloseFDRError("total_attempt_count does not match results")
        if self.failed_attempt_count > self.total_attempt_count:
            raise CampaignCloseFDRError(
                "failed_attempt_count cannot exceed total_attempt_count"
            )
        expected_failed_hypotheses = sum(
            item.authoritative_status
            in {TerminalAttemptStatus.FAILED.value, TerminalAttemptStatus.EXPIRED.value}
            for item in self.results
        )
        expected_missing = sum(
            item.authoritative_status == "missing" for item in self.results
        )
        expected_invalid_raw = sum(
            item.authoritative_status == TerminalAttemptStatus.SUCCEEDED.value
            and item.raw_p_imputed_one
            for item in self.results
        )
        expected_invalid_neutral = sum(
            item.authoritative_status == TerminalAttemptStatus.SUCCEEDED.value
            and item.neutral_p_imputed_one
            for item in self.results
        )
        expected_retries = sum(item.attempt_count > 1 for item in self.results)
        for name, expected_value in (
            ("failed_hypothesis_count", expected_failed_hypotheses),
            ("missing_evaluation_count", expected_missing),
            ("invalid_raw_primary_test_count", expected_invalid_raw),
            ("invalid_neutral_primary_test_count", expected_invalid_neutral),
            ("hypotheses_with_retries_count", expected_retries),
        ):
            if getattr(self, name) != expected_value:
                raise CampaignCloseFDRError(f"{name} does not match results")
        expected_raw_q = benjamini_hochberg_complete_family(
            [item.raw_primary_p for item in self.results]
        )
        expected_neutral_q = benjamini_hochberg_complete_family(
            [item.neutral_primary_p for item in self.results]
        )
        if expected_raw_q != tuple(item.raw_bh_q for item in self.results):
            raise CampaignCloseFDRError("raw BH q-values do not match the complete family")
        if expected_neutral_q != tuple(item.neutral_bh_q for item in self.results):
            raise CampaignCloseFDRError(
                "neutral BH q-values do not match the complete family"
            )

    def _content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CAMPAIGN_CLOSE_FDR_REPORT_SCHEMA,
            "campaign_id": self.campaign_id,
            "protocol_hash": self.protocol_hash,
            "execution_semantics_hash": self.execution_semantics_hash,
            "registry_content_hash": self.registry_content_hash,
            "request_sha256": self.request_sha256,
            "campaign_closed": True,
            "multiple_testing_method": "benjamini_hochberg",
            "tests_adjusted_separately": ["raw_primary", "neutral_primary"],
            "family_definition": CAMPAIGN_CLOSE_FDR_FAMILY_DEFINITION,
            "attempt_collapse_policy": ATTEMPT_COLLAPSE_POLICY,
            "failed_test_policy": FAILED_TEST_POLICY,
            "role": CAMPAIGN_CLOSE_FDR_ROLE,
            "interpretation": CAMPAIGN_CLOSE_FDR_INTERPRETATION,
            "not_external_out_of_sample_or_admission_evidence": True,
            "contains_external_teacher_or_official_results": False,
            "family_size": self.family_size,
            "total_attempt_count": self.total_attempt_count,
            "failed_attempt_count": self.failed_attempt_count,
            "failed_hypothesis_count": self.failed_hypothesis_count,
            "missing_evaluation_count": self.missing_evaluation_count,
            "invalid_raw_primary_test_count": self.invalid_raw_primary_test_count,
            "invalid_neutral_primary_test_count": self.invalid_neutral_primary_test_count,
            "hypotheses_with_retries_count": self.hypotheses_with_retries_count,
            "results": [item.to_dict() for item in self.results],
        }

    @property
    def content_hash(self) -> str:
        return hash_json(self._content_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_payload(), "content_sha256": self.content_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignCloseFDRReport":
        if not isinstance(value, Mapping):
            raise CampaignCloseFDRError("campaign-close FDR report must be an object")
        expected = {
            "schema_version",
            "content_sha256",
            "campaign_id",
            "protocol_hash",
            "execution_semantics_hash",
            "registry_content_hash",
            "request_sha256",
            "campaign_closed",
            "multiple_testing_method",
            "tests_adjusted_separately",
            "family_definition",
            "attempt_collapse_policy",
            "failed_test_policy",
            "role",
            "interpretation",
            "not_external_out_of_sample_or_admission_evidence",
            "contains_external_teacher_or_official_results",
            "family_size",
            "total_attempt_count",
            "failed_attempt_count",
            "failed_hypothesis_count",
            "missing_evaluation_count",
            "invalid_raw_primary_test_count",
            "invalid_neutral_primary_test_count",
            "hypotheses_with_retries_count",
            "results",
        }
        _strict_fields(value, expected, "campaign-close FDR report")
        content_hash = _require_digest(value["content_sha256"], "content_sha256")
        payload = dict(value)
        del payload["content_sha256"]
        if hash_json(payload) != content_hash:
            raise CampaignCloseFDRError("campaign-close FDR content hash mismatch")
        fixed = {
            "schema_version": CAMPAIGN_CLOSE_FDR_REPORT_SCHEMA,
            "campaign_closed": True,
            "multiple_testing_method": "benjamini_hochberg",
            "tests_adjusted_separately": ["raw_primary", "neutral_primary"],
            "family_definition": CAMPAIGN_CLOSE_FDR_FAMILY_DEFINITION,
            "attempt_collapse_policy": ATTEMPT_COLLAPSE_POLICY,
            "failed_test_policy": FAILED_TEST_POLICY,
            "role": CAMPAIGN_CLOSE_FDR_ROLE,
            "interpretation": CAMPAIGN_CLOSE_FDR_INTERPRETATION,
            "not_external_out_of_sample_or_admission_evidence": True,
            "contains_external_teacher_or_official_results": False,
        }
        for key, expected_value in fixed.items():
            if value[key] != expected_value:
                raise CampaignCloseFDRError(f"invalid fixed report governance field: {key}")
        if not isinstance(value["results"], list):
            raise CampaignCloseFDRError("results must be a JSON list")
        results = tuple(_result_from_dict(item) for item in value["results"])
        report = cls(
            campaign_id=value["campaign_id"],
            protocol_hash=value["protocol_hash"],
            execution_semantics_hash=value["execution_semantics_hash"],
            registry_content_hash=value["registry_content_hash"],
            request_sha256=value["request_sha256"],
            family_size=value["family_size"],
            total_attempt_count=value["total_attempt_count"],
            failed_attempt_count=value["failed_attempt_count"],
            failed_hypothesis_count=value["failed_hypothesis_count"],
            missing_evaluation_count=value["missing_evaluation_count"],
            invalid_raw_primary_test_count=value["invalid_raw_primary_test_count"],
            invalid_neutral_primary_test_count=value["invalid_neutral_primary_test_count"],
            hypotheses_with_retries_count=value["hypotheses_with_retries_count"],
            results=results,
        )
        if report.content_hash != content_hash:
            raise CampaignCloseFDRError("campaign-close FDR report does not round-trip")
        return report


def _result_from_dict(value: Mapping[str, Any]) -> CampaignCloseHypothesisResult:
    if not isinstance(value, Mapping):
        raise CampaignCloseFDRError("FDR result must be an object")
    expected = {
        "candidate_id",
        "spec_hash",
        "semantic_hash",
        "attempt_count",
        "authoritative_attempt_id",
        "authoritative_attempt_number",
        "authoritative_status",
        "raw_primary_p",
        "raw_bh_q",
        "raw_p_imputed_one",
        "neutral_primary_p",
        "neutral_bh_q",
        "neutral_p_imputed_one",
    }
    _strict_fields(value, expected, "FDR result")
    return CampaignCloseHypothesisResult(**dict(value))


def build_campaign_close_fdr_report(
    request: CampaignCloseFDRRequest,
) -> CampaignCloseFDRReport:
    """Collapse attempts by hypothesis, impute failed tests, then run two BH families."""

    if not isinstance(request, CampaignCloseFDRRequest):
        raise CampaignCloseFDRError("request must be a CampaignCloseFDRRequest")
    attempts_by_id: dict[str, list[PrimaryTestAttempt]] = {
        item.candidate_id: [] for item in request.hypotheses
    }
    for attempt in request.attempts:
        attempts_by_id[attempt.candidate_id].append(attempt)

    intermediate: list[dict[str, Any]] = []
    failed_hypotheses = 0
    missing_evaluations = 0
    invalid_raw = 0
    invalid_neutral = 0
    retries = 0
    for hypothesis in request.hypotheses:
        attempts = attempts_by_id[hypothesis.candidate_id]
        if len(attempts) > 1:
            retries += 1
        authoritative = (
            max(attempts, key=lambda item: (item.attempt_number, item.attempt_id))
            if attempts
            else None
        )
        if authoritative is None:
            missing_evaluations += 1
            status = "missing"
            raw_p = neutral_p = 1.0
            raw_imputed = neutral_imputed = True
        elif authoritative.status is not TerminalAttemptStatus.SUCCEEDED:
            failed_hypotheses += 1
            status = authoritative.status.value
            raw_p = neutral_p = 1.0
            raw_imputed = neutral_imputed = True
        else:
            status = authoritative.status.value
            raw_imputed = authoritative.raw_primary_p is None
            neutral_imputed = authoritative.neutral_primary_p is None
            raw_p = 1.0 if raw_imputed else authoritative.raw_primary_p
            neutral_p = 1.0 if neutral_imputed else authoritative.neutral_primary_p
            invalid_raw += int(raw_imputed)
            invalid_neutral += int(neutral_imputed)
        intermediate.append(
            {
                "hypothesis": hypothesis,
                "attempt_count": len(attempts),
                "authoritative": authoritative,
                "status": status,
                "raw_p": float(raw_p),
                "neutral_p": float(neutral_p),
                "raw_imputed": raw_imputed,
                "neutral_imputed": neutral_imputed,
            }
        )

    raw_q = benjamini_hochberg_complete_family(
        [item["raw_p"] for item in intermediate]
    )
    neutral_q = benjamini_hochberg_complete_family(
        [item["neutral_p"] for item in intermediate]
    )
    results: list[CampaignCloseHypothesisResult] = []
    for item, raw_q_value, neutral_q_value in zip(
        intermediate, raw_q, neutral_q, strict=True
    ):
        hypothesis = item["hypothesis"]
        authoritative = item["authoritative"]
        results.append(
            CampaignCloseHypothesisResult(
                candidate_id=hypothesis.candidate_id,
                spec_hash=hypothesis.spec_hash,
                semantic_hash=hypothesis.semantic_hash,
                attempt_count=item["attempt_count"],
                authoritative_attempt_id=(
                    None if authoritative is None else authoritative.attempt_id
                ),
                authoritative_attempt_number=(
                    None if authoritative is None else authoritative.attempt_number
                ),
                authoritative_status=item["status"],
                raw_primary_p=item["raw_p"],
                raw_bh_q=raw_q_value,
                raw_p_imputed_one=item["raw_imputed"],
                neutral_primary_p=item["neutral_p"],
                neutral_bh_q=neutral_q_value,
                neutral_p_imputed_one=item["neutral_imputed"],
            )
        )

    failed_attempts = sum(
        attempt.status is not TerminalAttemptStatus.SUCCEEDED
        for attempt in request.attempts
    )
    return CampaignCloseFDRReport(
        campaign_id=request.campaign_id,
        protocol_hash=request.protocol_hash,
        execution_semantics_hash=request.execution_semantics_hash,
        registry_content_hash=request.registry_content_hash,
        request_sha256=request.content_hash,
        family_size=len(request.hypotheses),
        total_attempt_count=len(request.attempts),
        failed_attempt_count=failed_attempts,
        failed_hypothesis_count=failed_hypotheses,
        missing_evaluation_count=missing_evaluations,
        invalid_raw_primary_test_count=invalid_raw,
        invalid_neutral_primary_test_count=invalid_neutral,
        hypotheses_with_retries_count=retries,
        results=tuple(results),
    )


def verify_campaign_close_fdr_report(
    report: CampaignCloseFDRReport,
    request: CampaignCloseFDRRequest,
) -> None:
    """Verify the report against the exact frozen request, not just its own hash."""

    if not isinstance(report, CampaignCloseFDRReport):
        raise CampaignCloseFDRError("report must be a CampaignCloseFDRReport")
    if not isinstance(request, CampaignCloseFDRRequest):
        raise CampaignCloseFDRError("request must be a CampaignCloseFDRRequest")
    if report.request_sha256 != request.content_hash:
        raise CampaignCloseFDRError("report is not bound to this FDR request")
    expected = build_campaign_close_fdr_report(request)
    if report.to_dict() != expected.to_dict():
        raise CampaignCloseFDRError("report differs from deterministic FDR recomputation")


def write_campaign_close_fdr_report(
    report: CampaignCloseFDRReport,
    path: str | Path,
) -> str:
    """Atomically write canonical JSON plus a file-level SHA-256 sidecar."""

    if not isinstance(report, CampaignCloseFDRReport):
        raise CampaignCloseFDRError("report must be a CampaignCloseFDRReport")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(report.to_dict()) + b"\n"
    temporary = target.with_name(f".{target.name}.{report.content_hash}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, target)
    digest = hash_file(target)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.{digest}.tmp")
    sidecar_temporary.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    os.replace(sidecar_temporary, sidecar)
    return digest


def load_campaign_close_fdr_report(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> CampaignCloseFDRReport:
    target = Path(path)
    if expected_sha256 is not None:
        expected = _require_digest(expected_sha256, "expected_sha256")
        if hash_file(target) != expected:
            raise CampaignCloseFDRError("campaign-close FDR file hash mismatch")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignCloseFDRError("invalid campaign-close FDR JSON") from exc
    if not isinstance(value, Mapping):
        raise CampaignCloseFDRError("campaign-close FDR JSON must be an object")
    return CampaignCloseFDRReport.from_dict(value)


def report_results_by_candidate(
    report: CampaignCloseFDRReport,
) -> Mapping[str, CampaignCloseHypothesisResult]:
    """Read-only ID lookup without creating a second statistical family."""

    return MappingProxyType({item.candidate_id: item for item in report.results})
