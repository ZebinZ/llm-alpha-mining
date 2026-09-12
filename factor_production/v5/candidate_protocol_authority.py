from __future__ import annotations

"""Immutable authorization for a run whose candidates use another protocol.

The V5 control protocol governs execution and remains the protocol stored on
the run.  A candidate research protocol may differ only when this exact,
hash-bound authority is present in both the workspace and repository.  The
scope deliberately does not authorize execution, evaluation, data access, or
campaign-ledger mutation.
"""

import re
from dataclasses import dataclass
from typing import Any, Mapping

from factor_production.v5.artifacts.hashing import hash_json


CANDIDATE_PROTOCOL_AUTHORITY_SCHEMA = "candidate-protocol-authority/v1"
CANDIDATE_PROTOCOL_AUTHORITY_SCOPE = (
    "candidate_registration",
    "candidate_repository_integrity",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CandidateProtocolAuthorityError(ValueError):
    pass


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CandidateProtocolAuthorityError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateProtocolAuthorityError(f"{name} must be a non-empty string")
    return value.strip()


def _canonical_relative(value: object, name: str) -> str:
    text = _nonempty(value, name)
    from pathlib import PurePosixPath

    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise CandidateProtocolAuthorityError(
            f"{name} must be canonical project-relative"
        )
    return text


@dataclass(frozen=True, slots=True)
class CandidateProtocolAuthority:
    authority_id: str
    run_id: str
    control_protocol_content_sha256: str
    candidate_protocol_content_sha256: str
    issuer_authority_path: str
    issuer_authority_file_sha256: str
    issuer_authority_content_sha256: str
    candidate_set_sha256: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "authority_id", _nonempty(self.authority_id, "authority_id")
        )
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "run_id"))
        if any(char in self.run_id for char in "/\\"):
            raise CandidateProtocolAuthorityError("run_id must be path-safe")
        for name in (
            "control_protocol_content_sha256",
            "candidate_protocol_content_sha256",
            "issuer_authority_file_sha256",
            "issuer_authority_content_sha256",
            "candidate_set_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "issuer_authority_path",
            _canonical_relative(self.issuer_authority_path, "issuer_authority_path"),
        )
        candidate_ids = tuple(
            _nonempty(candidate_id, "candidate_id")
            for candidate_id in self.candidate_ids
        )
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise CandidateProtocolAuthorityError(
                "candidate_ids must be a non-empty ordered unique sequence"
            )
        object.__setattr__(self, "candidate_ids", candidate_ids)
        if hash_json(list(candidate_ids)) != self.candidate_set_sha256:
            raise CandidateProtocolAuthorityError(
                "candidate_ids do not match candidate_set_sha256"
            )
        if (
            self.control_protocol_content_sha256
            == self.candidate_protocol_content_sha256
        ):
            raise CandidateProtocolAuthorityError(
                "candidate protocol authority is unnecessary for a single-protocol run"
            )

    def core_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_PROTOCOL_AUTHORITY_SCHEMA,
            "authority_id": self.authority_id,
            "run_id": self.run_id,
            "protocol_domains": {
                "control_protocol_content_sha256": self.control_protocol_content_sha256,
                "candidate_protocol_content_sha256": self.candidate_protocol_content_sha256,
            },
            "issuer_authority": {
                "path": self.issuer_authority_path,
                "file_sha256": self.issuer_authority_file_sha256,
                "content_sha256": self.issuer_authority_content_sha256,
                "candidate_set_sha256": self.candidate_set_sha256,
                "candidate_ids": list(self.candidate_ids),
            },
            "authorization_scope": list(CANDIDATE_PROTOCOL_AUTHORITY_SCOPE),
            "denied_capabilities": {
                "market_data_read": True,
                "signal_computation": True,
                "evaluation": True,
                "campaign_attempt_reservation": True,
                "campaign_ledger_write": True,
                "campaign_close": True,
            },
            "release_claims": {
                "data_mode": "degraded_research_only",
                "production_ready": False,
                "admission_claim": False,
            },
        }

    @property
    def content_hash(self) -> str:
        return hash_json(self.core_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.core_dict(), "content_sha256": self.content_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateProtocolAuthority":
        expected = {
            "schema_version",
            "authority_id",
            "run_id",
            "protocol_domains",
            "issuer_authority",
            "authorization_scope",
            "denied_capabilities",
            "release_claims",
            "content_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CandidateProtocolAuthorityError(
                "candidate protocol authority has an invalid schema"
            )
        if value["schema_version"] != CANDIDATE_PROTOCOL_AUTHORITY_SCHEMA:
            raise CandidateProtocolAuthorityError(
                "unsupported candidate protocol authority schema"
            )
        domains = value["protocol_domains"]
        issuer = value["issuer_authority"]
        if not isinstance(domains, Mapping) or set(domains) != {
            "control_protocol_content_sha256",
            "candidate_protocol_content_sha256",
        }:
            raise CandidateProtocolAuthorityError(
                "protocol domain binding has an invalid schema"
            )
        if not isinstance(issuer, Mapping) or set(issuer) != {
            "path",
            "file_sha256",
            "content_sha256",
            "candidate_set_sha256",
            "candidate_ids",
        }:
            raise CandidateProtocolAuthorityError(
                "issuer authority binding has an invalid schema"
            )
        if not isinstance(issuer["candidate_ids"], list) or not all(
            isinstance(candidate_id, str) for candidate_id in issuer["candidate_ids"]
        ):
            raise CandidateProtocolAuthorityError(
                "issuer candidate_ids must be a JSON string array"
            )
        if value["authorization_scope"] != list(CANDIDATE_PROTOCOL_AUTHORITY_SCOPE):
            raise CandidateProtocolAuthorityError(
                "candidate protocol authority scope differs"
            )
        if value["denied_capabilities"] != {
            "market_data_read": True,
            "signal_computation": True,
            "evaluation": True,
            "campaign_attempt_reservation": True,
            "campaign_ledger_write": True,
            "campaign_close": True,
        }:
            raise CandidateProtocolAuthorityError(
                "candidate protocol denied capabilities differ"
            )
        if value["release_claims"] != {
            "data_mode": "degraded_research_only",
            "production_ready": False,
            "admission_claim": False,
        }:
            raise CandidateProtocolAuthorityError(
                "candidate protocol research boundary differs"
            )
        item = cls(
            authority_id=value["authority_id"],
            run_id=value["run_id"],
            control_protocol_content_sha256=domains["control_protocol_content_sha256"],
            candidate_protocol_content_sha256=domains[
                "candidate_protocol_content_sha256"
            ],
            issuer_authority_path=issuer["path"],
            issuer_authority_file_sha256=issuer["file_sha256"],
            issuer_authority_content_sha256=issuer["content_sha256"],
            candidate_set_sha256=issuer["candidate_set_sha256"],
            candidate_ids=tuple(issuer["candidate_ids"]),
        )
        if value["content_sha256"] != item.content_hash:
            raise CandidateProtocolAuthorityError(
                "candidate protocol authority content hash mismatch"
            )
        return item


__all__ = [
    "CANDIDATE_PROTOCOL_AUTHORITY_SCHEMA",
    "CANDIDATE_PROTOCOL_AUTHORITY_SCOPE",
    "CandidateProtocolAuthority",
    "CandidateProtocolAuthorityError",
]
