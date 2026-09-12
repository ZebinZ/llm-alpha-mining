from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from llm_alpha_mining.research.core.hashing import hash_json, require_sha256


@dataclass(frozen=True, slots=True)
class ApprovalGrant:
    experiment_spec_hash: str
    action: str
    scope: str
    artifact_hash: str
    actor: str
    actor_role: str
    signature_hash: str
    nonce: str
    issued_at: str
    expires_at: str
    schema_version: str = "approval-grant/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "approval-grant/v1":
            raise ValueError("unsupported ApprovalGrant schema")
        for name in (
            "experiment_spec_hash",
            "artifact_hash",
            "signature_hash",
        ):
            require_sha256(str(getattr(self, name)), name=f"approval {name}")
        for name in ("action", "scope", "actor", "actor_role", "nonce"):
            value = str(getattr(self, name))
            if not value.strip() or len(value) > 256:
                raise ValueError(f"approval {name} must be non-empty and bounded")
        issued = pd.Timestamp(self.issued_at)
        expires = pd.Timestamp(self.expires_at)
        if issued.tzinfo is None or expires.tzinfo is None:
            raise ValueError("approval timestamps must be timezone-aware")
        if expires <= issued:
            raise ValueError("approval expiry must follow issue time")
        object.__setattr__(self, "issued_at", issued.isoformat())
        object.__setattr__(self, "expires_at", expires.isoformat())

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_spec_hash": self.experiment_spec_hash,
            "action": self.action,
            "scope": self.scope,
            "artifact_hash": self.artifact_hash,
            "actor": self.actor,
            "actor_role": self.actor_role,
            "signature_hash": self.signature_hash,
            "nonce": self.nonce,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class ApprovalRevocation:
    approval_hash: str
    revoked_at: str
    revoked_by: str
    reason_code: str

    def __post_init__(self) -> None:
        require_sha256(self.approval_hash, name="revocation approval_hash")
        timestamp = pd.Timestamp(self.revoked_at)
        if timestamp.tzinfo is None:
            raise ValueError("revocation timestamp must be timezone-aware")
        object.__setattr__(self, "revoked_at", timestamp.isoformat())
        if not self.revoked_by.strip() or not self.reason_code.strip():
            raise ValueError("revocation actor and reason are required")


__all__ = ["ApprovalGrant", "ApprovalRevocation"]
