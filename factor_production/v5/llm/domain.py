from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from factor_production.v5.artifacts.hashing import hash_json


_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROVIDER_REQUEST_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_PROVIDER_REQUEST_ID_DOMAIN = b"structured-llm/provider-request-id/v1\x00"
_OPERATION_ID = re.compile(r"llm-[0-9a-f]{40}")
_SAFE_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9._/-]{0,63}")
_CREDENTIAL_PREFIX = re.compile(
    r"(?<![A-Za-z0-9])(?:sk|ghp|gho|github_pat|xox[abprs])[-_][A-Za-z0-9_-]{6,}|"
    r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{12,}|"
    r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{6,}[.][A-Za-z0-9_-]{6,}[.][A-Za-z0-9_-]{6,}",
    flags=re.IGNORECASE,
)
_SENSITIVE_METADATA_TOKENS = (
    "apikey",
    "bearer",
    "accesstoken",
    "refreshtoken",
    "clientsecret",
    "privatesecret",
    "privatekey",
    "credential",
    "password",
    "secret",
    "token",
    "teacher",
    "official",
    "holdout",
)


class LLMContractError(ValueError):
    """A structured LLM request or response broke the frozen wire contract."""


class _ProviderRequestDigest(str):
    """Internal marker for an already irreversibly transformed provider ID.

    A plain string is *always* treated as live provider input, even when it
    happens to contain 64 hexadecimal characters.  Only deserializers in this
    package create this marker after validating the persisted wire form.
    """

    def __new__(cls, value: str) -> "_ProviderRequestDigest":
        if not isinstance(value, str) or _PROVIDER_REQUEST_DIGEST.fullmatch(value) is None:
            raise LLMContractError(
                "persisted provider_request_id must use sha256:<digest>"
            )
        return str.__new__(cls, value)


class LLMRole(str, Enum):
    PROPOSER = "proposer"
    CRITIC = "critic"
    RISK = "risk"
    ARBITER = "arbiter"


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LLMContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _metadata_compact(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    # Keep only letters and digits.  In particular, underscores, punctuation,
    # whitespace and zero-width controls cannot split a blocked token.
    return "".join(character for character in normalized if character.isalnum())


def _reject_sensitive_metadata(value: str, name: str) -> None:
    compact = _metadata_compact(value)
    if any(token in compact for token in _SENSITIVE_METADATA_TOKENS):
        raise LLMContractError(f"{name} contains a forbidden metadata marker")
    if _CREDENTIAL_PREFIX.search(value):
        raise LLMContractError(f"{name} resembles a credential")


def validate_safe_model_id(value: str) -> str:
    """Validate the non-secret model alias used in the persisted wire form.

    The executor additionally requires membership in the policy's closed model
    allow-list.  This syntactic gate prevents provider metadata from becoming
    an unbounded free-text persistence channel before that policy check.
    """

    if not isinstance(value, str) or _SAFE_MODEL_ID.fullmatch(value) is None:
        raise LLMContractError("model_id must be a lowercase safe identifier")
    _reject_sensitive_metadata(value, "model_id")
    return value


def hash_provider_request_id(value: str) -> str:
    """Return the only representation of a live provider request ID we persist.

    The domain-separated preimage and versioned wire prefix make this digest
    unambiguous with respect to unrelated SHA-256 fields.  In particular, a
    provider-controlled 64-hex string is hashed again rather than being
    mistaken for trusted persistence.
    """

    if isinstance(value, _ProviderRequestDigest):
        return value
    if not isinstance(value, str) or not value or len(value) > 512:
        raise LLMContractError("provider_request_id must be a bounded non-empty string")
    _reject_sensitive_metadata(value, "provider_request_id")
    digest = hashlib.sha256(
        _PROVIDER_REQUEST_ID_DOMAIN + value.encode("utf-8")
    ).hexdigest()
    return _ProviderRequestDigest(f"sha256:{digest}")


def validate_provider_request_digest(value: str) -> str:
    return _ProviderRequestDigest(value)


def _json_copy(value: Any, *, path: str = "$") -> Any:
    """Return an immutable JSON-compatible copy and reject exotic numbers."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LLMContractError(f"non-finite number at {path}")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LLMContractError(f"non-string object key at {path}")
            copied[key] = _json_copy(item, path=f"{path}.{key}")
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_json_copy(item, path=f"{path}[]") for item in value)
    raise LLMContractError(f"non-JSON value at {path}: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cost_microusd: int

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cost_microusd"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise LLMContractError(f"{name} must be a non-negative integer")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_microusd": self.cost_microusd,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Usage":
        expected = {"input_tokens", "output_tokens", "cost_microusd"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise LLMContractError("usage fields must be exact")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class StructuredCallRequest:
    operation_id: str
    role: LLMRole | str
    prompt_template_id: str
    prompt_hash: str
    schema_id: str
    schema_hash: str
    policy_hash: str
    context: Mapping[str, Any]
    max_output_tokens: int
    max_cost_microusd: int
    wire_version: str = "structured-llm-request/v1"

    def __post_init__(self) -> None:
        if self.wire_version != "structured-llm-request/v1":
            raise LLMContractError("unsupported structured request wire version")
        if not isinstance(self.operation_id, str) or _OPERATION_ID.fullmatch(self.operation_id) is None:
            raise LLMContractError("operation_id must be an opaque llm SHA-256 prefix")
        try:
            role = LLMRole(self.role)
        except ValueError:
            raise LLMContractError("unsupported LLM role") from None
        object.__setattr__(self, "role", role)
        if not isinstance(self.prompt_template_id, str) or not self.prompt_template_id.strip():
            raise LLMContractError("prompt_template_id must not be empty")
        if not isinstance(self.schema_id, str) or not self.schema_id.strip():
            raise LLMContractError("schema_id must not be empty")
        for name in ("prompt_hash", "schema_hash", "policy_hash"):
            _digest(getattr(self, name), name)
        for name in ("max_output_tokens", "max_cost_microusd"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise LLMContractError(f"{name} must be a positive integer")
        if not isinstance(self.context, Mapping):
            raise LLMContractError("context must be an object")
        object.__setattr__(self, "context", _json_copy(self.context, path="$.context"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "wire_version": self.wire_version,
            "operation_id": self.operation_id,
            "role": self.role.value,
            "prompt_template_id": self.prompt_template_id,
            "prompt_hash": self.prompt_hash,
            "schema_id": self.schema_id,
            "schema_hash": self.schema_hash,
            "policy_hash": self.policy_hash,
            "context": thaw_json(self.context),
            "max_output_tokens": self.max_output_tokens,
            "max_cost_microusd": self.max_cost_microusd,
        }

    @property
    def request_hash(self) -> str:
        return hash_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class StructuredCallResponse:
    request_hash: str
    output: Mapping[str, Any]
    usage: Usage
    model_id: str
    provider_request_id: str

    def __post_init__(self) -> None:
        _digest(self.request_hash, "request_hash")
        if not isinstance(self.output, Mapping):
            raise LLMContractError("structured output must be an object")
        if not isinstance(self.usage, Usage):
            raise LLMContractError("usage must be a Usage instance")
        object.__setattr__(self, "model_id", validate_safe_model_id(self.model_id))
        object.__setattr__(
            self,
            "provider_request_id",
            hash_provider_request_id(self.provider_request_id),
        )
        object.__setattr__(self, "output", _json_copy(self.output, path="$.output"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_hash": self.request_hash,
            "output": thaw_json(self.output),
            "usage": self.usage.to_dict(),
            "model_id": self.model_id,
            "provider_request_id": self.provider_request_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StructuredCallResponse":
        expected = {
            "request_hash",
            "output",
            "usage",
            "model_id",
            "provider_request_id",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise LLMContractError("response fields must be exact")
        provider_request_id = validate_provider_request_digest(
            value["provider_request_id"]
        )
        return cls(
            request_hash=value["request_hash"],
            output=value["output"],
            usage=Usage.from_dict(value["usage"]),
            model_id=value["model_id"],
            provider_request_id=provider_request_id,
        )
