from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from llm_alpha_mining.mining.artifacts.hashing import (
    canonical_json_bytes,
    hash_file,
    hash_json,
)
from llm_alpha_mining.mining.llm.domain import (
    LLMContractError,
    LLMRole,
    StructuredCallRequest,
    StructuredCallResponse,
    Usage,
    validate_provider_request_digest,
)
from llm_alpha_mining.mining.llm.safe_context import UnsafeLLMContext
from llm_alpha_mining.mining.llm.transport.base import (
    StructuredTransport,
    TransportError,
)
from llm_alpha_mining.mining.llm.validation import (
    request_descriptor_ids,
    validate_response_against_request,
)


class ReplayMiss(TransportError):
    pass


def _freeze_request_by_hash(
    request_by_hash: Mapping[str, StructuredCallRequest],
) -> Mapping[str, StructuredCallRequest]:
    """Copy and validate the externally frozen replay request registry."""

    if not isinstance(request_by_hash, Mapping):
        raise ValueError("frozen replay request_by_hash is required")
    copied: dict[str, StructuredCallRequest] = {}
    for key, request in request_by_hash.items():
        if type(key) is not str or type(request) is not StructuredCallRequest:
            raise ValueError("replay request registry fields must be exact")
        if key != request.request_hash:
            raise ValueError("replay request registry hash mismatch")
        if key in copied:
            raise ValueError("duplicate replay request registry hash")
        # Also rejects malformed reviewer descriptor collections up front.
        request_descriptor_ids(request)
        copied[key] = request
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True, init=False)
class ReplayEntry:
    """One response bound to an exact frozen request and descriptor-ID set."""

    request_hash: str
    role: LLMRole
    descriptor_ids: tuple[str, ...]
    output: Mapping[str, Any]
    usage: Usage
    model_id: str
    provider_request_id: str

    def __init__(
        self,
        *,
        request: StructuredCallRequest,
        output: Mapping[str, Any],
        usage: Usage,
        model_id: str = "offline-exact-replay/v1",
        provider_request_id: str = "replay",
    ) -> None:
        if type(request) is not StructuredCallRequest:
            raise ValueError("exact replay request is required")
        try:
            response = StructuredCallResponse(
                request_hash=request.request_hash,
                output=output,
                usage=usage,
                model_id=model_id,
                provider_request_id=provider_request_id,
            )
            validate_response_against_request(request, response)
            descriptor_ids = request_descriptor_ids(request)
        except LLMContractError:
            raise UnsafeLLMContext("unsafe replay response metadata") from None
        object.__setattr__(self, "request_hash", request.request_hash)
        object.__setattr__(self, "role", request.role)
        object.__setattr__(self, "descriptor_ids", descriptor_ids)
        object.__setattr__(self, "output", response.output)
        object.__setattr__(self, "usage", response.usage)
        object.__setattr__(self, "model_id", response.model_id)
        object.__setattr__(self, "provider_request_id", response.provider_request_id)

    @classmethod
    def from_response(
        cls,
        response: StructuredCallResponse,
        *,
        request: StructuredCallRequest,
    ) -> "ReplayEntry":
        if type(response) is not StructuredCallResponse:
            raise ValueError("exact structured response is required")
        if response.request_hash != request.request_hash:
            raise ValueError("replay response request hash mismatch")
        return cls(
            request=request,
            output=response.output,
            usage=response.usage,
            model_id=response.model_id,
            provider_request_id=response.provider_request_id,
        )

    def _to_response(self) -> StructuredCallResponse:
        return StructuredCallResponse(
            request_hash=self.request_hash,
            output=self.output,
            usage=self.usage,
            model_id=self.model_id,
            provider_request_id=validate_provider_request_digest(
                self.provider_request_id
            ),
        )

    def validate_against(self, request: StructuredCallRequest) -> None:
        if self.request_hash != request.request_hash or self.role is not request.role:
            raise ValueError("replay entry request binding mismatch")
        expected_ids = request_descriptor_ids(request)
        if tuple(self.descriptor_ids) != expected_ids:
            raise ValueError("replay entry descriptor binding mismatch")
        validate_response_against_request(request, self._to_response())

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": "structured-llm-replay-entry/v3",
            "request_hash": self.request_hash,
            "role": self.role.value,
            "descriptor_ids": list(self.descriptor_ids),
            "output": self.output,
            "usage": self.usage.to_dict(),
            "model_id": self.model_id,
            "provider_request_id": self.provider_request_id,
        }
        if include_hash:
            result["entry_hash"] = self.content_hash
        return result

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        request: StructuredCallRequest,
    ) -> "ReplayEntry":
        expected = {
            "schema_version",
            "request_hash",
            "role",
            "descriptor_ids",
            "output",
            "usage",
            "model_id",
            "provider_request_id",
            "entry_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("replay entry fields must be exact")
        if value["schema_version"] != "structured-llm-replay-entry/v3":
            raise ValueError("unsupported replay entry schema")
        if (
            value["request_hash"] != request.request_hash
            or value["role"] != request.role.value
        ):
            raise ValueError("persisted replay request binding mismatch")
        if not isinstance(value["descriptor_ids"], list) or not all(
            isinstance(item, str) for item in value["descriptor_ids"]
        ):
            raise ValueError("persisted replay descriptor IDs must be a list")
        try:
            provider_request_id = validate_provider_request_digest(
                value["provider_request_id"]
            )
        except LLMContractError:
            raise ValueError(
                "persisted replay provider request ID must be hashed"
            ) from None
        entry = cls(
            request=request,
            output=value["output"],
            usage=Usage.from_dict(value["usage"]),
            model_id=value["model_id"],
            provider_request_id=provider_request_id,
        )
        if tuple(value["descriptor_ids"]) != entry.descriptor_ids:
            raise ValueError("persisted replay descriptor binding mismatch")
        if value["entry_hash"] != entry.content_hash:
            raise ValueError("replay entry hash mismatch")
        return entry


def _validate_complete_entry_set(
    entries: Iterable[ReplayEntry],
    request_by_hash: Mapping[str, StructuredCallRequest],
) -> tuple[tuple[ReplayEntry, ...], Mapping[str, StructuredCallRequest]]:
    frozen_requests = _freeze_request_by_hash(request_by_hash)
    checked_entries = tuple(entries)
    by_hash: dict[str, ReplayEntry] = {}
    for entry in checked_entries:
        if type(entry) is not ReplayEntry:
            raise ValueError("replay cassette accepts exact ReplayEntry objects only")
        if entry.request_hash in by_hash:
            raise ValueError("duplicate replay request hash")
        request = frozen_requests.get(entry.request_hash)
        if request is None:
            raise ValueError("replay entry has no frozen request")
        entry.validate_against(request)
        by_hash[entry.request_hash] = entry
    if set(by_hash) != set(frozen_requests):
        raise ValueError("replay entries and frozen requests must match exactly")
    return checked_entries, frozen_requests


def write_replay_cassette(
    entries: Iterable[ReplayEntry],
    path: str | Path,
    *,
    request_by_hash: Mapping[str, StructuredCallRequest],
) -> Path:
    checked_entries, _ = _validate_complete_entry_set(entries, request_by_hash)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(
        canonical_json_bytes(entry.to_dict()) + b"\n" for entry in checked_entries
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return destination


def load_replay_cassette(
    path: str | Path,
    *,
    request_by_hash: Mapping[str, StructuredCallRequest],
    expected_file_hash: str | None = None,
) -> tuple[ReplayEntry, ...]:
    source = Path(path)
    frozen_requests = _freeze_request_by_hash(request_by_hash)
    if expected_file_hash is None:
        raise ValueError("external expected replay cassette file hash is required")
    if hash_file(source) != expected_file_hash:
        raise ValueError("replay cassette file hash mismatch")
    entries: list[ReplayEntry] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            request_hash = (
                payload.get("request_hash") if isinstance(payload, Mapping) else None
            )
            request = (
                frozen_requests.get(request_hash)
                if isinstance(request_hash, str)
                else None
            )
            if request is None or request_hash in seen:
                raise ValueError("unbound or duplicate persisted replay request")
            entries.append(ReplayEntry.from_dict(payload, request=request))
            seen.add(request_hash)
        except Exception:
            # Cassette contents are untrusted persistence.  Never reflect a
            # provider-controlled value through exception text or chaining.
            raise ValueError(f"invalid replay cassette line {line_number}") from None
    checked, _ = _validate_complete_entry_set(entries, frozen_requests)
    return checked


class ExactReplayTransport(StructuredTransport):
    """Serve only exact requests from a fully request-bound frozen cassette."""

    name = "exact-replay"

    def __init__(
        self,
        entries: Iterable[ReplayEntry],
        *,
        request_by_hash: Mapping[str, StructuredCallRequest],
    ) -> None:
        checked_entries, frozen_requests = _validate_complete_entry_set(
            entries, request_by_hash
        )
        self._entries = {entry.request_hash: entry for entry in checked_entries}
        self._request_by_hash = frozen_requests
        self.call_count = 0

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        request_by_hash: Mapping[str, StructuredCallRequest],
        expected_file_hash: str | None = None,
    ) -> "ExactReplayTransport":
        if expected_file_hash is None:
            raise ValueError("external expected replay cassette file hash is required")
        return cls(
            load_replay_cassette(
                path,
                request_by_hash=request_by_hash,
                expected_file_hash=expected_file_hash,
            ),
            request_by_hash=request_by_hash,
        )

    def complete(
        self,
        request: StructuredCallRequest,
        *,
        idempotency_key: str,
        timeout_seconds: float,
    ) -> StructuredCallResponse:
        self.call_count += 1
        entry = self._entries.get(request.request_hash)
        frozen_request = self._request_by_hash.get(request.request_hash)
        if (
            entry is None
            or frozen_request is None
            or entry.role is not request.role
            or frozen_request.to_dict() != request.to_dict()
        ):
            raise ReplayMiss(
                "exact replay request miss",
                retryable=False,
                dispatch_unknown=False,
                definitely_unsent=True,
            )
        try:
            # Revalidate mutable in-memory state against the actual call.  This
            # final gate is intentionally repeated immediately before return.
            entry.validate_against(request)
            response = entry._to_response()
            validate_response_against_request(request, response)
        except Exception:
            raise ReplayMiss(
                "exact replay response binding failed",
                retryable=False,
                dispatch_unknown=False,
                definitely_unsent=True,
            ) from None
        return response


__all__ = [
    "ExactReplayTransport",
    "ReplayEntry",
    "ReplayMiss",
    "load_replay_cassette",
    "write_replay_cassette",
]
