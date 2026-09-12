"""Content-addressed freeze receipt for the formal R0 candidate family."""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import cast

from alpha_research.core.hashing import (
    canonical_json_bytes,
    hash_json,
    require_sha256,
)


_SCHEMA_VERSION = "r0-candidate-family-freeze-receipt/v1"


def r0_operator_implementation_hash(
    *,
    operator_registry_version: str,
    operator_registry_factory_name: str,
    operator_registry_digest: str,
    factor_engine_execution_policy_hash: str,
    operator_source_closure_hash: str,
) -> str:
    """Bind declarative registry semantics to their exact Python closure."""

    version = _canonical_text(
        operator_registry_version,
        name="operator_registry_version",
    )
    factory_name = _canonical_text(
        operator_registry_factory_name,
        name="operator_registry_factory_name",
    )
    for name, digest in (
        ("operator_registry_digest", operator_registry_digest),
        (
            "factor_engine_execution_policy_hash",
            factor_engine_execution_policy_hash,
        ),
        ("operator_source_closure_hash", operator_source_closure_hash),
    ):
        require_sha256(digest, name=f"R0 freeze {name}")
    return cast(
        str,
        hash_json(
            {
                "schema_version": "r0-operator-implementation-identity/v1",
                "operator_registry_version": version,
                "operator_registry_factory_name": factory_name,
                "operator_registry_digest": operator_registry_digest,
                "factor_engine_execution_policy_hash": (
                    factor_engine_execution_policy_hash
                ),
                "operator_source_closure_hash": operator_source_closure_hash,
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class R0CandidateFamilyFreezeReceipt:
    """Immutable boundary between adaptive search and terminal evaluation."""

    search_design_authority_hash: str
    protocol_hash: str
    candidate_family_hash: str
    candidate_ids: tuple[str, ...]
    factor_spec_hashes: tuple[str, ...]
    hypothesis_hashes: tuple[str, ...]
    declared_signs: tuple[int, ...]
    candidate_view_binding_hashes: tuple[str, ...]
    common_prediction_data_hash: str
    preterminal_research_history_hash: str
    operator_registry_version: str
    operator_registry_factory_name: str
    operator_registry_digest: str
    factor_engine_execution_policy_hash: str
    operator_source_closure_hash: str
    operator_implementation_hash: str
    code_snapshot_hash: str
    runtime_environment_hash: str
    research_only: bool = True
    production_ready: bool = False
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported R0 candidate family freeze schema")
        if self.research_only is not True or self.production_ready is not False:
            raise ValueError("R0 candidate family freeze must remain research-only")
        candidate_ids = tuple(
            _canonical_text(candidate_id, name="candidate_id")
            for candidate_id in self.candidate_ids
        )
        factor_hashes = tuple(self.factor_spec_hashes)
        hypothesis_hashes = tuple(self.hypothesis_hashes)
        signs = tuple(self.declared_signs)
        binding_hashes = tuple(self.candidate_view_binding_hashes)
        if (
            type(self.candidate_ids) is not tuple
            or type(self.factor_spec_hashes) is not tuple
            or type(self.hypothesis_hashes) is not tuple
            or type(self.declared_signs) is not tuple
            or type(self.candidate_view_binding_hashes) is not tuple
            or not candidate_ids
            or len(
                {
                    len(candidate_ids),
                    len(factor_hashes),
                    len(hypothesis_hashes),
                    len(signs),
                    len(binding_hashes),
                }
            )
            != 1
        ):
            raise ValueError(
                "R0 freeze candidate fields must have the same nonzero length"
            )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("R0 freeze candidate ids must be unique")
        if len(set(factor_hashes)) != len(factor_hashes):
            raise ValueError("R0 freeze factor spec hashes must be unique")
        if any(type(sign) is not int or sign not in {-1, 1} for sign in signs):
            raise ValueError("R0 freeze declared signs must be exact -1 or 1")
        for offset, digest in enumerate(factor_hashes):
            require_sha256(
                digest,
                name=f"R0 freeze factor spec hash:{offset}",
            )
        for offset, digest in enumerate(hypothesis_hashes):
            require_sha256(
                digest,
                name=f"R0 freeze hypothesis hash:{offset}",
            )
        for offset, digest in enumerate(binding_hashes):
            require_sha256(
                digest,
                name=f"R0 freeze candidate view binding hash:{offset}",
            )
        for name in (
            "search_design_authority_hash",
            "protocol_hash",
            "candidate_family_hash",
            "common_prediction_data_hash",
            "preterminal_research_history_hash",
            "operator_registry_digest",
            "factor_engine_execution_policy_hash",
            "operator_source_closure_hash",
            "operator_implementation_hash",
            "code_snapshot_hash",
            "runtime_environment_hash",
        ):
            require_sha256(
                str(getattr(self, name)),
                name=f"R0 freeze {name}",
            )
        version = _canonical_text(
            self.operator_registry_version,
            name="operator_registry_version",
        )
        factory_name = _canonical_text(
            self.operator_registry_factory_name,
            name="operator_registry_factory_name",
        )
        expected_implementation = r0_operator_implementation_hash(
            operator_registry_version=version,
            operator_registry_factory_name=factory_name,
            operator_registry_digest=self.operator_registry_digest,
            factor_engine_execution_policy_hash=(
                self.factor_engine_execution_policy_hash
            ),
            operator_source_closure_hash=self.operator_source_closure_hash,
        )
        if not hmac.compare_digest(
            self.operator_implementation_hash,
            expected_implementation,
        ):
            raise ValueError("R0 freeze operator implementation hash differs")
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "factor_spec_hashes", factor_hashes)
        object.__setattr__(self, "hypothesis_hashes", hypothesis_hashes)
        object.__setattr__(self, "declared_signs", signs)
        object.__setattr__(
            self,
            "candidate_view_binding_hashes",
            binding_hashes,
        )
        object.__setattr__(self, "operator_registry_version", version)
        object.__setattr__(
            self,
            "operator_registry_factory_name",
            factory_name,
        )

    @property
    def content_hash(self) -> str:
        return cast(str, hash_json(self._payload_dict()))

    def _payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "search_design_authority_hash": self.search_design_authority_hash,
            "protocol_hash": self.protocol_hash,
            "candidate_family_hash": self.candidate_family_hash,
            "candidate_ids": list(self.candidate_ids),
            "factor_spec_hashes": list(self.factor_spec_hashes),
            "hypothesis_hashes": list(self.hypothesis_hashes),
            "declared_signs": list(self.declared_signs),
            "candidate_view_binding_hashes": list(
                self.candidate_view_binding_hashes
            ),
            "common_prediction_data_hash": self.common_prediction_data_hash,
            "preterminal_research_history_hash": (
                self.preterminal_research_history_hash
            ),
            "operator_registry_version": self.operator_registry_version,
            "operator_registry_factory_name": (
                self.operator_registry_factory_name
            ),
            "operator_registry_digest": self.operator_registry_digest,
            "factor_engine_execution_policy_hash": (
                self.factor_engine_execution_policy_hash
            ),
            "operator_source_closure_hash": self.operator_source_closure_hash,
            "operator_implementation_hash": self.operator_implementation_hash,
            "code_snapshot_hash": self.code_snapshot_hash,
            "runtime_environment_hash": self.runtime_environment_hash,
            "research_only": self.research_only,
            "production_ready": self.production_ready,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload_dict()
        return {**payload, "content_hash": self.content_hash}

    def to_wire_bytes(self) -> bytes:
        return cast(bytes, canonical_json_bytes(self.to_dict()) + b"\n")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "R0CandidateFamilyFreezeReceipt":
        expected = {field.name for field in fields(cls)} | {"content_hash"}
        if set(value) != expected:
            raise ValueError("R0 candidate family freeze wire fields differ")
        content_hash = _wire_text(value["content_hash"], name="content_hash")
        require_sha256(content_hash, name="R0 freeze wire content_hash")
        result = cls(
            search_design_authority_hash=_wire_text(
                value["search_design_authority_hash"],
                name="search_design_authority_hash",
            ),
            protocol_hash=_wire_text(value["protocol_hash"], name="protocol_hash"),
            candidate_family_hash=_wire_text(
                value["candidate_family_hash"],
                name="candidate_family_hash",
            ),
            candidate_ids=_wire_text_tuple(
                value["candidate_ids"],
                name="candidate_ids",
            ),
            factor_spec_hashes=_wire_text_tuple(
                value["factor_spec_hashes"],
                name="factor_spec_hashes",
            ),
            hypothesis_hashes=_wire_text_tuple(
                value["hypothesis_hashes"],
                name="hypothesis_hashes",
            ),
            declared_signs=_wire_integer_tuple(
                value["declared_signs"],
                name="declared_signs",
            ),
            candidate_view_binding_hashes=_wire_text_tuple(
                value["candidate_view_binding_hashes"],
                name="candidate_view_binding_hashes",
            ),
            common_prediction_data_hash=_wire_text(
                value["common_prediction_data_hash"],
                name="common_prediction_data_hash",
            ),
            preterminal_research_history_hash=_wire_text(
                value["preterminal_research_history_hash"],
                name="preterminal_research_history_hash",
            ),
            operator_registry_version=_wire_text(
                value["operator_registry_version"],
                name="operator_registry_version",
            ),
            operator_registry_factory_name=_wire_text(
                value["operator_registry_factory_name"],
                name="operator_registry_factory_name",
            ),
            operator_registry_digest=_wire_text(
                value["operator_registry_digest"],
                name="operator_registry_digest",
            ),
            factor_engine_execution_policy_hash=_wire_text(
                value["factor_engine_execution_policy_hash"],
                name="factor_engine_execution_policy_hash",
            ),
            operator_source_closure_hash=_wire_text(
                value["operator_source_closure_hash"],
                name="operator_source_closure_hash",
            ),
            operator_implementation_hash=_wire_text(
                value["operator_implementation_hash"],
                name="operator_implementation_hash",
            ),
            code_snapshot_hash=_wire_text(
                value["code_snapshot_hash"],
                name="code_snapshot_hash",
            ),
            runtime_environment_hash=_wire_text(
                value["runtime_environment_hash"],
                name="runtime_environment_hash",
            ),
            research_only=_wire_boolean(
                value["research_only"],
                name="research_only",
            ),
            production_ready=_wire_boolean(
                value["production_ready"],
                name="production_ready",
            ),
            schema_version=_wire_text(
                value["schema_version"],
                name="schema_version",
            ),
        )
        if not hmac.compare_digest(result.content_hash, content_hash):
            raise ValueError("R0 candidate family freeze content_hash mismatch")
        return result

    @classmethod
    def from_wire_bytes(
        cls,
        payload: bytes,
    ) -> "R0CandidateFamilyFreezeReceipt":
        if type(payload) is not bytes:
            raise TypeError("R0 candidate family freeze wire payload must be bytes")
        try:
            decoded = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("R0 candidate family freeze wire is invalid JSON") from error
        if not isinstance(decoded, dict):
            raise TypeError("R0 candidate family freeze wire root must be an object")
        if payload != canonical_json_bytes(decoded) + b"\n":
            raise ValueError("R0 candidate family freeze wire is not canonical")
        return cls.from_mapping(decoded)


def _canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be canonical and non-empty")
    return text


def _wire_text(value: object, *, name: str) -> str:
    return _canonical_text(value, name=name)


def _wire_text_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return tuple(
        _wire_text(item, name=f"{name}:{offset}")
        for offset, item in enumerate(value)
    )


def _wire_integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        type(item) is not int for item in value
    ):
        raise TypeError(f"{name} must be an integer list")
    return tuple(int(item) for item in value)


def _wire_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key:{key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant:{value}")


__all__ = [
    "R0CandidateFamilyFreezeReceipt",
    "r0_operator_implementation_hash",
]
