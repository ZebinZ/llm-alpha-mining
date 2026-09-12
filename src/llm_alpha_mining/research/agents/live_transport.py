from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol
from urllib.parse import urlsplit

from llm_alpha_mining.research.core.hashing import canonical_json_bytes, hash_json
from llm_alpha_mining.mining.llm.domain import (
    StructuredCallRequest,
    StructuredCallResponse,
    Usage,
    validate_safe_model_id,
)
from llm_alpha_mining.mining.llm.policy import LLMPolicy
from llm_alpha_mining.mining.llm.schemas import ROLE_SCHEMAS, schema_hash
from llm_alpha_mining.mining.llm.transport.base import (
    StructuredTransport,
    TransportError,
    TransportTimeout,
)


_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_PROVIDER_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class BoundedHTTPClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> HTTPResponse: ...


class UrllibBoundedHTTPClient:
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> HTTPResponse:
        request = urllib.request.Request(
            url, data=body, headers=dict(headers), method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read(maximum_response_bytes + 1)
                status = int(response.status)
                response_headers = dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            payload = exc.read(maximum_response_bytes + 1)
            status = int(exc.code)
            response_headers = dict(exc.headers.items()) if exc.headers else {}
        except (TimeoutError, urllib.error.URLError, OSError):
            raise TransportTimeout(
                "live transport network outcome is unknown"
            ) from None
        if len(payload) > maximum_response_bytes:
            raise TransportError(
                "live transport response exceeds byte limit", dispatch_unknown=True
            )
        return HTTPResponse(
            status=status,
            headers=MappingProxyType(response_headers),
            body=payload,
        )


@dataclass(frozen=True, slots=True)
class LiveProviderConfig:
    base_url: str
    allowed_hosts: tuple[str, ...]
    api_key_env: str
    provider_model_id: str
    accepted_provider_model_ids: tuple[str, ...]
    wire_model_id: str
    input_cost_microusd_per_million: int
    output_cost_microusd_per_million: int
    maximum_request_bytes: int = 1_000_000
    maximum_response_bytes: int = 2_000_000
    temperature: float = 0.0
    schema_version: str = "live-provider-config/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "live-provider-config/v1":
            raise ValueError("unsupported LiveProviderConfig schema")
        parsed = urlsplit(self.base_url)
        hosts = tuple(sorted(set(host.lower() for host in self.allowed_hosts)))
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.hostname.lower() not in hosts
        ):
            raise ValueError(
                "live provider base_url is outside the HTTPS host allowlist"
            )
        if not hosts or len(hosts) != len(self.allowed_hosts):
            raise ValueError("live provider allowed_hosts must be non-empty and unique")
        if not _ENV_NAME.fullmatch(self.api_key_env):
            raise ValueError("live provider api_key_env is invalid")
        model_ids = tuple(self.accepted_provider_model_ids)
        if (
            not _PROVIDER_MODEL.fullmatch(self.provider_model_id)
            or not model_ids
            or len(set(model_ids)) != len(model_ids)
            or any(not _PROVIDER_MODEL.fullmatch(value) for value in model_ids)
        ):
            raise ValueError("live provider model allowlist is invalid")
        if self.provider_model_id not in model_ids:
            raise ValueError("requested provider model is not accepted")
        validate_safe_model_id(self.wire_model_id)
        for name in (
            "input_cost_microusd_per_million",
            "output_cost_microusd_per_million",
            "maximum_request_bytes",
            "maximum_response_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"live provider {name} must be positive")
        if (
            not isinstance(self.temperature, (int, float))
            or not 0 <= self.temperature <= 1
        ):
            raise ValueError("live provider temperature must lie in [0,1]")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        object.__setattr__(self, "allowed_hosts", hosts)
        object.__setattr__(self, "accepted_provider_model_ids", model_ids)

    @property
    def content_hash(self) -> str:
        return hash_json(
            {
                "schema_version": self.schema_version,
                "base_url": self.base_url,
                "allowed_hosts": list(self.allowed_hosts),
                "api_key_env": self.api_key_env,
                "provider_model_id": self.provider_model_id,
                "accepted_provider_model_ids": list(self.accepted_provider_model_ids),
                "wire_model_id": self.wire_model_id,
                "input_cost_microusd_per_million": self.input_cost_microusd_per_million,
                "output_cost_microusd_per_million": self.output_cost_microusd_per_million,
                "maximum_request_bytes": self.maximum_request_bytes,
                "maximum_response_bytes": self.maximum_response_bytes,
                "temperature": float(self.temperature),
            }
        )


class LiveStructuredTransport(StructuredTransport):
    """Optional bounded live adapter for an OpenAI-compatible JSON-schema API."""

    name = "live-structured-https"

    def __init__(
        self,
        *,
        config: LiveProviderConfig,
        policy: LLMPolicy,
        client: BoundedHTTPClient | None = None,
    ) -> None:
        if config.wire_model_id not in policy.allowed_model_ids:
            raise ValueError("live wire model is outside LLMPolicy")
        self.config = config
        self.policy = policy
        self.client = client or UrllibBoundedHTTPClient()

    @property
    def authority_hash(self) -> str:
        return hash_json(
            {
                "implementation": "live-structured-https/v1",
                "provider_config_hash": self.config.content_hash,
                "llm_policy_hash": self.policy.content_hash,
            }
        )

    def complete(
        self,
        request: StructuredCallRequest,
        *,
        idempotency_key: str,
        timeout_seconds: float,
    ) -> StructuredCallResponse:
        self._validate_request(request, idempotency_key, timeout_seconds)
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise TransportError(
                "live provider credential is unavailable", definitely_unsent=True
            )
        if not 8 <= len(api_key) <= 512 or "\r" in api_key or "\n" in api_key:
            raise TransportError(
                "live provider credential format is invalid", definitely_unsent=True
            )
        body = canonical_json_bytes(
            {
                "model": self.config.provider_model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": self.policy.prompts[request.role],
                    },
                    {
                        "role": "user",
                        "content": canonical_json_bytes(
                            {
                                "operation_id": request.operation_id,
                                "context": request.to_dict()["context"],
                            }
                        ).decode("utf-8"),
                    },
                ],
                "temperature": float(self.config.temperature),
                "max_tokens": request.max_output_tokens,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.schema_id.replace("/", "_"),
                        "strict": True,
                        "schema": ROLE_SCHEMAS[request.role],
                    },
                },
            }
        )
        if len(body) > self.config.maximum_request_bytes:
            raise TransportError(
                "live provider request exceeds byte limit", definitely_unsent=True
            )
        response = self.client.post(
            self.config.base_url + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
                "User-Agent": "alpha-research-framework/0.1",
            },
            body=body,
            timeout_seconds=timeout_seconds,
            maximum_response_bytes=self.config.maximum_response_bytes,
        )
        if (
            not isinstance(response.status, int)
            or len(response.body) > self.config.maximum_response_bytes
        ):
            raise TransportError(
                "live provider response violates bounds", dispatch_unknown=True
            )
        if response.status == 429:
            raise TransportError(
                "live provider throttled before generation", retryable=True
            )
        if response.status >= 500:
            raise TransportError(
                "live provider server outcome is unknown", dispatch_unknown=True
            )
        if response.status < 200 or response.status >= 300:
            raise TransportError(
                "live provider rejected request", definitely_unsent=True
            )
        try:
            value = json.loads(response.body.decode("utf-8"))
            if not isinstance(value, dict):
                raise TypeError
            provider_request_id = value["id"]
            provider_model_id = value["model"]
            choices = value["choices"]
            usage = value["usage"]
            if (
                not isinstance(provider_request_id, str)
                or provider_model_id not in self.config.accepted_provider_model_ids
                or not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(usage, dict)
            ):
                raise TypeError
            choice = choices[0]
            message = choice["message"]
            if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
                raise TypeError
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError
            output = json.loads(content)
            if not isinstance(output, dict):
                raise TypeError
            input_tokens = _non_negative_int(usage, "prompt_tokens")
            output_tokens = _non_negative_int(usage, "completion_tokens")
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            raise TransportError(
                "live provider response is malformed", dispatch_unknown=True
            ) from None
        cost = _ceil_million(
            input_tokens, self.config.input_cost_microusd_per_million
        ) + _ceil_million(output_tokens, self.config.output_cost_microusd_per_million)
        return StructuredCallResponse(
            request_hash=request.request_hash,
            output=output,
            usage=Usage(input_tokens, output_tokens, cost),
            model_id=self.config.wire_model_id,
            provider_request_id=provider_request_id,
        )

    def _validate_request(
        self,
        request: StructuredCallRequest,
        idempotency_key: str,
        timeout_seconds: float,
    ) -> None:
        if type(request) is not StructuredCallRequest:
            raise TypeError("live transport requires exact StructuredCallRequest")
        if (
            request.policy_hash != self.policy.content_hash
            or request.prompt_hash != self.policy.prompt_hash(request.role)
            or request.schema_hash != schema_hash(request.role)
            or request.prompt_template_id
            != self.policy.prompt_template_id(request.role)
            or request.schema_id != self.policy.schema_id(request.role)
        ):
            raise TransportError(
                "live request policy binding differs", definitely_unsent=True
            )
        if idempotency_key != request.operation_id:
            raise TransportError(
                "live idempotency binding differs", definitely_unsent=True
            )
        if timeout_seconds <= 0 or timeout_seconds > self.policy.timeout_seconds:
            raise TransportError("live timeout exceeds policy", definitely_unsent=True)


def _non_negative_int(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise ValueError(f"invalid provider usage:{key}")
    return item


def _ceil_million(tokens: int, rate: int) -> int:
    return (tokens * rate + 999_999) // 1_000_000


__all__ = [
    "BoundedHTTPClient",
    "HTTPResponse",
    "LiveProviderConfig",
    "LiveStructuredTransport",
    "UrllibBoundedHTTPClient",
]
