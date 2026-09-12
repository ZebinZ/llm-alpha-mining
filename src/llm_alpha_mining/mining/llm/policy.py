from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from llm_alpha_mining.mining.artifacts.hashing import hash_json
from llm_alpha_mining.mining.llm.domain import LLMRole, validate_safe_model_id
from llm_alpha_mining.mining.llm.schemas import ROLE_SCHEMA_IDS, schema_hash


_BASE_GUARD = (
    "Return only the supplied closed JSON schema. Never return chain-of-thought, "
    "scratchpad, prose outside JSON, raw observations, security identifiers, dates, "
    "exact local metrics, or teacher/official/holdout information."
)

DEFAULT_PROMPTS: Mapping[LLMRole, str] = MappingProxyType(
    {
        LLMRole.PROPOSER: _BASE_GUARD
        + " Propose mechanistically distinct factor specifications using only the declared DSL.",
        LLMRole.CRITIC: _BASE_GUARD
        + " Review mechanism clarity, redundancy and fragility using closed reason codes.",
        LLMRole.RISK: _BASE_GUARD
        + " Block leakage, unavailable fields, unsafe semantics, complexity or lineage violations.",
        LLMRole.ARBITER: _BASE_GUARD
        + " Select a diverse batch from prior structured decisions; never override a risk block.",
    }
)


@dataclass(frozen=True, slots=True)
class LLMPolicy:
    prompts: Mapping[LLMRole, str] = DEFAULT_PROMPTS
    timeout_seconds: float = 20.0
    max_retries: int = 1
    circuit_failure_threshold: int = 3
    max_input_tokens_per_call: int = 65_536
    max_output_tokens: int = 4096
    max_cost_microusd_per_call: int = 250_000
    allowed_model_ids: tuple[str, ...] = (
        "offline-fake/v1",
        "offline-exact-replay/v1",
    )
    policy_version: str = "structured-alpha-panel-policy/v2"

    def __post_init__(self) -> None:
        prompts = {LLMRole(key): value for key, value in self.prompts.items()}
        if set(prompts) != set(LLMRole):
            raise ValueError("policy must pin one prompt for every LLM role")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in prompts.values()
        ):
            raise ValueError("prompt templates must not be empty")
        object.__setattr__(self, "prompts", MappingProxyType(prompts))
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        for name in (
            "max_retries",
            "circuit_failure_threshold",
            "max_input_tokens_per_call",
            "max_output_tokens",
            "max_cost_microusd_per_call",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "max_retries" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.policy_version != "structured-alpha-panel-policy/v2":
            raise ValueError("unsupported LLM policy version")
        models = tuple(self.allowed_model_ids)
        if not models or len(models) != len(set(models)):
            raise ValueError("allowed_model_ids must be non-empty and unique")
        for model_id in models:
            validate_safe_model_id(model_id)
        object.__setattr__(self, "allowed_model_ids", models)

    def prompt_hash(self, role: LLMRole | str) -> str:
        return hash_json({"prompt": self.prompts[LLMRole(role)]})

    def prompt_template_id(self, role: LLMRole | str) -> str:
        return f"alpha-panel-{LLMRole(role).value}/v2"

    def schema_id(self, role: LLMRole | str) -> str:
        return ROLE_SCHEMA_IDS[LLMRole(role)]

    @property
    def content_hash(self) -> str:
        return hash_json(
            {
                "policy_version": self.policy_version,
                "timeout_seconds": float(self.timeout_seconds),
                "max_retries": self.max_retries,
                "circuit_failure_threshold": self.circuit_failure_threshold,
                "max_input_tokens_per_call": self.max_input_tokens_per_call,
                "max_output_tokens": self.max_output_tokens,
                "max_cost_microusd_per_call": self.max_cost_microusd_per_call,
                "allowed_model_ids": list(self.allowed_model_ids),
                "roles": {
                    role.value: {
                        "prompt_template_id": self.prompt_template_id(role),
                        "prompt_hash": self.prompt_hash(role),
                        "schema_id": self.schema_id(role),
                        "schema_hash": schema_hash(role),
                    }
                    for role in LLMRole
                },
            }
        )
