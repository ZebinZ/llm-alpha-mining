"""Candidate proposal providers with holdout-safe contexts."""

from llm_alpha_mining.mining.providers.base import (
    CandidateProvider,
    LocalFeedback,
    ProposalBatch,
    ProposalContext,
    SanitizedFeedback,
)
from llm_alpha_mining.mining.providers.mock import MockProvider
from llm_alpha_mining.mining.providers.replay import ReplayProvider

__all__ = [
    "CandidateProvider",
    "LocalFeedback",
    "MockProvider",
    "ProposalBatch",
    "ProposalContext",
    "ReplayProvider",
    "SanitizedFeedback",
    "LLMPanelProvider",
]


def __getattr__(name: str):
    # Lazy import avoids a cycle because the LLM safe-context type deliberately
    # reuses the authoritative SanitizedFeedback wire model from providers.base.
    if name == "LLMPanelProvider":
        from llm_alpha_mining.mining.providers.llm_panel import LLMPanelProvider

        return LLMPanelProvider
    raise AttributeError(name)
