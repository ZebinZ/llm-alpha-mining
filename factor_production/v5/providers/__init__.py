"""Candidate proposal providers with holdout-safe contexts."""

from factor_production.v5.providers.base import (
    CandidateProvider,
    LocalFeedback,
    ProposalBatch,
    ProposalContext,
    SanitizedFeedback,
)
from factor_production.v5.providers.mock import MockProvider
from factor_production.v5.providers.replay import ReplayProvider

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
        from factor_production.v5.providers.llm_panel import LLMPanelProvider

        return LLMPanelProvider
    raise AttributeError(name)
