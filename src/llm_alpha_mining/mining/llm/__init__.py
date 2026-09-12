"""Offline-first, structured LLM control plane for Alpha research.

Nothing in this package imports a vendor SDK or opens a network connection.
Transports are injected explicitly and every accepted response crosses both a
JSON-schema-shaped validator and a role-specific domain parser.
"""

from .budget import LLMBudget, LLMBudgetLimits
from .domain import LLMRole, StructuredCallRequest, StructuredCallResponse, Usage
from .panel import AlphaResearchPanel, PanelResult
from .policy import LLMPolicy
from .ledger import LedgerPin
from .safe_context import ParentCatalogEntry, SafeResearchContext
from .validation import validate_response_against_request

__all__ = [
    "AlphaResearchPanel",
    "LLMBudget",
    "LLMBudgetLimits",
    "LLMPolicy",
    "LLMRole",
    "LedgerPin",
    "ParentCatalogEntry",
    "PanelResult",
    "SafeResearchContext",
    "StructuredCallRequest",
    "StructuredCallResponse",
    "Usage",
    "validate_response_against_request",
]
