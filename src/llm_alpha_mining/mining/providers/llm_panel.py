from __future__ import annotations

from typing import Callable, Iterable

from llm_alpha_mining.mining.artifacts.hashing import hash_json
from llm_alpha_mining.mining.domain.enums import ProposalKind
from llm_alpha_mining.mining.domain.models import CandidateSpec
from llm_alpha_mining.mining.llm.panel import AlphaResearchPanel, PanelResult
from llm_alpha_mining.mining.llm.safe_context import (
    ParentCatalogEntry,
    SafeResearchContext,
    UnsafeLLMContext,
)
from llm_alpha_mining.mining.protocol import MiningProtocol
from llm_alpha_mining.mining.providers.base import (
    CandidateProvider,
    ProposalBatch,
    ProposalContext,
)


class LLMPanelProvider(CandidateProvider):
    """CandidateProvider adapter for the structured, teacher-isolated panel."""

    def __init__(
        self,
        *,
        protocol: MiningProtocol,
        panel: AlphaResearchPanel,
        name: str = "structured_llm_panel_v2",
        parent_catalog: tuple[ParentCatalogEntry, ...] = (),
        parent_catalog_resolver: Callable[
            [ProposalContext], Iterable[ParentCatalogEntry]
        ]
        | None = None,
    ) -> None:
        self.protocol = protocol
        self.panel = panel
        self.name = name
        self.last_result: PanelResult | None = None
        if parent_catalog and parent_catalog_resolver is not None:
            raise ValueError(
                "configure a static parent_catalog or a resolver, not both"
            )
        self._parent_catalog = tuple(parent_catalog)
        self._parent_catalog_resolver = parent_catalog_resolver
        if any(
            not isinstance(item, ParentCatalogEntry) for item in self._parent_catalog
        ):
            raise TypeError("parent_catalog accepts ParentCatalogEntry only")
        if len({item.spec_hash for item in self._parent_catalog}) != len(
            self._parent_catalog
        ):
            raise ValueError("parent_catalog spec hashes must be unique")

    def _safe_context(self, context: ProposalContext) -> SafeResearchContext:
        if context.protocol_hash != self.protocol.content_hash:
            raise UnsafeLLMContext(
                "proposal context is not bound to the configured protocol"
            )
        if context.local_feedback:
            # LocalFeedback contains exact metric/value pairs.  Adaptive LLMs
            # receive only the coarse, closed SanitizedFeedback vocabulary.
            raise UnsafeLLMContext("exact LocalFeedback cannot enter the LLM panel")
        resolved = (
            tuple(self._parent_catalog_resolver(context))
            if self._parent_catalog_resolver is not None
            else self._parent_catalog
        )
        if any(not isinstance(item, ParentCatalogEntry) for item in resolved):
            raise UnsafeLLMContext("parent catalog resolver returned an unsafe entry")
        if len({item.spec_hash for item in resolved}) != len(resolved):
            raise UnsafeLLMContext("resolved parent catalog spec hashes must be unique")
        by_hash = {item.spec_hash: item for item in resolved}
        missing = set(context.prior_candidate_hashes) - set(by_hash)
        extra = set(by_hash) - set(context.prior_candidate_hashes)
        if context.generation > 0 and (missing or extra):
            raise UnsafeLLMContext(
                "derived proposal context must exactly match the frozen parent catalog"
            )
        if context.generation == 0 and (context.prior_candidate_hashes or resolved):
            raise UnsafeLLMContext("generation zero cannot receive prior candidates")
        ordered_catalog = tuple(
            by_hash[digest] for digest in context.prior_candidate_hashes
        )
        return SafeResearchContext(
            campaign_hash=hash_json({"run_id": context.run_id}),
            protocol_hash=context.protocol_hash,
            generation=context.generation,
            requested_count=context.request_count,
            allowed_fields=tuple(self.protocol.allowed_fields),
            allowed_operators=tuple(self.protocol.allowed_operators),
            allowed_windows=tuple(self.protocol.allowed_windows),
            parent_catalog=ordered_catalog,
            feedback=tuple(context.sanitized_feedback),
        )

    def propose(self, context: ProposalContext) -> ProposalBatch:
        safe = self._safe_context(context)
        result = self.panel.run(safe)
        self.last_result = result
        specs: list[CandidateSpec] = []
        parents_by_id = {item.candidate_id: item for item in safe.parent_catalog}
        for draft in result.selected:
            if context.generation == 0:
                kind = ProposalKind.ROOT
                depth = 0
            elif len(draft.parent_ids) == 1:
                kind = ProposalKind.MUTATION
                try:
                    depth = 1 + max(
                        parents_by_id[parent_id].lineage_depth
                        for parent_id in draft.parent_ids
                    )
                except KeyError:  # defense in depth after panel preflight
                    raise UnsafeLLMContext(
                        "selected draft has an unknown parent"
                    ) from None
            else:
                kind = ProposalKind.CROSSOVER
                try:
                    depth = 1 + max(
                        parents_by_id[parent_id].lineage_depth
                        for parent_id in draft.parent_ids
                    )
                except KeyError:  # defense in depth after panel preflight
                    raise UnsafeLLMContext(
                        "selected draft has an unknown parent"
                    ) from None
            aggregation = dict(draft.aggregation)
            if aggregation.get("method") == "none":
                aggregation = {}
            specs.append(
                CandidateSpec(
                    candidate_id=draft.candidate_id,
                    hypothesis=draft.hypothesis,
                    expression=draft.expression,
                    direction=draft.direction,
                    frequency=draft.frequency,
                    generation=context.generation,
                    family=draft.family,
                    required_fields=draft.required_fields,
                    protocol_hash=context.protocol_hash,
                    provider=self.name,
                    parent_ids=draft.parent_ids,
                    aggregation=aggregation,
                    parameters={},
                    tags=tuple(draft.tags) + ("structured_llm", "offline_contract_v2"),
                    campaign_round=context.generation,
                    lineage_depth=depth,
                    proposal_kind=kind,
                )
            )
        return ProposalBatch(
            provider=self.name, candidates=tuple(specs), exhausted=False
        )
