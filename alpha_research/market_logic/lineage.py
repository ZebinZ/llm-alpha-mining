from __future__ import annotations

from dataclasses import dataclass

from alpha_research.core.hashing import hash_json, require_sha256
from alpha_research.factors.spec import FactorProvenance
from alpha_research.market_logic.registry import (
    MARKET_LOGIC_REGISTRY_SCHEMA,
    MarketLogicRegistry,
)
from alpha_research.market_logic.spec import (
    LogicOrigin,
    MarketLogicProvenance,
)


PARENT_LOGIC_FACTOR_MAPPING_SCHEMA = "parent-logic-factor-mapping/v1"
PARENT_LOGIC_FACTOR_LINEAGE_SCHEMA = "parent-logic-factor-lineage/v1"
PARENT_LINEAGE_VERIFICATION_SCHEMA = "parent-lineage-verification/v1"
PARENT_LINEAGE_REGISTRY_PROOF_SCHEMA = "parent-lineage-registry-proof/v1"


class ParentLineageError(ValueError):
    """An evolutionary child does not have authoritative parent lineage."""


@dataclass(frozen=True, slots=True)
class ParentLogicFactorMapping:
    """One registered parent-logic -> parent-factor relationship.

    The binding and experiment hashes are included explicitly so that callers
    cannot splice a registered logic from one experiment onto a factor from
    another experiment.
    """

    parent_logic_hash: str
    parent_factor_hash: str
    registered_binding_hash: str
    experiment_hash: str
    schema_version: str = PARENT_LOGIC_FACTOR_MAPPING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PARENT_LOGIC_FACTOR_MAPPING_SCHEMA:
            raise ValueError("unsupported parent logic/factor mapping schema")
        for name in (
            "parent_logic_hash",
            "parent_factor_hash",
            "registered_binding_hash",
            "experiment_hash",
        ):
            require_sha256(str(getattr(self, name)), name=name)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "parent_logic_hash": self.parent_logic_hash,
            "parent_factor_hash": self.parent_factor_hash,
            "registered_binding_hash": self.registered_binding_hash,
            "experiment_hash": self.experiment_hash,
        }


@dataclass(frozen=True, slots=True)
class ParentLineageVerificationReceipt:
    """Metric-free proof of a fresh, read-only registry verification."""

    lineage_hash: str
    child_logic_hash: str
    child_factor_hash: str
    parent_mapping_hashes: tuple[str, ...]
    registered_binding_hashes: tuple[str, ...]
    experiment_hashes: tuple[str, ...]
    registry_proof_hash: str
    schema_version: str = PARENT_LINEAGE_VERIFICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PARENT_LINEAGE_VERIFICATION_SCHEMA:
            raise ValueError("unsupported parent lineage verification schema")
        for name in (
            "lineage_hash",
            "child_logic_hash",
            "child_factor_hash",
            "registry_proof_hash",
        ):
            require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "parent_mapping_hashes",
            "registered_binding_hashes",
            "experiment_hashes",
        ):
            values = tuple(sorted(getattr(self, name)))
            if not values:
                raise ValueError(f"{name} must not be empty")
            for value in values:
                require_sha256(value, name=name)
            # Several parents may legitimately come from one experiment.  Only
            # mapping and binding identities must be one-to-one.
            if name != "experiment_hashes" and len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
            object.__setattr__(self, name, values)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lineage_hash": self.lineage_hash,
            "child_logic_hash": self.child_logic_hash,
            "child_factor_hash": self.child_factor_hash,
            "parent_mapping_hashes": list(self.parent_mapping_hashes),
            "registered_binding_hashes": list(self.registered_binding_hashes),
            "experiment_hashes": list(self.experiment_hashes),
            "registry_proof_hash": self.registry_proof_hash,
        }


@dataclass(frozen=True, slots=True)
class ParentLogicFactorLineage:
    """Immutable, bijective parent mapping for mutation and crossover.

    Construction only validates shape.  :meth:`verify` is the authority
    boundary: it checks both child provenance sets and every exact binding in a
    ``MarketLogicRegistry``.  The registry is read-only throughout verification.
    """

    child_logic_hash: str
    child_factor_hash: str
    parents: tuple[ParentLogicFactorMapping, ...]
    schema_version: str = PARENT_LOGIC_FACTOR_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PARENT_LOGIC_FACTOR_LINEAGE_SCHEMA:
            raise ValueError("unsupported parent logic/factor lineage schema")
        require_sha256(self.child_logic_hash, name="child_logic_hash")
        require_sha256(self.child_factor_hash, name="child_factor_hash")
        parents = tuple(self.parents)
        if not parents or any(
            not isinstance(item, ParentLogicFactorMapping) for item in parents
        ):
            raise TypeError("parent lineage requires mapping objects")
        parents = tuple(
            sorted(
                parents,
                key=lambda item: (
                    item.parent_logic_hash,
                    item.parent_factor_hash,
                    item.registered_binding_hash,
                ),
            )
        )
        for field in (
            "parent_logic_hash",
            "parent_factor_hash",
            "registered_binding_hash",
        ):
            values = tuple(getattr(item, field) for item in parents)
            if len(values) != len(set(values)):
                raise ParentLineageError(f"duplicate_{field}")
        if self.child_logic_hash in {item.parent_logic_hash for item in parents}:
            raise ParentLineageError("child_logic_cannot_be_its_own_parent")
        if self.child_factor_hash in {item.parent_factor_hash for item in parents}:
            raise ParentLineageError("child_factor_cannot_be_its_own_parent")
        object.__setattr__(self, "parents", parents)

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "child_logic_hash": self.child_logic_hash,
            "child_factor_hash": self.child_factor_hash,
            "parents": [item.to_dict() for item in self.parents],
        }

    def verify(
        self,
        *,
        logic_hash: str,
        factor_hash: str,
        logic_provenance: MarketLogicProvenance,
        factor_provenance: FactorProvenance,
        registry: MarketLogicRegistry,
    ) -> ParentLineageVerificationReceipt:
        """Recheck exact child provenance and all parent bindings.

        No evidence table, result metric, date, security identifier, or arbitrary
        logic metadata enters the returned receipt.
        """

        if not isinstance(logic_provenance, MarketLogicProvenance):
            raise TypeError("logic_provenance must be MarketLogicProvenance")
        if not isinstance(factor_provenance, FactorProvenance):
            raise TypeError("factor_provenance must be FactorProvenance")
        if not isinstance(registry, MarketLogicRegistry):
            raise TypeError("lineage verification requires MarketLogicRegistry")
        require_sha256(logic_hash, name="logic_hash")
        require_sha256(factor_hash, name="factor_hash")
        if logic_hash != self.child_logic_hash:
            raise ParentLineageError("child_logic_hash_mismatch")
        if factor_hash != self.child_factor_hash:
            raise ParentLineageError("child_factor_hash_mismatch")

        logic_origin = LogicOrigin(logic_provenance.origin).value
        if logic_origin != factor_provenance.origin:
            raise ParentLineageError("child_origin_mismatch")
        if logic_origin not in {
            LogicOrigin.MUTATION.value,
            LogicOrigin.CROSSOVER.value,
        }:
            raise ParentLineageError("lineage_requires_mutation_or_crossover")
        if logic_origin == LogicOrigin.MUTATION.value and len(self.parents) != 1:
            raise ParentLineageError("mutation_requires_exactly_one_parent")
        if logic_origin == LogicOrigin.CROSSOVER.value and len(self.parents) < 2:
            raise ParentLineageError("crossover_requires_at_least_two_parents")

        declared_logic = tuple(logic_provenance.parent_logic_hashes)
        declared_factor = tuple(sorted(factor_provenance.parent_factor_hashes))
        mapped_logic = tuple(item.parent_logic_hash for item in self.parents)
        mapped_factor = tuple(sorted(item.parent_factor_hash for item in self.parents))
        if mapped_logic != declared_logic:
            raise ParentLineageError("parent_logic_set_mismatch")
        if mapped_factor != declared_factor:
            raise ParentLineageError("parent_factor_set_mismatch")

        # Check the append-only database before producing a proof.  This catches
        # tampering as well as unknown references before individual lookups.
        registry.verify_integrity()
        proof_records: list[dict[str, object]] = []
        for item in self.parents:
            try:
                logic_record = registry.get_logic(item.parent_logic_hash)
                binding_record = registry.read_binding_for_audit(
                    item.registered_binding_hash
                )
            except (KeyError, RuntimeError) as exc:
                raise ParentLineageError("unregistered_parent_reference") from exc
            binding = binding_record.binding
            if binding_record.binding_hash != item.registered_binding_hash:
                raise ParentLineageError("registered_binding_hash_mismatch")
            if binding.logic_hash != item.parent_logic_hash:
                raise ParentLineageError("binding_parent_logic_mismatch")
            if binding.factor_hash != item.parent_factor_hash:
                raise ParentLineageError("binding_parent_factor_mismatch")
            if binding.experiment_hash != item.experiment_hash:
                raise ParentLineageError("binding_experiment_mismatch")
            proof_records.append(
                {
                    "logic_hash": logic_record.logic_hash,
                    "logic_id": logic_record.logic_id,
                    "logic_version": logic_record.version,
                    "registered_binding_hash": binding_record.binding_hash,
                    "binding": binding.to_dict(),
                }
            )

        registry_proof_hash = hash_json(
            {
                "schema_version": PARENT_LINEAGE_REGISTRY_PROOF_SCHEMA,
                "registry_schema_version": MARKET_LOGIC_REGISTRY_SCHEMA,
                "records": proof_records,
            }
        )
        return ParentLineageVerificationReceipt(
            lineage_hash=self.content_hash,
            child_logic_hash=self.child_logic_hash,
            child_factor_hash=self.child_factor_hash,
            parent_mapping_hashes=tuple(item.content_hash for item in self.parents),
            registered_binding_hashes=tuple(
                item.registered_binding_hash for item in self.parents
            ),
            experiment_hashes=tuple(item.experiment_hash for item in self.parents),
            registry_proof_hash=registry_proof_hash,
        )

    def revalidate_receipt(
        self,
        receipt: ParentLineageVerificationReceipt,
        *,
        logic_hash: str,
        factor_hash: str,
        logic_provenance: MarketLogicProvenance,
        factor_provenance: FactorProvenance,
        registry: MarketLogicRegistry,
    ) -> ParentLineageVerificationReceipt:
        """Freshly verify and reject a stale or forged supplied receipt."""

        if not isinstance(receipt, ParentLineageVerificationReceipt):
            raise TypeError("parent lineage receipt has the wrong type")
        expected = self.verify(
            logic_hash=logic_hash,
            factor_hash=factor_hash,
            logic_provenance=logic_provenance,
            factor_provenance=factor_provenance,
            registry=registry,
        )
        if receipt != expected:
            raise ParentLineageError("stale_or_forged_parent_lineage_receipt")
        return expected
