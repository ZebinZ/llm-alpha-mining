from __future__ import annotations

from dataclasses import dataclass

from alpha_research.core.data import DataRole
from alpha_research.core.hashing import hash_json


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    evaluation_id: str
    version: str
    quantile_count: int = 5
    minimum_cross_sectional_observations: int = 5
    sample_role: DataRole | str = DataRole.VALIDATION
    correlation_method: str = "pairwise_complete"
    quantile_tie_method: str = "first"
    schema_version: str = "evaluation-spec/v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_role", DataRole(self.sample_role))
        if self.schema_version != "evaluation-spec/v1":
            raise ValueError("unsupported EvaluationSpec schema")
        if not self.evaluation_id.strip() or not self.version.strip():
            raise ValueError("evaluation id and version are required")
        if self.quantile_count < 2:
            raise ValueError("evaluation requires at least two quantiles")
        if self.minimum_cross_sectional_observations < self.quantile_count:
            raise ValueError("minimum observations must cover every quantile")
        if self.sample_role is not DataRole.VALIDATION:
            raise ValueError("factor admission evaluation may only use validation data")
        if self.correlation_method != "pairwise_complete":
            raise ValueError("unsupported evaluation correlation method")
        if self.quantile_tie_method != "first":
            raise ValueError("quantile tie handling must be deterministic")

    @property
    def content_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        role = self.sample_role
        if not isinstance(role, DataRole):  # pragma: no cover
            raise RuntimeError("evaluation role was not normalized")
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "version": self.version,
            "quantile_count": self.quantile_count,
            "minimum_cross_sectional_observations": (
                self.minimum_cross_sectional_observations
            ),
            "sample_role": role.value,
            "correlation_method": self.correlation_method,
            "quantile_tie_method": self.quantile_tie_method,
        }


__all__ = ["EvaluationSpec"]
