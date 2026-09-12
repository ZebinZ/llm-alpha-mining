from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize

from alpha_research.core.hashing import hash_frame, hash_json
from alpha_research.portfolio.spec import (
    ExposureMode,
    PortfolioConstruction,
    PortfolioSpec,
    RiskSpec,
)


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    portfolio_spec_hash: str
    risk_spec_hash: str | None
    weights_hash: str
    diagnostics_hash: str
    risk_exposures_hash: str
    production_ready: bool
    weights: pd.DataFrame
    diagnostics: pd.DataFrame
    risk_exposures: pd.DataFrame

    def __post_init__(self) -> None:
        weights = pd.DataFrame(self.weights).copy(deep=True)
        diagnostics = pd.DataFrame(self.diagnostics).copy(deep=True)
        exposures = pd.DataFrame(self.risk_exposures).copy(deep=True)
        if hash_frame(weights) != self.weights_hash:
            raise ValueError("portfolio weights hash differs")
        if hash_frame(diagnostics) != self.diagnostics_hash:
            raise ValueError("portfolio diagnostics hash differs")
        if hash_frame(exposures) != self.risk_exposures_hash:
            raise ValueError("portfolio risk exposures hash differs")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "risk_exposures", exposures)

    def verify_content(self) -> None:
        if hash_frame(self.weights) != self.weights_hash:
            raise RuntimeError("portfolio weights changed after construction")
        if hash_frame(self.diagnostics) != self.diagnostics_hash:
            raise RuntimeError("portfolio diagnostics changed after construction")
        if hash_frame(self.risk_exposures) != self.risk_exposures_hash:
            raise RuntimeError("portfolio risk exposures changed after construction")

    @property
    def content_hash(self) -> str:
        return hash_json(
            {
                "portfolio_spec_hash": self.portfolio_spec_hash,
                "risk_spec_hash": self.risk_spec_hash,
                "weights_hash": self.weights_hash,
                "diagnostics_hash": self.diagnostics_hash,
                "risk_exposures_hash": self.risk_exposures_hash,
                "production_ready": self.production_ready,
            }
        )


class PortfolioBuilder:
    def build(
        self,
        spec: PortfolioSpec,
        scores: pd.DataFrame,
        universe: pd.DataFrame,
        *,
        risk_spec: RiskSpec | None = None,
        risk_exposures: Mapping[str, pd.DataFrame] | None = None,
        risk_known_at: Mapping[str, pd.DataFrame] | None = None,
    ) -> PortfolioResult:
        scores = pd.DataFrame(scores).copy(deep=True)
        universe = pd.DataFrame(universe).copy(deep=True).astype(bool)
        if not scores.index.equals(universe.index) or not scores.columns.equals(
            universe.columns
        ):
            raise ValueError("portfolio score and universe axes differ")
        if hash_frame(scores) != spec.score_hash:
            raise ValueError("portfolio score binding differs")
        if hash_frame(universe) != spec.universe_hash:
            raise ValueError("portfolio universe binding differs")
        exposures, availability = self._validate_risk(
            risk_spec,
            risk_exposures or {},
            risk_known_at or {},
            reference=scores,
        )
        previous = pd.Series(0.0, index=scores.columns)
        rows: list[pd.Series] = []
        diagnostics: list[dict[str, float | int | str]] = []
        exposure_rows: list[dict[str, float | str]] = []
        for timestamp in scores.index:
            valid = universe.loc[timestamp] & np.isfinite(scores.loc[timestamp])
            raw = _raw_weights(spec, scores.loc[timestamp, valid])
            target = pd.Series(0.0, index=scores.columns)
            target.loc[raw.index] = raw
            optimized = _apply_risk_constraints(
                target,
                spec,
                risk_spec,
                {name: frame.loc[timestamp] for name, frame in exposures.items()},
                {name: frame.loc[timestamp] for name, frame in availability.items()},
                signal_timestamp=pd.Timestamp(timestamp),
                valid=valid,
            )
            unconstrained_turnover = float((optimized - previous).abs().sum() / 2.0)
            if unconstrained_turnover > spec.maximum_one_way_turnover:
                scale = spec.maximum_one_way_turnover / unconstrained_turnover
                optimized = previous + scale * (optimized - previous)
            turnover = float((optimized - previous).abs().sum() / 2.0)
            gross = float(optimized.abs().sum())
            net = float(optimized.sum())
            _assert_portfolio_postconditions(
                optimized,
                spec,
                risk_spec,
                {name: frame.loc[timestamp] for name, frame in exposures.items()},
                {name: frame.loc[timestamp] for name, frame in availability.items()},
                signal_timestamp=pd.Timestamp(timestamp),
                turnover=turnover,
            )
            rows.append(optimized)
            diagnostics.append(
                {
                    "signal_timestamp": timestamp,
                    "eligible_count": int(valid.sum()),
                    "long_count": int((optimized > 1e-14).sum()),
                    "short_count": int((optimized < -1e-14).sum()),
                    "gross_exposure": gross,
                    "net_exposure": net,
                    "target_gross_exposure": spec.gross_leverage,
                    "gross_exposure_shortfall": max(0.0, spec.gross_leverage - gross),
                    "target_net_exposure": spec.target_net_exposure,
                    "net_exposure_deviation": net - spec.target_net_exposure,
                    "one_way_turnover": turnover,
                    "turnover_scaled": int(
                        unconstrained_turnover > spec.maximum_one_way_turnover
                    ),
                }
            )
            for name, frame in exposures.items():
                available = valid & np.isfinite(frame.loc[timestamp])
                exposure_rows.append(
                    {
                        "signal_timestamp": timestamp,
                        "exposure": name,
                        "value": float(
                            np.dot(
                                optimized.loc[available],
                                frame.loc[timestamp, available],
                            )
                        ),
                    }
                )
            previous = optimized
        weights = pd.DataFrame(rows, index=scores.index, columns=scores.columns)
        diagnostic_frame = pd.DataFrame(diagnostics).set_index("signal_timestamp")
        exposure_frame = pd.DataFrame(
            exposure_rows,
            columns=["signal_timestamp", "exposure", "value"],
        )
        if len(exposure_frame):
            exposure_frame = exposure_frame.set_index(["signal_timestamp", "exposure"])
        return PortfolioResult(
            portfolio_spec_hash=spec.content_hash,
            risk_spec_hash=None if risk_spec is None else risk_spec.content_hash,
            weights_hash=hash_frame(weights),
            diagnostics_hash=hash_frame(diagnostic_frame),
            risk_exposures_hash=hash_frame(exposure_frame),
            production_ready=risk_spec is not None and risk_spec.production_ready,
            weights=weights,
            diagnostics=diagnostic_frame,
            risk_exposures=exposure_frame,
        )

    @staticmethod
    def _validate_risk(
        spec: RiskSpec | None,
        exposures: Mapping[str, pd.DataFrame],
        availability: Mapping[str, pd.DataFrame],
        *,
        reference: pd.DataFrame,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        if spec is None:
            if exposures or availability:
                raise ValueError(
                    "risk exposures/availability supplied without RiskSpec"
                )
            return {}, {}
        reference_index = reference.index
        if (
            not isinstance(reference_index, pd.DatetimeIndex)
            or reference_index.tz is None
        ):
            raise ValueError("risk-controlled portfolio signals must be timezone-aware")
        if set(exposures) != set(spec.exposure_hashes):
            raise ValueError("risk exposure names differ from RiskSpec")
        if set(availability) != set(spec.availability_hashes):
            raise ValueError("risk availability names differ from RiskSpec")
        normalized: dict[str, pd.DataFrame] = {}
        normalized_availability: dict[str, pd.DataFrame] = {}
        for name in sorted(exposures):
            frame = pd.DataFrame(exposures[name]).copy(deep=True)
            if not frame.index.equals(reference.index) or not frame.columns.equals(
                reference.columns
            ):
                raise ValueError(f"risk exposure axes differ:{name}")
            if hash_frame(frame) != spec.exposure_hashes[name]:
                raise ValueError(f"risk exposure hash differs:{name}")
            normalized[name] = frame
            known = pd.DataFrame(availability[name]).copy(deep=True)
            if not known.index.equals(reference.index) or not known.columns.equals(
                reference.columns
            ):
                raise ValueError(f"risk availability axes differ:{name}")
            if hash_frame(known) != spec.availability_hashes[name]:
                raise ValueError(f"risk availability hash differs:{name}")
            parsed = pd.DataFrame(index=known.index, columns=known.columns)
            for column in known.columns:
                series = pd.to_datetime(known[column], errors="raise")
                if series.dt.tz is None:
                    raise ValueError(
                        f"risk observation availability must be timezone-aware:{name}"
                    )
                parsed[column] = series
            normalized_availability[name] = parsed
        return normalized, normalized_availability


def _raw_weights(spec: PortfolioSpec, score: pd.Series) -> pd.Series:
    minimum = spec.minimum_names_per_side
    mode = spec.exposure_mode
    if not isinstance(mode, ExposureMode):  # pragma: no cover
        raise RuntimeError("portfolio mode was not normalized")
    if len(score) < minimum * (1 if mode is ExposureMode.LONG_ONLY else 2):
        raise ValueError("insufficient securities for portfolio construction")
    long_budget = (spec.gross_leverage + spec.target_net_exposure) / 2.0
    short_budget = (spec.gross_leverage - spec.target_net_exposure) / 2.0
    construction = spec.construction
    if construction is PortfolioConstruction.TOP_QUANTILE_EQUAL:
        count = max(minimum, int(np.floor(len(score) * spec.selection_fraction)))
        if mode is ExposureMode.MARKET_NEUTRAL and count * 2 > len(score):
            raise ValueError("portfolio long and short selections overlap")
        ordered = score.sort_values(kind="stable")
        long_names = ordered.index[-count:]
        short_names = ordered.index[:count] if short_budget > 0 else pd.Index([])
        output = pd.Series(0.0, index=score.index)
        output.loc[long_names] = long_budget / len(long_names)
        if len(short_names):
            output.loc[short_names] = -short_budget / len(short_names)
    else:
        ranks = score.rank(method="first", pct=True) - 0.5
        positive = ranks.clip(lower=0.0)
        negative = -ranks.clip(upper=0.0)
        output = pd.Series(0.0, index=score.index)
        output.loc[:] = _scale_book(positive, long_budget) - _scale_book(
            negative, short_budget
        )
    if (output.abs() > spec.maximum_absolute_weight + 1e-12).any():
        output = _project_books(output, spec)
    return output


def _scale_book(values: pd.Series, budget: float) -> pd.Series:
    if budget == 0:
        return pd.Series(0.0, index=values.index)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("portfolio book has no eligible score mass")
    return values / total * budget


def _project_books(weights: pd.Series, spec: PortfolioSpec) -> pd.Series:
    long_budget = (spec.gross_leverage + spec.target_net_exposure) / 2.0
    short_budget = (spec.gross_leverage - spec.target_net_exposure) / 2.0
    output = pd.Series(0.0, index=weights.index)
    long = weights.clip(lower=0.0)
    short = -weights.clip(upper=0.0)
    if long_budget:
        output += _capped_simplex(long, long_budget, spec.maximum_absolute_weight)
    if short_budget:
        output -= _capped_simplex(short, short_budget, spec.maximum_absolute_weight)
    return output


def _capped_simplex(values: pd.Series, total: float, cap: float) -> pd.Series:
    eligible = values > 0
    if int(eligible.sum()) * cap + 1e-12 < total:
        raise ValueError("portfolio maximum weight makes requested exposure infeasible")
    source = values.loc[eligible].to_numpy(dtype=float)
    lower = float(source.min() - cap)
    upper = float(source.max())
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        allocated = np.clip(source - midpoint, 0.0, cap)
        if allocated.sum() > total:
            lower = midpoint
        else:
            upper = midpoint
    allocated = np.clip(source - upper, 0.0, cap)
    allocated *= total / allocated.sum()
    output = pd.Series(0.0, index=values.index)
    output.loc[eligible] = allocated
    return output


def _apply_risk_constraints(
    initial: pd.Series,
    portfolio: PortfolioSpec,
    risk: RiskSpec | None,
    exposures: Mapping[str, pd.Series],
    availability: Mapping[str, pd.Series],
    *,
    signal_timestamp: pd.Timestamp,
    valid: pd.Series,
) -> pd.Series:
    if risk is None or risk.constraint_policy == "diagnostic_only":
        return initial
    names = initial.index[valid]
    x0 = initial.loc[names].to_numpy(dtype=float)
    constraint_rows: list[NDArray[np.float64]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for name, exposure in exposures.items():
        vector = exposure.loc[names].to_numpy(dtype=float)
        if not np.isfinite(vector).all():
            raise ValueError(f"risk exposure missing for enforced constraint:{name}")
        known = pd.to_datetime(availability[name].loc[names], errors="raise")
        unavailable = known.isna() | (known > signal_timestamp)
        if unavailable.any():
            raise ValueError(f"risk exposure unavailable at portfolio signal:{name}")
        limit = float(risk.maximum_absolute_exposure[name])
        constraint_rows.append(vector)
        lower_bounds.append(-limit)
        upper_bounds.append(limit)
    linear_constraints = [
        LinearConstraint(
            np.ones((1, len(names)), dtype=float),
            [portfolio.target_net_exposure],
            [portfolio.target_net_exposure],
        )
    ]
    if constraint_rows:
        linear_constraints.append(
            LinearConstraint(
                np.vstack(constraint_rows),
                np.asarray(lower_bounds),
                np.asarray(upper_bounds),
            )
        )
    result = minimize(
        lambda value: float(np.square(value - x0).sum()),
        x0,
        method="SLSQP",
        bounds=[
            (
                (
                    0.0
                    if portfolio.exposure_mode is ExposureMode.LONG_ONLY
                    else -portfolio.maximum_absolute_weight
                ),
                portfolio.maximum_absolute_weight,
            )
            for _ in names
        ],
        constraints=linear_constraints,
        jac=lambda value: 2.0 * (value - x0),
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        raise ValueError(f"portfolio risk optimization infeasible:{result.message}")
    output = pd.Series(0.0, index=initial.index)
    output.loc[names] = result.x
    realized_gross = float(output.abs().sum())
    if realized_gross > portfolio.gross_leverage:
        output *= portfolio.gross_leverage / realized_gross
    for name, exposure in exposures.items():
        value = float(np.dot(output.loc[names], exposure.loc[names]))
        if abs(value) > risk.maximum_absolute_exposure[name] + 1e-7:
            raise RuntimeError(f"portfolio risk constraint failed:{name}")
    return output


def _assert_portfolio_postconditions(
    weights: pd.Series,
    portfolio: PortfolioSpec,
    risk: RiskSpec | None,
    exposures: Mapping[str, pd.Series],
    availability: Mapping[str, pd.Series],
    *,
    signal_timestamp: pd.Timestamp,
    turnover: float,
) -> None:
    tolerance = 1e-7
    if turnover > portfolio.maximum_one_way_turnover + tolerance:
        raise RuntimeError("portfolio turnover cap failed after construction")
    if float(weights.abs().max()) > portfolio.maximum_absolute_weight + tolerance:
        raise RuntimeError("portfolio individual weight cap failed after construction")
    gross = float(weights.abs().sum())
    net = float(weights.sum())
    if gross > portfolio.gross_leverage + tolerance:
        raise RuntimeError("portfolio gross exposure cap failed after construction")
    mode = portfolio.exposure_mode
    if mode is ExposureMode.MARKET_NEUTRAL:
        if abs(net - portfolio.target_net_exposure) > tolerance:
            raise RuntimeError(
                "market-neutral net exposure failed after turnover scaling"
            )
    elif (weights < -tolerance).any() or abs(net - gross) > tolerance:
        raise RuntimeError("long-only exposure failed after turnover scaling")
    if risk is not None and risk.constraint_policy == "enforce":
        for name, exposure in exposures.items():
            held = weights.abs() > tolerance
            missing_held = held & ~np.isfinite(exposure)
            if missing_held.any():
                raise RuntimeError(
                    "portfolio held weight lacks enforced risk exposure after "
                    f"turnover scaling:{name}"
                )
            known = pd.to_datetime(availability[name], errors="raise")
            unavailable_held = held & (known.isna() | (known > signal_timestamp))
            if unavailable_held.any():
                raise RuntimeError(
                    "portfolio held weight uses unavailable risk exposure after "
                    f"turnover scaling:{name}"
                )
            if not np.isfinite(weights).all():
                raise RuntimeError("portfolio weights are nonfinite after construction")
            realized = float(np.dot(weights.loc[held], exposure.loc[held]))
            if abs(realized) > risk.maximum_absolute_exposure[name] + tolerance:
                raise RuntimeError(
                    f"portfolio risk constraint failed after turnover scaling:{name}"
                )


__all__ = ["PortfolioBuilder", "PortfolioResult"]
