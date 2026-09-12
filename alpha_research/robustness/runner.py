from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from alpha_research.backtest import (
    BacktestDataView,
    BacktestEngine,
    BacktestResult,
    BacktestSpec,
)
from alpha_research.backtest.metrics import performance_metrics
from alpha_research.core.hashing import hash_frame, hash_json
from alpha_research.costs import CostModel
from alpha_research.portfolio import (
    PortfolioBuilder,
    PortfolioResult,
    PortfolioSpec,
    RiskSpec,
)
from alpha_research.robustness.spec import RobustnessSpec


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    robustness_spec_hash: str
    base_backtest_hash: str
    scenario_results_hash: str
    regime_results_hash: str
    production_ready: bool
    scenario_results: pd.DataFrame
    regime_results: pd.DataFrame

    def __post_init__(self) -> None:
        scenarios = pd.DataFrame(self.scenario_results).copy(deep=True)
        regimes = pd.DataFrame(self.regime_results).copy(deep=True)
        if hash_frame(scenarios) != self.scenario_results_hash:
            raise ValueError("robustness scenarios hash differs")
        if hash_frame(regimes) != self.regime_results_hash:
            raise ValueError("robustness regimes hash differs")
        object.__setattr__(self, "scenario_results", scenarios)
        object.__setattr__(self, "regime_results", regimes)


@dataclass(frozen=True, slots=True)
class PseudoFactorReport:
    score_hash: str
    label_hash: str
    validity_hash: str
    trials: int
    random_seed: int
    observed_rank_ic_mean: float | None
    two_sided_pvalue: float | None
    null_distribution_hash: str
    null_distribution: pd.DataFrame

    def __post_init__(self) -> None:
        distribution = pd.DataFrame(self.null_distribution).copy(deep=True)
        if hash_frame(distribution) != self.null_distribution_hash:
            raise ValueError("pseudo-factor null distribution hash differs")
        object.__setattr__(self, "null_distribution", distribution)


@dataclass(frozen=True, slots=True)
class PortfolioPerturbationReport:
    robustness_spec_hash: str
    base_portfolio_spec_hash: str
    score_hash: str
    universe_scenario_hashes: Mapping[str, str]
    scenario_results_hash: str
    production_ready: bool
    scenario_results: pd.DataFrame

    def __post_init__(self) -> None:
        universe_hashes = MappingProxyType(
            dict(sorted(self.universe_scenario_hashes.items()))
        )
        frame = pd.DataFrame(self.scenario_results).copy(deep=True)
        if hash_frame(frame) != self.scenario_results_hash:
            raise ValueError("portfolio perturbation scenarios hash differs")
        object.__setattr__(self, "universe_scenario_hashes", universe_hashes)
        object.__setattr__(self, "scenario_results", frame)


@dataclass(frozen=True, slots=True)
class MarginalContributionReport:
    base_backtest_hash: str
    candidate_backtest_hash: str
    metrics_hash: str
    admission_eligible: bool
    metrics: Mapping[str, float | None]

    def __post_init__(self) -> None:
        metrics = MappingProxyType(dict(sorted(self.metrics.items())))
        if hash_json(dict(metrics)) != self.metrics_hash:
            raise ValueError("marginal contribution metrics hash differs")
        object.__setattr__(self, "metrics", metrics)


class RobustnessRunner:
    def run_backtest_scenarios(
        self,
        robustness: RobustnessSpec,
        base_spec: BacktestSpec,
        portfolio: PortfolioResult,
        cost_model: CostModel,
        data: BacktestDataView,
    ) -> RobustnessReport:
        engine = BacktestEngine()
        base = engine.run(base_spec, portfolio, cost_model, data)
        scenarios: list[dict[str, object]] = [
            _scenario_row("base", "base", base_spec.initial_capital, base)
        ]
        scenario_readiness = [base.production_ready]
        for capital in robustness.capital_levels:
            scenario_spec = replace(
                base_spec,
                backtest_id=f"{base_spec.backtest_id}__capital_{capital:g}",
                initial_capital=float(capital),
            )
            result = engine.run(scenario_spec, portfolio, cost_model, data)
            scenarios.append(
                _scenario_row(f"capital_{capital:g}", "capital", capital, result)
            )
            scenario_readiness.append(result.production_ready)
        for delay in robustness.execution_delay_sessions:
            execution = replace(base_spec.execution, execution_lag_sessions=delay)
            scenario_spec = replace(
                base_spec,
                backtest_id=f"{base_spec.backtest_id}__delay_{delay}",
                execution=execution,
            )
            result = engine.run(scenario_spec, portfolio, cost_model, data)
            scenarios.append(
                _scenario_row(f"delay_{delay}", "delay", float(delay), result)
            )
            scenario_readiness.append(result.production_ready)
        scenario_frame = pd.DataFrame(scenarios).set_index("scenario_id")
        regime_rows: list[dict[str, object]] = []
        for period in robustness.regimes:
            start = pd.Timestamp(period.start)
            end = pd.Timestamp(period.end)
            nav = base.nav.loc[base.nav.index.to_series().between(start, end)]
            if len(nav) < 2:
                regime_rows.append(
                    {
                        "regime": period.name,
                        "start": period.start,
                        "end": period.end,
                        "session_count": len(nav),
                        "gross_annual_return": np.nan,
                        "net_annual_return": np.nan,
                        "net_maximum_drawdown": np.nan,
                    }
                )
                continue
            fills = base.fills
            if len(fills):
                execution_time = pd.to_datetime(fills["execution_timestamp"])
                fills = fills.loc[execution_time.between(start, end)]
            metrics, _ = performance_metrics(
                nav,
                fills,
                annualization=base_spec.annualization_sessions,
                risk_free_rate=base_spec.risk_free_rate,
            )
            regime_rows.append(
                {
                    "regime": period.name,
                    "start": period.start,
                    "end": period.end,
                    "session_count": len(nav),
                    "gross_annual_return": metrics["gross_annual_return"],
                    "net_annual_return": metrics["net_annual_return"],
                    "net_maximum_drawdown": metrics["net_maximum_drawdown"],
                }
            )
        regime_frame = pd.DataFrame(
            regime_rows,
            columns=[
                "regime",
                "start",
                "end",
                "session_count",
                "gross_annual_return",
                "net_annual_return",
                "net_maximum_drawdown",
            ],
        )
        if len(regime_frame):
            regime_frame = regime_frame.set_index("regime")
        return RobustnessReport(
            robustness_spec_hash=robustness.content_hash,
            base_backtest_hash=base.content_hash,
            scenario_results_hash=hash_frame(scenario_frame),
            regime_results_hash=hash_frame(regime_frame),
            production_ready=bool(scenario_readiness) and all(scenario_readiness),
            scenario_results=scenario_frame,
            regime_results=regime_frame,
        )

    def run_portfolio_perturbations(
        self,
        robustness: RobustnessSpec,
        base_portfolio_spec: PortfolioSpec,
        scores: pd.DataFrame,
        universe_scenarios: Mapping[str, pd.DataFrame],
        base_backtest_spec: BacktestSpec,
        cost_model: CostModel,
        data: BacktestDataView,
        *,
        risk_spec: RiskSpec | None = None,
        risk_exposures: Mapping[str, pd.DataFrame] | None = None,
    ) -> PortfolioPerturbationReport:
        scores = pd.DataFrame(scores).copy(deep=True)
        if hash_frame(scores) != base_portfolio_spec.score_hash:
            raise ValueError("portfolio perturbation score binding differs")
        if not universe_scenarios or "base" not in universe_scenarios:
            raise ValueError("portfolio perturbations require a named base universe")
        universes: dict[str, pd.DataFrame] = {}
        for name, raw in sorted(universe_scenarios.items()):
            if not name.strip():
                raise ValueError("portfolio perturbation universe name is empty")
            universe = pd.DataFrame(raw).copy(deep=True).astype(bool)
            if not universe.index.equals(scores.index) or not universe.columns.equals(
                scores.columns
            ):
                raise ValueError(f"portfolio perturbation universe axes differ:{name}")
            universes[name] = universe
        if hash_frame(universes["base"]) != base_portfolio_spec.universe_hash:
            raise ValueError("base universe differs from PortfolioSpec binding")

        rows: list[dict[str, object]] = []
        readiness: list[bool] = []
        builder = PortfolioBuilder()
        engine = BacktestEngine()
        for universe_name, universe in universes.items():
            universe_hash = hash_frame(universe)
            for multiplier in robustness.selection_fraction_multipliers:
                fraction = base_portfolio_spec.selection_fraction * multiplier
                scenario_id = f"{universe_name}__selection_x{multiplier:g}"
                if fraction > 0.5:
                    rows.append(
                        _invalid_portfolio_perturbation_row(
                            scenario_id, universe_name, multiplier, fraction
                        )
                    )
                    readiness.append(False)
                    continue
                portfolio_spec = replace(
                    base_portfolio_spec,
                    portfolio_id=f"{base_portfolio_spec.portfolio_id}__{scenario_id}",
                    universe_hash=universe_hash,
                    selection_fraction=fraction,
                )
                portfolio = builder.build(
                    portfolio_spec,
                    scores,
                    universe,
                    risk_spec=risk_spec,
                    risk_exposures=risk_exposures,
                )
                backtest_spec = replace(
                    base_backtest_spec,
                    backtest_id=f"{base_backtest_spec.backtest_id}__{scenario_id}",
                    portfolio_spec_hash=portfolio.portfolio_spec_hash,
                    portfolio_weights_hash=portfolio.weights_hash,
                    portfolio_result_hash=portfolio.content_hash,
                )
                result = engine.run(backtest_spec, portfolio, cost_model, data)
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "universe": universe_name,
                        "selection_fraction_multiplier": multiplier,
                        "selection_fraction": fraction,
                        "status": "completed",
                        "gross_annual_return": result.metrics["gross_annual_return"],
                        "net_annual_return": result.metrics["net_annual_return"],
                        "net_maximum_drawdown": result.metrics["net_maximum_drawdown"],
                        "total_cost": result.metrics["total_cost"],
                        "production_ready": result.production_ready,
                        "portfolio_result_hash": portfolio.content_hash,
                        "backtest_hash": result.content_hash,
                    }
                )
                readiness.append(result.production_ready)
        frame = pd.DataFrame(rows).set_index("scenario_id")
        return PortfolioPerturbationReport(
            robustness_spec_hash=robustness.content_hash,
            base_portfolio_spec_hash=base_portfolio_spec.content_hash,
            score_hash=hash_frame(scores),
            universe_scenario_hashes={
                name: hash_frame(universe) for name, universe in universes.items()
            },
            scenario_results_hash=hash_frame(frame),
            production_ready=bool(readiness) and all(readiness),
            scenario_results=frame,
        )

    def pseudo_factor_control(
        self,
        scores: pd.DataFrame,
        labels: pd.DataFrame,
        validity: pd.DataFrame,
        *,
        trials: int,
        random_seed: int,
        minimum_cross_sectional_observations: int = 5,
    ) -> PseudoFactorReport:
        scores = pd.DataFrame(scores).copy(deep=True)
        labels = pd.DataFrame(labels).copy(deep=True)
        validity = pd.DataFrame(validity).copy(deep=True).astype(bool)
        if not scores.index.equals(labels.index) or not scores.columns.equals(
            labels.columns
        ):
            raise ValueError("pseudo-factor score and label axes differ")
        if not validity.index.equals(scores.index) or not validity.columns.equals(
            scores.columns
        ):
            raise ValueError("pseudo-factor validity axes differ")
        if trials < 20:
            raise ValueError("pseudo-factor control requires at least 20 trials")
        observed = _mean_rank_ic(
            scores,
            labels,
            validity,
            minimum=minimum_cross_sectional_observations,
        )
        rng = np.random.default_rng(random_seed)
        null_values: list[float] = []
        for _ in range(trials):
            shuffled = scores.copy()
            for timestamp in scores.index:
                valid = validity.loc[timestamp] & np.isfinite(scores.loc[timestamp])
                values = scores.loc[timestamp, valid].to_numpy(dtype=float).copy()
                rng.shuffle(values)
                shuffled.loc[timestamp, valid] = values
            value = _mean_rank_ic(
                shuffled,
                labels,
                validity,
                minimum=minimum_cross_sectional_observations,
            )
            null_values.append(np.nan if value is None else value)
        null = pd.DataFrame({"rank_ic_mean": null_values})
        finite = pd.to_numeric(null["rank_ic_mean"], errors="coerce").dropna()
        pvalue = None
        if observed is not None and len(finite):
            pvalue = float(
                (1 + (finite.abs() >= abs(observed)).sum()) / (len(finite) + 1)
            )
        return PseudoFactorReport(
            score_hash=hash_frame(scores),
            label_hash=hash_frame(labels),
            validity_hash=hash_frame(validity),
            trials=trials,
            random_seed=random_seed,
            observed_rank_ic_mean=observed,
            two_sided_pvalue=pvalue,
            null_distribution_hash=hash_frame(null),
            null_distribution=null,
        )

    def marginal_portfolio_contribution(
        self,
        base: BacktestResult,
        candidate: BacktestResult,
        *,
        annualization_sessions: int = 252,
        minimum_residual_annual_return: float = 0.0,
        minimum_information_ratio_delta: float = 0.0,
    ) -> MarginalContributionReport:
        if base.data_view_hash != candidate.data_view_hash:
            raise ValueError("marginal comparison requires one data/benchmark view")
        if not base.nav.index.equals(candidate.nav.index):
            raise ValueError("marginal comparison calendars differ")
        if annualization_sessions <= 0:
            raise ValueError("marginal comparison annualization must be positive")
        base_return = pd.to_numeric(base.nav["net_excess_return"], errors="coerce")
        candidate_return = pd.to_numeric(
            candidate.nav["net_excess_return"], errors="coerce"
        )
        valid = np.isfinite(base_return) & np.isfinite(candidate_return)
        beta: float | None = None
        residual_annual_return: float | None = None
        return_correlation: float | None = None
        if int(valid.sum()) >= 3:
            x = base_return.loc[valid].to_numpy(dtype=float)
            y = candidate_return.loc[valid].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficients
            beta = float(coefficients[1])
            residual_annual_return = float(
                (coefficients[0] + residual.mean()) * annualization_sessions
            )
            if np.std(x) > 0 and np.std(y) > 0:
                return_correlation = float(np.corrcoef(x, y)[0, 1])
        metrics: dict[str, float | None] = {
            "net_annual_return_delta": _metric_delta(
                candidate, base, "net_annual_return"
            ),
            "net_sharpe_delta": _metric_delta(candidate, base, "net_sharpe"),
            "net_information_ratio_delta": _metric_delta(
                candidate, base, "net_information_ratio"
            ),
            "net_maximum_drawdown_delta": _metric_delta(
                candidate, base, "net_maximum_drawdown"
            ),
            "turnover_delta": _metric_delta(candidate, base, "one_way_turnover"),
            "total_cost_delta": _metric_delta(candidate, base, "total_cost"),
            "base_return_beta": beta,
            "residual_annual_return": residual_annual_return,
            "net_return_correlation": return_correlation,
        }
        information_delta = metrics["net_information_ratio_delta"]
        eligible = (
            base.production_ready
            and candidate.production_ready
            and residual_annual_return is not None
            and residual_annual_return > minimum_residual_annual_return
            and information_delta is not None
            and information_delta > minimum_information_ratio_delta
        )
        return MarginalContributionReport(
            base_backtest_hash=base.content_hash,
            candidate_backtest_hash=candidate.content_hash,
            metrics_hash=hash_json(metrics),
            admission_eligible=eligible,
            metrics=metrics,
        )


def benjamini_hochberg(
    pvalues: Mapping[str, float], *, alpha: float
) -> Mapping[str, Mapping[str, float | bool]]:
    if not 0 < alpha < 1 or not pvalues:
        raise ValueError("BH correction requires p-values and alpha in (0,1)")
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    for name, value in ordered:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid p-value:{name}")
    raw_q = [value * count / rank for rank, (_, value) in enumerate(ordered, start=1)]
    adjusted = [0.0] * count
    running = 1.0
    for offset in range(count - 1, -1, -1):
        running = min(running, raw_q[offset])
        adjusted[offset] = min(1.0, running)
    output = {
        name: {
            "pvalue": float(value),
            "qvalue": float(adjusted[offset]),
            "rejected": bool(adjusted[offset] <= alpha),
        }
        for offset, (name, value) in enumerate(ordered)
    }
    return MappingProxyType(output)


def _scenario_row(
    scenario_id: str,
    scenario_type: str,
    scenario_value: float,
    result: BacktestResult,
) -> dict[str, object]:
    requested = (
        float(
            pd.to_numeric(result.orders["requested_shares"], errors="coerce")
            .abs()
            .sum()
        )
        if len(result.orders)
        else 0.0
    )
    filled = (
        float(pd.to_numeric(result.fills["filled_shares"], errors="coerce").abs().sum())
        if len(result.fills)
        else 0.0
    )
    impact = float(result.cost_breakdown["market_impact"].sum())
    return {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "scenario_value": scenario_value,
        "fill_ratio": None if requested == 0 else filled / requested,
        "gross_annual_return": result.metrics["gross_annual_return"],
        "net_annual_return": result.metrics["net_annual_return"],
        "net_maximum_drawdown": result.metrics["net_maximum_drawdown"],
        "total_cost": result.metrics["total_cost"],
        "market_impact": impact,
        "production_ready": result.production_ready,
        "backtest_hash": result.content_hash,
    }


def _invalid_portfolio_perturbation_row(
    scenario_id: str,
    universe_name: str,
    multiplier: float,
    fraction: float,
) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "universe": universe_name,
        "selection_fraction_multiplier": multiplier,
        "selection_fraction": fraction,
        "status": "invalid_fraction_above_half",
        "gross_annual_return": np.nan,
        "net_annual_return": np.nan,
        "net_maximum_drawdown": np.nan,
        "total_cost": np.nan,
        "production_ready": False,
        "portfolio_result_hash": None,
        "backtest_hash": None,
    }


def _mean_rank_ic(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    validity: pd.DataFrame,
    *,
    minimum: int,
) -> float | None:
    values: list[float] = []
    for timestamp in scores.index:
        valid = (
            validity.loc[timestamp]
            & np.isfinite(scores.loc[timestamp])
            & np.isfinite(labels.loc[timestamp])
        )
        if int(valid.sum()) < minimum:
            continue
        correlation = scores.loc[timestamp, valid].corr(
            labels.loc[timestamp, valid], method="spearman"
        )
        if np.isfinite(correlation):
            values.append(float(correlation))
    return None if not values else float(np.mean(values))


def _metric_delta(
    candidate: BacktestResult, base: BacktestResult, metric: str
) -> float | None:
    candidate_value = candidate.metrics.get(metric)
    base_value = base.metrics.get(metric)
    if candidate_value is None or base_value is None:
        return None
    candidate_numeric = float(candidate_value)
    base_numeric = float(base_value)
    if not np.isfinite(candidate_numeric) or not np.isfinite(base_numeric):
        return None
    return candidate_numeric - base_numeric


__all__ = [
    "MarginalContributionReport",
    "PortfolioPerturbationReport",
    "PseudoFactorReport",
    "RobustnessReport",
    "RobustnessRunner",
    "benjamini_hochberg",
]
