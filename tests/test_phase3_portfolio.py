from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha_research.core.hashing import hash_frame
from alpha_research.portfolio import (
    ExposureMode,
    PortfolioBuilder,
    PortfolioConstruction,
    PortfolioSpec,
    RiskSpec,
)


def _fixture():
    timestamps = pd.date_range(
        "2020-09-01 15:00:00",
        periods=4,
        freq="B",
        tz="Asia/Shanghai",
        name="signal_timestamp",
    )
    securities = pd.Index([f"{number:06d}" for number in range(1, 11)])
    scores = pd.DataFrame(
        [np.roll(np.arange(10, dtype=float), offset) for offset in range(4)],
        index=timestamps,
        columns=securities,
    )
    universe = pd.DataFrame(True, index=timestamps, columns=securities)
    size = pd.DataFrame(
        np.tile(np.linspace(-1.0, 1.0, 10), (4, 1)),
        index=timestamps,
        columns=securities,
    )
    return timestamps, scores, universe, size


def _spec(scores, universe, **changes) -> PortfolioSpec:
    values = {
        "portfolio_id": "phase3-test",
        "version": "1",
        "factor_definition_hash": "a" * 64,
        "score_hash": hash_frame(scores),
        "universe_hash": hash_frame(universe),
        "construction": PortfolioConstruction.TOP_QUANTILE_EQUAL,
        "exposure_mode": ExposureMode.MARKET_NEUTRAL,
        "selection_fraction": 0.2,
        "gross_leverage": 1.0,
        "target_net_exposure": 0.0,
        "maximum_absolute_weight": 0.30,
        "maximum_one_way_turnover": 1.0,
        "minimum_names_per_side": 2,
    }
    values.update(changes)
    return PortfolioSpec(**values)


def _known_at_panel(reference: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [[timestamp] * len(reference.columns) for timestamp in reference.index],
        index=reference.index,
        columns=reference.columns,
    )


def test_equal_weight_market_neutral_portfolio_has_exact_books() -> None:
    _, scores, universe, _ = _fixture()
    result = PortfolioBuilder().build(_spec(scores, universe), scores, universe)
    np.testing.assert_allclose(result.weights.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.weights.abs().sum(axis=1), 1.0, atol=1e-12)
    assert (result.diagnostics["long_count"] == 2).all()
    assert (result.diagnostics["short_count"] == 2).all()
    assert result.production_ready is False


def test_weight_cap_and_turnover_constraints_are_enforced() -> None:
    _, scores, universe, _ = _fixture()
    spec = _spec(
        scores,
        universe,
        selection_fraction=0.5,
        maximum_absolute_weight=0.10,
        maximum_one_way_turnover=0.05,
    )
    result = PortfolioBuilder().build(spec, scores, universe)
    assert float(result.weights.abs().max().max()) <= 0.10 + 1e-12
    assert float(result.diagnostics["one_way_turnover"].max()) <= 0.05 + 1e-12
    assert int(result.diagnostics["turnover_scaled"].sum()) > 0


def test_pit_risk_constraint_is_enforced_and_audited() -> None:
    _, scores, universe, size = _fixture()
    known_at = _known_at_panel(size)
    risk = RiskSpec(
        risk_model_id="pit-size",
        version="1",
        known_at="2020-08-31T18:00:00+08:00",
        exposure_hashes={"size": hash_frame(size)},
        availability_hashes={"size": hash_frame(known_at)},
        maximum_absolute_exposure={"size": 0.02},
        point_in_time=True,
        constraint_policy="enforce",
    )
    result = PortfolioBuilder().build(
        _spec(scores, universe),
        scores,
        universe,
        risk_spec=risk,
        risk_exposures={"size": size},
        risk_known_at={"size": known_at},
    )
    assert result.production_ready is True
    assert (result.risk_exposures["value"].abs() <= 0.0200001).all()


def test_risk_artifact_known_after_first_signal_is_rejected() -> None:
    timestamps, scores, universe, size = _fixture()
    known_at = _known_at_panel(size)
    known_at.loc[timestamps[0], :] = timestamps[0] + pd.Timedelta("1s")
    future_risk = RiskSpec(
        risk_model_id="future-size",
        version="1",
        known_at=(timestamps[0] + pd.Timedelta("1s")).isoformat(),
        exposure_hashes={"size": hash_frame(size)},
        availability_hashes={"size": hash_frame(known_at)},
        maximum_absolute_exposure={"size": 0.1},
        point_in_time=True,
    )
    with pytest.raises(ValueError, match="unavailable at portfolio signal"):
        PortfolioBuilder().build(
            _spec(scores, universe),
            scores,
            universe,
            risk_spec=future_risk,
            risk_exposures={"size": size},
            risk_known_at={"size": known_at},
        )


def test_turnover_cannot_hide_missing_risk_on_a_retained_position() -> None:
    _, scores, universe, size = _fixture()
    # The first row holds the highest-score security.  On the next rebalance it
    # becomes ineligible, but a tight turnover cap retains part of the position.
    # A missing exposure for that retained weight must fail closed.
    universe = universe.copy()
    size = size.copy()
    retained_security = scores.iloc[0].idxmax()
    universe.loc[universe.index[1], retained_security] = False
    size.loc[size.index[1], retained_security] = np.nan
    known_at = _known_at_panel(size)
    risk = RiskSpec(
        risk_model_id="pit-size-with-gap",
        version="1",
        known_at="2020-08-31T18:00:00+08:00",
        exposure_hashes={"size": hash_frame(size)},
        availability_hashes={"size": hash_frame(known_at)},
        maximum_absolute_exposure={"size": 1.0},
        point_in_time=True,
        constraint_policy="enforce",
    )
    with pytest.raises(RuntimeError, match="held weight lacks enforced risk exposure"):
        PortfolioBuilder().build(
            _spec(scores, universe, maximum_one_way_turnover=0.01),
            scores,
            universe,
            risk_spec=risk,
            risk_exposures={"size": size},
            risk_known_at={"size": known_at},
        )


def test_enforced_risk_requires_per_observation_availability() -> None:
    _, _, _, size = _fixture()
    with pytest.raises(ValueError, match="per-observation availability"):
        RiskSpec(
            risk_model_id="legacy-global-known-at",
            version="1",
            known_at="2020-08-31T18:00:00+08:00",
            exposure_hashes={"size": hash_frame(size)},
            maximum_absolute_exposure={"size": 0.1},
            point_in_time=True,
            constraint_policy="enforce",
        )


def test_non_pit_risk_model_cannot_enforce_constraints() -> None:
    _, _, _, size = _fixture()
    with pytest.raises(ValueError, match="non-PIT"):
        RiskSpec(
            risk_model_id="bad-risk",
            version="1",
            known_at="2020-08-31T18:00:00+08:00",
            exposure_hashes={"size": hash_frame(size)},
            maximum_absolute_exposure={"size": 0.1},
            point_in_time=False,
            constraint_policy="enforce",
        )


def test_portfolio_bindings_fail_closed_on_score_or_universe_drift() -> None:
    _, scores, universe, _ = _fixture()
    spec = _spec(scores, universe)
    with pytest.raises(ValueError, match="score binding"):
        PortfolioBuilder().build(spec, scores + 1.0, universe)
    with pytest.raises(ValueError, match="universe binding"):
        PortfolioBuilder().build(spec, scores, ~universe)
    with pytest.raises(ValueError, match="long-only"):
        replace(
            spec,
            exposure_mode=ExposureMode.LONG_ONLY,
            target_net_exposure=0.0,
        )
