from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_alpha_mining.research.core.hashing import hash_json
from llm_alpha_mining.research.contracts.campaign_protocol import (
    ResearchCampaignProtocol,
)
from llm_alpha_mining.research.robustness.significance import (
    FrozenDailyICCalendar,
    FrozenSignificanceCandidate,
    FrozenSignificanceFamily,
    SignificanceMethod,
    SignificanceTestSpec,
    benjamini_hochberg_complete_family,
    evaluate_campaign_significance,
)


def _digest(name: str) -> str:
    return hash_json({"component": name})


def _index(count: int = 120) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-02", periods=count, freq="B", tz="Asia/Shanghai")


def _calendar(count: int = 120) -> FrozenDailyICCalendar:
    return FrozenDailyICCalendar(
        calendar_id="cn-a-share-primary-ic",
        version=f"test-{count}",
        sessions=tuple(item.strftime("%Y%m%d") for item in _index(count)),
        source_calendar_hash=_digest(f"source-calendar-{count}"),
    )


def _raw_ic(
    *,
    mean: float,
    seed: int,
    count: int = 120,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 0.01, size=count)
    values: np.ndarray = np.empty(count, dtype=float)
    values[0] = innovations[0]
    for offset in range(1, count):
        values[offset] = 0.4 * values[offset - 1] + innovations[offset]
    return pd.Series(mean + values, index=_index(count))


def _family(
    declarations: tuple[tuple[str, int], ...],
) -> FrozenSignificanceFamily:
    return FrozenSignificanceFamily(
        family_id="frozen-test-family",
        version="1",
        candidates=tuple(
            FrozenSignificanceCandidate(
                candidate_id=candidate_id,
                factor_spec_hash=_digest(f"factor:{candidate_id}"),
                hypothesis_hash=_digest(f"hypothesis:{candidate_id}"),
                declared_sign=sign,
            )
            for candidate_id, sign in sorted(declarations)
        ),
    )


def _hac_spec() -> SignificanceTestSpec:
    return SignificanceTestSpec(
        test_id="daily-raw-rank-ic",
        version="1",
        method="hac",
        minimum_observations=60,
        hac_lags=5,
        bootstrap_trials=None,
        block_length=None,
        random_seed=20260723,
    )


def _bootstrap_spec() -> SignificanceTestSpec:
    return SignificanceTestSpec(
        test_id="daily-raw-rank-ic",
        version="1",
        method="block_bootstrap",
        minimum_observations=60,
        hac_lags=None,
        bootstrap_trials=2_000,
        block_length=5,
        random_seed=20260723,
    )


def _protocol(
    *,
    spec: SignificanceTestSpec,
    family: FrozenSignificanceFamily,
    calendar: FrozenDailyICCalendar,
) -> ResearchCampaignProtocol:
    source_hashes = {
        "snapshot": _digest("snapshot"),
        "order": _digest("order"),
        "trade": _digest("trade"),
    }
    method = spec.method
    assert isinstance(method, SignificanceMethod)
    return ResearchCampaignProtocol(
        search_design_authority_hash=_digest("search-design-authority"),
        data_identity_hash=hash_json(source_hashes),
        source_data_identity_hashes=source_hashes,
        joint_daily_spec_hash=_digest("joint"),
        cutover_policy_hash=_digest("cutover"),
        candidate_family_hash=family.content_hash,
        primary_label_spec_hash=_digest("label"),
        evaluation_calendar_hash=calendar.content_hash,
        validation_spec_hash=_digest("validation"),
        evaluation_spec_hash=_digest("evaluation"),
        statistical_test_spec_hash=spec.content_hash,
        robustness_spec_hash=_digest("robustness"),
        cost_model_hash=_digest("cost"),
        raw_idea_budget=800,
        frozen_factor_spec_cap=300,
        adaptive_validation_cap=45,
        locked_test_cap=30,
        final_submission_target=24,
        final_submission_maximum=30,
        p_value_method=method.value,
    )


def _evaluate(
    spec: SignificanceTestSpec,
    family: FrozenSignificanceFamily,
    calendar: FrozenDailyICCalendar,
    observations: dict[str, pd.Series | None],
):
    return evaluate_campaign_significance(
        spec,
        protocol=_protocol(spec=spec, family=family, calendar=calendar),
        candidate_family=family,
        calendar=calendar,
        raw_daily_ic=observations,
    )


def test_hac_primary_test_and_complete_family_bh_fail_closed() -> None:
    family = _family(
        (
            ("positive", 1),
            ("negative", 1),
            ("failed", 1),
            ("missing", 1),
        )
    )
    report = _evaluate(
        _hac_spec(),
        family,
        _calendar(),
        {
            "positive": _raw_ic(mean=0.025, seed=1),
            "negative": _raw_ic(mean=-0.015, seed=2),
            "failed": None,
        },
    )
    values = report.results

    assert values.loc["positive", "status"] == "completed"
    assert values.loc["positive", "primary_p_value"] < 0.01
    assert values.loc["positive", "bh_q_value"] < 0.05
    assert values.loc["negative", "primary_p_value"] > 0.9
    assert values.loc["failed", "status"] == "failed"
    assert values.loc["failed", "primary_p_value"] == 1.0
    assert values.loc["missing", "status"] == "missing"
    assert values.loc["missing", "bh_q_value"] == 1.0
    assert len(values) == 4
    assert set(report.input_series_hashes) == set(family.candidate_ids)


def test_declared_negative_direction_is_applied_inside_test() -> None:
    family = _family((("reversal", -1),))
    report = _evaluate(
        _hac_spec(),
        family,
        _calendar(),
        {"reversal": _raw_ic(mean=-0.025, seed=11)},
    )
    assert report.results.loc["reversal", "directed_mean"] > 0
    assert report.results.loc["reversal", "primary_p_value"] < 0.01


def test_direction_cannot_be_changed_after_protocol_freeze() -> None:
    spec = _hac_spec()
    calendar = _calendar()
    original = _family((("alpha", 1),))
    protocol = _protocol(spec=spec, family=original, calendar=calendar)
    altered = _family((("alpha", -1),))
    with pytest.raises(ValueError, match="candidate family binding differs"):
        evaluate_campaign_significance(
            spec,
            protocol=protocol,
            candidate_family=altered,
            calendar=calendar,
            raw_daily_ic={"alpha": _raw_ic(mean=-0.02, seed=7)},
        )


def test_block_bootstrap_is_candidate_stable_and_reproducible() -> None:
    spec = _bootstrap_spec()
    family = _family((("alpha", 1),))
    calendar = _calendar()
    observations = {"alpha": _raw_ic(mean=0.018, seed=9)}
    first = _evaluate(spec, family, calendar, observations)
    second = _evaluate(spec, family, calendar, observations)

    assert first.content_hash == second.content_hash
    pd.testing.assert_frame_equal(first.results, second.results)
    assert first.results.loc["alpha", "primary_p_value"] < 0.05


def test_insufficient_and_degenerate_series_receive_p_one() -> None:
    short_calendar = _calendar(20)
    short_family = _family((("short", 1),))
    short = _evaluate(
        _hac_spec(),
        short_family,
        short_calendar,
        {"short": _raw_ic(mean=0.02, seed=3, count=20)},
    )
    assert short.results.loc["short", "status"] == "insufficient_observations"
    assert short.results.loc["short", "primary_p_value"] == 1.0

    calendar = _calendar()
    constant_family = _family((("constant", 1),))
    constant = _evaluate(
        _hac_spec(),
        constant_family,
        calendar,
        {"constant": pd.Series(0.02, index=calendar.index)},
    )
    assert constant.results.loc["constant", "status"] == "degenerate"
    assert constant.results.loc["constant", "primary_p_value"] == 1.0


def test_primary_series_requires_exact_frozen_calendar() -> None:
    spec = _hac_spec()
    family = _family((("alpha", 1),))
    calendar = _calendar()
    naive = pd.Series(
        np.arange(120, dtype=float),
        index=pd.date_range("2020-01-01", periods=120, freq="D"),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        _evaluate(spec, family, calendar, {"alpha": naive})

    missing_day = _raw_ic(mean=0.01, seed=1).iloc[1:]
    with pytest.raises(ValueError, match="exactly match frozen calendar"):
        _evaluate(spec, family, calendar, {"alpha": missing_day})

    nonfinite = _raw_ic(mean=0.01, seed=1)
    nonfinite.iloc[5] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        _evaluate(spec, family, calendar, {"alpha": nonfinite})


def test_unknown_or_subset_candidate_cannot_replace_full_family() -> None:
    spec = _hac_spec()
    calendar = _calendar()
    family = _family((("alpha", 1), ("beta", 1)))
    with pytest.raises(ValueError, match="outside frozen family"):
        _evaluate(
            spec,
            family,
            calendar,
            {
                "alpha": _raw_ic(mean=0.01, seed=1),
                "late_added": _raw_ic(mean=0.02, seed=2),
            },
        )

    protocol = _protocol(spec=spec, family=family, calendar=calendar)
    subset = _family((("alpha", 1),))
    with pytest.raises(ValueError, match="candidate family binding differs"):
        evaluate_campaign_significance(
            spec,
            protocol=protocol,
            candidate_family=subset,
            calendar=calendar,
            raw_daily_ic={"alpha": _raw_ic(mean=0.01, seed=1)},
        )


def test_protocol_rejects_test_or_calendar_substitution() -> None:
    spec = _hac_spec()
    family = _family((("alpha", 1),))
    calendar = _calendar()
    protocol = _protocol(spec=spec, family=family, calendar=calendar)
    other_calendar = FrozenDailyICCalendar(
        calendar_id=calendar.calendar_id,
        version="different",
        sessions=calendar.sessions,
        source_calendar_hash=calendar.source_calendar_hash,
    )
    with pytest.raises(ValueError, match="evaluation calendar binding differs"):
        evaluate_campaign_significance(
            spec,
            protocol=protocol,
            candidate_family=family,
            calendar=other_calendar,
            raw_daily_ic={"alpha": _raw_ic(mean=0.01, seed=1)},
        )


def test_frozen_family_cannot_exceed_protocol_locked_test_cap() -> None:
    spec = _hac_spec()
    calendar = _calendar()
    family = _family(tuple((f"candidate-{offset:02d}", 1) for offset in range(31)))
    protocol = _protocol(spec=spec, family=family, calendar=calendar)

    with pytest.raises(ValueError, match="exceeds campaign locked test cap"):
        evaluate_campaign_significance(
            spec,
            protocol=protocol,
            candidate_family=family,
            calendar=calendar,
            raw_daily_ic={},
        )


def test_bh_adjustment_uses_entire_family() -> None:
    assert benjamini_hochberg_complete_family((0.01, 0.04, 1.0)) == pytest.approx(
        (0.03, 0.06, 1.0)
    )
    with pytest.raises(ValueError, match="finite probabilities"):
        benjamini_hochberg_complete_family((0.01, float("nan")))


@pytest.mark.parametrize(
    "changes",
    [
        {"alternative": "two_sided"},
        {"missing_or_failed_p_value": 0.05},
        {"multiple_testing_scope": "selected_candidates"},
        {"method": "hac", "hac_lags": 5, "bootstrap_trials": 2_000},
        {
            "method": "block_bootstrap",
            "hac_lags": None,
            "bootstrap_trials": 1_999,
            "block_length": 5,
        },
    ],
)
def test_invalid_primary_test_contract_fails_closed(
    changes: dict[str, object],
) -> None:
    values = _hac_spec().to_dict()
    values.update(changes)
    with pytest.raises(ValueError):
        SignificanceTestSpec.from_mapping(values)


def test_specs_family_calendar_roundtrip_and_report_mutation_detection() -> None:
    spec = _bootstrap_spec()
    family = _family((("alpha", 1),))
    calendar = _calendar()
    assert SignificanceTestSpec.from_mapping(spec.to_dict()) == spec
    assert FrozenSignificanceFamily.from_mapping(family.to_dict()) == family
    assert FrozenDailyICCalendar.from_mapping(calendar.to_dict()) == calendar

    report = _evaluate(
        spec,
        family,
        calendar,
        {"alpha": _raw_ic(mean=0.02, seed=8)},
    )
    report.results.loc["alpha", "primary_p_value"] = 1.0
    with pytest.raises(ValueError, match="results changed"):
        _ = report.content_hash
