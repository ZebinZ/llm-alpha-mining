"""Robustness scenarios and family-level significance tests."""

from .runner import (
    MarginalContributionReport,
    PortfolioPerturbationReport,
    PseudoFactorReport,
    RobustnessReport,
    RobustnessRunner,
    benjamini_hochberg,
)
from .spec import RegimePeriod, RobustnessSpec
from .significance import (
    CampaignSignificanceReport,
    FrozenDailyICCalendar,
    FrozenSignificanceCandidate,
    FrozenSignificanceFamily,
    SignificanceMethod,
    SignificanceTestSpec,
    benjamini_hochberg_complete_family,
    evaluate_campaign_significance,
)

__all__ = [
    "MarginalContributionReport",
    "PortfolioPerturbationReport",
    "PseudoFactorReport",
    "RobustnessReport",
    "RobustnessRunner",
    "benjamini_hochberg",
    "RegimePeriod",
    "RobustnessSpec",
    "CampaignSignificanceReport",
    "FrozenDailyICCalendar",
    "FrozenSignificanceCandidate",
    "FrozenSignificanceFamily",
    "SignificanceMethod",
    "SignificanceTestSpec",
    "benjamini_hochberg_complete_family",
    "evaluate_campaign_significance",
]
