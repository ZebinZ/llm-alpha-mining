from __future__ import annotations

"""Versioned security-identity, universe, and tradability contracts.

This module deliberately does not reinterpret ``stock_name.notna()`` as an
instrument master.  Names are descriptive attributes: they may disappear on
suspended dates and they do not establish an exchange or instrument type.

There are two entry points:

``build_security_data_contract``
    The production-shaped path.  It consumes bitemporal status events with an
    explicit instrument type and exchange.  Production readiness additionally
    requires an affirmative source attestation and a point-in-time price-limit
    channel.

``build_frdata_research_security_contract``
    A compatibility adapter for the currently available ``frdata`` files.  It
    derives a conservative research identity from listing dates and the
    intersection of multiple stock-domain data axes.  Because those files do
    not prove instrument type, exchange namespace, delisting history, or
    point-in-time provenance, this path is always labelled
    ``degraded_research_only``.

All daily masks use exact-date alignment.  Membership, suspension, or market
observations are never backfilled or carried forward.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


SECURITY_DATA_CONTRACT_VERSION = "v5_security_identity_universe_execution_v2"
FEATURE_RETURN_POLICY_VERSION = "v5_feature_return_confirmed_suspend_zero_v1"

_INSTRUMENT_EVENT_COLUMNS = frozenset(
    {
        "instrument_id",
        "instrument_type",
        "exchange",
        "status",
        "effective_date",
        "known_at",
    }
)
_ACTIVE_STATUS = "ACTIVE"
_INACTIVE_STATUS = "INACTIVE"
_ALLOWED_STATUS = frozenset({_ACTIVE_STATUS, _INACTIVE_STATUS})
_DEFAULT_STOCK_TYPES = frozenset({"stock", "common_stock", "a_share"})
_TRUE_SUSPENSION_MARKERS = frozenset({"1", "S", "SUSPENDED", "SUSPEND", "停牌", "TRUE"})
_FALSE_SUSPENSION_MARKERS = frozenset({"0", "N", "NORMAL", "FALSE"})


@dataclass(frozen=True, slots=True)
class SecuritySourceAttestation:
    """Explicit provenance claims required for a production-ready contract.

    Defaults are intentionally false.  A filename or a successful parse is not
    evidence of point-in-time or instrument-type correctness.
    """

    source_id: str
    schema_version: str
    instrument_type_authoritative: bool = False
    exchange_namespace_authoritative: bool = False
    point_in_time_status_history: bool = False
    delisting_history_complete: bool = False
    signal_universe_point_in_time: bool = False
    suspension_events_point_in_time: bool = False
    suspension_absence_means_not_suspended: bool = False
    market_data_point_in_time: bool = False
    price_limit_status_point_in_time: bool = False

    def degraded_reasons(self) -> tuple[str, ...]:
        requirements = {
            "instrument_type_not_authoritative": self.instrument_type_authoritative,
            "exchange_namespace_not_authoritative": (
                self.exchange_namespace_authoritative
            ),
            "status_history_not_point_in_time": self.point_in_time_status_history,
            "delisting_history_incomplete": self.delisting_history_complete,
            "signal_universe_not_point_in_time": self.signal_universe_point_in_time,
            "suspension_events_not_point_in_time": (
                self.suspension_events_point_in_time
            ),
            "suspension_absence_semantics_unattested": (
                self.suspension_absence_means_not_suspended
            ),
            "market_data_not_point_in_time": self.market_data_point_in_time,
            "price_limit_status_not_point_in_time": (
                self.price_limit_status_point_in_time
            ),
        }
        return tuple(key for key, satisfied in requirements.items() if not satisfied)


@dataclass(frozen=True, slots=True)
class SecurityDataContractAudit:
    """JSON-serializable diagnostics for one three-layer contract build."""

    contract_version: str
    mode: str
    production_ready: bool
    degraded_reasons: tuple[str, ...]
    source_id: str
    source_schema_version: str
    identifier_namespace: str
    target_date_count: int
    target_security_count: int
    known_instrument_count: int
    unknown_instrument_count: int
    identity_instrument_count: int
    verified_stock_instrument_count: int
    non_stock_instrument_count: int
    pre_active_cell_count: int
    inactive_cell_count: int
    identity_cell_count: int
    signal_universe_cell_count: int
    execution_tradable_cell_count: int
    signal_source_missing_date_count: int
    signal_source_missing_security_count: int
    return_missing_cell_count: int
    close_missing_cell_count: int
    nonpositive_or_missing_amount_cell_count: int
    suspension_marker_cell_count: int
    confirmed_suspension_cell_count: int
    suspension_trade_conflict_cell_count: int
    price_limit_channel_present: bool
    price_limit_source_missing_date_count: int
    price_limit_source_missing_security_count: int
    name_panel_used_for_identity: bool
    name_identity_disagreement_cell_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["degraded_reasons"] = list(self.degraded_reasons)
        return payload


@dataclass(frozen=True, slots=True)
class SecurityDataContract:
    """Distinct identity, universe, execution, and suspension masks."""

    instrument_identity_mask: pd.DataFrame
    signal_universe_mask: pd.DataFrame
    execution_tradability_mask: pd.DataFrame
    suspension_marker_mask: pd.DataFrame
    confirmed_suspension_mask: pd.DataFrame
    audit: SecurityDataContractAudit


def security_data_contract_hash(contract: SecurityDataContract) -> str:
    """Return the canonical identity of a complete security-data contract.

    All five masks are identity-bearing.  In particular, the suspension masks
    cannot be inferred from execution tradability: they also govern the
    conservative feature-return fill policy and preserve suspension/trade
    conflicts for audit.  Keeping the hash here gives every downstream view
    one binding implementation instead of allowing factor and label factories
    to select different subsets of the contract.
    """

    if not isinstance(contract, SecurityDataContract):
        raise TypeError("contract must be a SecurityDataContract")
    payload = {
        "identity_schema": "security_data_contract_hash_v1",
        "instrument_identity_mask": _hash_contract_frame(
            contract.instrument_identity_mask
        ),
        "signal_universe_mask": _hash_contract_frame(contract.signal_universe_mask),
        "execution_tradability_mask": _hash_contract_frame(
            contract.execution_tradability_mask
        ),
        "suspension_marker_mask": _hash_contract_frame(contract.suspension_marker_mask),
        "confirmed_suspension_mask": _hash_contract_frame(
            contract.confirmed_suspension_mask
        ),
        "audit": contract.audit.to_dict(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_contract_frame(frame: pd.DataFrame) -> str:
    values = pd.DataFrame(frame)
    digest = hashlib.sha256()
    columns = json.dumps(
        list(map(str, values.columns)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    dtypes = json.dumps(
        [str(dtype) for dtype in values.dtypes],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest.update(columns)
    digest.update(dtypes)
    digest.update(pd.util.hash_pandas_object(values, index=True).values.tobytes())
    return digest.hexdigest()


class ProductionDataContractError(ValueError):
    """Raised when a caller requires production data but evidence is degraded."""

    def __init__(self, audit: SecurityDataContractAudit):
        self.audit = audit
        reasons = ",".join(audit.degraded_reasons) or "unspecified"
        super().__init__(f"security_data_contract_not_production_ready:{reasons}")


@dataclass(frozen=True, slots=True)
class FeatureReturnPolicy:
    """Policy for missing historical returns on suspension-status dates.

    A zero is permitted only when all three facts hold on the same date:

    * the instrument identity is active;
    * an explicit suspension marker exists; and
    * the return is missing and positive turnover is absent.

    An observed return always remains unchanged.  In particular, a listing or
    resumption row that conflicts with a broad status marker is never replaced
    with zero.
    """

    policy_id: str = FEATURE_RETURN_POLICY_VERSION
    confirmed_suspension_return: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.confirmed_suspension_return):
            raise ValueError("confirmed suspension return must be finite")


@dataclass(frozen=True, slots=True)
class FeatureReturnAudit:
    policy_id: str
    active_identity_cell_count: int
    observed_return_cell_count: int
    confirmed_suspension_fill_count: int
    unexplained_missing_cell_count: int
    suspension_trade_conflict_cell_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureReturnResult:
    returns: pd.DataFrame
    observed_return_mask: pd.DataFrame
    suspension_filled_mask: pd.DataFrame
    unexplained_missing_mask: pd.DataFrame
    suspension_trade_conflict_mask: pd.DataFrame
    audit: FeatureReturnAudit


@dataclass(frozen=True, slots=True)
class _IdentityDiagnostics:
    known_instrument_count: int
    unknown_instrument_count: int
    identity_instrument_count: int
    verified_stock_instrument_count: int
    non_stock_instrument_count: int
    pre_active_cell_count: int
    inactive_cell_count: int


def build_security_data_contract(
    dates: Sequence[object] | pd.Index,
    securities: Sequence[object] | pd.Index,
    *,
    instrument_events: pd.DataFrame,
    signal_universe_mask: pd.DataFrame,
    stock_returns: pd.DataFrame,
    stock_close: pd.DataFrame,
    stock_amount: pd.DataFrame,
    suspension_events: pd.DataFrame,
    source_attestation: SecuritySourceAttestation,
    price_limit_tradable_mask: pd.DataFrame | None = None,
    stock_types: Sequence[str] = tuple(sorted(_DEFAULT_STOCK_TYPES)),
    require_production_ready: bool = False,
) -> SecurityDataContract:
    """Build identity, signal-universe, and execution masks from explicit data.

    Instrument status is bitemporal: a status event is visible on date ``t``
    only when both ``effective_date <= t`` and ``known_at <= t``.  Among visible
    rows, the latest effective event (then latest known timestamp) determines
    the state.  Unknown, ambiguous, and non-stock instruments fail closed.
    """

    target_dates, target_securities = _normalize_target_axes(dates, securities)
    identity, identity_diagnostics = _identity_from_events(
        target_dates,
        target_securities,
        instrument_events,
        stock_types=stock_types,
    )
    return _assemble_contract(
        target_dates,
        target_securities,
        identity_mask=identity,
        identity_diagnostics=identity_diagnostics,
        signal_universe_mask=signal_universe_mask,
        stock_returns=stock_returns,
        stock_close=stock_close,
        stock_amount=stock_amount,
        suspension_events=suspension_events,
        source_attestation=source_attestation,
        identifier_namespace="explicit_instrument_id",
        price_limit_tradable_mask=price_limit_tradable_mask,
        name_identity_disagreement_cell_count=0,
        extra_degraded_reasons=(),
        require_production_ready=require_production_ready,
    )


def build_frdata_research_security_contract(
    dates: Sequence[object] | pd.Index,
    securities: Sequence[object] | pd.Index,
    *,
    list_dates: pd.DataFrame | pd.Series,
    signal_universe_mask: pd.DataFrame,
    stock_returns: pd.DataFrame,
    stock_close: pd.DataFrame,
    stock_amount: pd.DataFrame,
    stock_suspend: pd.DataFrame,
    stock_name: pd.DataFrame | None = None,
    require_production_ready: bool = False,
) -> SecurityDataContract:
    """Build the strongest honest research-only contract from current frdata.

    Identity requires a valid listing date, a six-digit code, and presence in
    every supplied stock-domain market panel.  Daily name availability is only
    compared for audit purposes and never changes identity.  No delisting date
    is inferred from the last name, price, or return because doing so would use
    future availability to rewrite history.
    """

    target_dates, target_securities = _normalize_target_axes(dates, securities)
    identity, identity_diagnostics = _research_identity_from_frdata(
        target_dates,
        target_securities,
        list_dates=list_dates,
        stock_domain_panels=(stock_returns, stock_close, stock_amount),
    )

    name_disagreement = 0
    if stock_name is not None:
        names = _align_raw_panel(
            stock_name,
            target_dates,
            target_securities,
            source_name="stock_name audit panel",
        )
        name_disagreement = int(
            (names.notna().to_numpy(dtype=bool) != identity.to_numpy(dtype=bool)).sum()
        )

    attestation = SecuritySourceAttestation(
        source_id="frdata_research_adapter",
        schema_version="unattested_legacy_pickles_v1",
    )
    return _assemble_contract(
        target_dates,
        target_securities,
        identity_mask=identity,
        identity_diagnostics=identity_diagnostics,
        signal_universe_mask=signal_universe_mask,
        stock_returns=stock_returns,
        stock_close=stock_close,
        stock_amount=stock_amount,
        suspension_events=stock_suspend,
        source_attestation=attestation,
        identifier_namespace="frdata_stock_channel_bare_six_digit_code",
        price_limit_tradable_mask=None,
        name_identity_disagreement_cell_count=name_disagreement,
        extra_degraded_reasons=(
            "frdata_identity_is_axis_evidence_not_authoritative_master",
            "stock_name_not_used_for_identity",
        ),
        require_production_ready=require_production_ready,
    )


def apply_feature_return_policy(
    stock_returns: pd.DataFrame,
    *,
    instrument_identity_mask: pd.DataFrame,
    suspension_marker_mask: pd.DataFrame,
    stock_amount: pd.DataFrame,
    policy: FeatureReturnPolicy | None = None,
) -> FeatureReturnResult:
    """Apply the versioned, observation-aware feature-return policy.

    ``observed_return_mask`` counts genuine return observations.  Researchers
    should use it, not merely the finite values in ``returns``, when enforcing
    a minimum number of actually traded sessions in rolling features.  This
    prevents long suspensions filled with economic zero returns from creating
    spurious low-volatility signals.
    """

    active_policy = policy or FeatureReturnPolicy()
    source = _normalize_numeric_panel(stock_returns, source_name="stock returns")
    target_dates = pd.Index(source.index, name="date")
    target_securities = pd.Index(source.columns, name="instrument_id")
    identity, _, _ = _align_daily_bool(
        instrument_identity_mask,
        target_dates,
        target_securities,
        source_name="instrument identity mask",
    )
    marker, _, _ = _align_suspension_markers(
        suspension_marker_mask,
        target_dates,
        target_securities,
    )
    amount = _align_numeric_panel(
        stock_amount,
        target_dates,
        target_securities,
        source_name="stock amount",
    )

    values = source.where(identity)
    positive_amount = amount.gt(0.0) & amount.notna()
    observed = identity & values.notna()
    marker_trade_conflict = identity & marker & (observed | positive_amount)
    confirmed_suspension = identity & marker & values.isna() & ~positive_amount
    unexplained_missing = identity & values.isna() & ~confirmed_suspension

    output = values.mask(
        confirmed_suspension,
        float(active_policy.confirmed_suspension_return),
    )
    audit = FeatureReturnAudit(
        policy_id=active_policy.policy_id,
        active_identity_cell_count=_true_count(identity),
        observed_return_cell_count=_true_count(observed),
        confirmed_suspension_fill_count=_true_count(confirmed_suspension),
        unexplained_missing_cell_count=_true_count(unexplained_missing),
        suspension_trade_conflict_cell_count=_true_count(marker_trade_conflict),
    )
    return FeatureReturnResult(
        returns=output,
        observed_return_mask=observed.astype(bool),
        suspension_filled_mask=confirmed_suspension.astype(bool),
        unexplained_missing_mask=unexplained_missing.astype(bool),
        suspension_trade_conflict_mask=marker_trade_conflict.astype(bool),
        audit=audit,
    )


def _identity_from_events(
    dates: pd.Index,
    securities: pd.Index,
    events: pd.DataFrame,
    *,
    stock_types: Sequence[str],
) -> tuple[pd.DataFrame, _IdentityDiagnostics]:
    normalized = _normalize_instrument_events(events)
    allowed_types = {str(item).strip().lower() for item in stock_types}
    if not allowed_types:
        raise ValueError("stock_types must not be empty")

    identity = _false_mask(dates, securities)
    pre_active = _false_mask(dates, securities)
    inactive = _false_mask(dates, securities)
    event_ids = set(normalized["instrument_id"])
    target_ids = set(securities)
    known_ids = target_ids & event_ids
    non_stock_ids: set[str] = set()

    grouped = {key: value for key, value in normalized.groupby("instrument_id")}
    for column_position, instrument_id in enumerate(securities):
        group = grouped.get(instrument_id)
        if group is None:
            continue
        instrument_type = str(group["instrument_type"].iloc[0])
        if instrument_type not in allowed_types:
            non_stock_ids.add(instrument_id)
            continue
        ordered = group.sort_values(["effective_date", "known_at"], kind="mergesort")
        for row_position, date in enumerate(dates):
            visible = ordered[
                (ordered["effective_date"] <= date) & (ordered["known_at"] <= date)
            ]
            if visible.empty:
                pre_active.iat[row_position, column_position] = True
                continue
            state = str(visible.iloc[-1]["status"])
            if state == _ACTIVE_STATUS:
                identity.iat[row_position, column_position] = True
            else:
                inactive.iat[row_position, column_position] = True

    ever_identity = identity.any(axis=0)
    diagnostics = _IdentityDiagnostics(
        known_instrument_count=len(known_ids),
        unknown_instrument_count=len(target_ids - event_ids),
        identity_instrument_count=int(ever_identity.sum()),
        verified_stock_instrument_count=int(ever_identity.sum()),
        non_stock_instrument_count=len(non_stock_ids),
        pre_active_cell_count=_true_count(pre_active),
        inactive_cell_count=_true_count(inactive),
    )
    return identity, diagnostics


def _research_identity_from_frdata(
    dates: pd.Index,
    securities: pd.Index,
    *,
    list_dates: pd.DataFrame | pd.Series,
    stock_domain_panels: Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, _IdentityDiagnostics]:
    listing = _normalize_listing_dates(list_dates)
    axes: list[set[str]] = []
    for position, panel in enumerate(stock_domain_panels):
        axes.append(
            set(
                _normalized_panel_columns(
                    panel,
                    source_name=f"stock domain panel {position}",
                )
            )
        )
    evidence_ids = set(listing.index)
    for axis in axes:
        evidence_ids &= axis
    evidence_ids = {item for item in evidence_ids if len(item) == 6 and item.isdigit()}

    identity = _false_mask(dates, securities)
    pre_active = _false_mask(dates, securities)
    for column_position, instrument_id in enumerate(securities):
        if instrument_id not in evidence_ids:
            continue
        listed_at = str(listing.loc[instrument_id])
        active_values = dates >= listed_at
        identity.iloc[:, column_position] = active_values
        pre_active.iloc[:, column_position] = ~active_values

    target_ids = set(securities)
    known = target_ids & evidence_ids
    diagnostics = _IdentityDiagnostics(
        known_instrument_count=len(known),
        unknown_instrument_count=len(target_ids - evidence_ids),
        identity_instrument_count=int(identity.any(axis=0).sum()),
        # The legacy file names are evidence, not authoritative type metadata.
        verified_stock_instrument_count=0,
        non_stock_instrument_count=0,
        pre_active_cell_count=_true_count(pre_active),
        inactive_cell_count=0,
    )
    return identity, diagnostics


def _assemble_contract(
    dates: pd.Index,
    securities: pd.Index,
    *,
    identity_mask: pd.DataFrame,
    identity_diagnostics: _IdentityDiagnostics,
    signal_universe_mask: pd.DataFrame,
    stock_returns: pd.DataFrame,
    stock_close: pd.DataFrame,
    stock_amount: pd.DataFrame,
    suspension_events: pd.DataFrame,
    source_attestation: SecuritySourceAttestation,
    identifier_namespace: str,
    price_limit_tradable_mask: pd.DataFrame | None,
    name_identity_disagreement_cell_count: int,
    extra_degraded_reasons: Sequence[str],
    require_production_ready: bool,
) -> SecurityDataContract:
    signal_source, signal_missing_dates, signal_missing_securities = _align_daily_bool(
        signal_universe_mask,
        dates,
        securities,
        source_name="signal universe mask",
    )
    returns = _align_numeric_panel(
        stock_returns, dates, securities, source_name="stock returns"
    )
    close = _align_numeric_panel(
        stock_close, dates, securities, source_name="stock close"
    )
    amount = _align_numeric_panel(
        stock_amount, dates, securities, source_name="stock amount"
    )
    suspension_marker, _, _ = _align_suspension_markers(
        suspension_events, dates, securities
    )

    positive_amount = amount.gt(0.0) & amount.notna()
    trade_evidence = returns.notna() | positive_amount
    suspension_trade_conflict = identity_mask & suspension_marker & trade_evidence
    confirmed_suspension = (
        identity_mask & suspension_marker & returns.isna() & ~positive_amount
    )
    observable_execution = returns.notna() & close.notna() & positive_amount

    signal = (identity_mask & signal_source).astype(bool)
    execution = signal & observable_execution & ~suspension_marker

    price_limit_missing_dates = 0
    price_limit_missing_securities = 0
    if price_limit_tradable_mask is not None:
        price_limit, price_limit_missing_dates, price_limit_missing_securities = (
            _align_daily_bool(
                price_limit_tradable_mask,
                dates,
                securities,
                source_name="price limit tradability mask",
            )
        )
        execution &= price_limit

    reasons = list(source_attestation.degraded_reasons())
    reasons.extend(str(item) for item in extra_degraded_reasons)
    if price_limit_tradable_mask is None:
        reasons.append("price_limit_tradability_channel_missing")
    if signal_missing_dates:
        reasons.append("signal_universe_missing_target_dates")
    if signal_missing_securities:
        reasons.append("signal_universe_missing_target_securities")
    if price_limit_tradable_mask is not None and price_limit_missing_dates:
        reasons.append("price_limit_status_missing_target_dates")
    if price_limit_tradable_mask is not None and price_limit_missing_securities:
        reasons.append("price_limit_status_missing_target_securities")
    if _true_count(suspension_trade_conflict):
        reasons.append("suspension_marker_trade_conflicts_detected")
    reasons = list(dict.fromkeys(reasons))
    production_ready = len(reasons) == 0

    audit = SecurityDataContractAudit(
        contract_version=SECURITY_DATA_CONTRACT_VERSION,
        mode="production_ready" if production_ready else "degraded_research_only",
        production_ready=production_ready,
        degraded_reasons=tuple(reasons),
        source_id=source_attestation.source_id,
        source_schema_version=source_attestation.schema_version,
        identifier_namespace=identifier_namespace,
        target_date_count=len(dates),
        target_security_count=len(securities),
        known_instrument_count=identity_diagnostics.known_instrument_count,
        unknown_instrument_count=identity_diagnostics.unknown_instrument_count,
        identity_instrument_count=identity_diagnostics.identity_instrument_count,
        verified_stock_instrument_count=(
            identity_diagnostics.verified_stock_instrument_count
        ),
        non_stock_instrument_count=identity_diagnostics.non_stock_instrument_count,
        pre_active_cell_count=identity_diagnostics.pre_active_cell_count,
        inactive_cell_count=identity_diagnostics.inactive_cell_count,
        identity_cell_count=_true_count(identity_mask),
        signal_universe_cell_count=_true_count(signal),
        execution_tradable_cell_count=_true_count(execution),
        signal_source_missing_date_count=signal_missing_dates,
        signal_source_missing_security_count=signal_missing_securities,
        return_missing_cell_count=int(returns.isna().to_numpy().sum()),
        close_missing_cell_count=int(close.isna().to_numpy().sum()),
        nonpositive_or_missing_amount_cell_count=int(
            (~positive_amount).to_numpy().sum()
        ),
        suspension_marker_cell_count=_true_count(suspension_marker),
        confirmed_suspension_cell_count=_true_count(confirmed_suspension),
        suspension_trade_conflict_cell_count=_true_count(suspension_trade_conflict),
        price_limit_channel_present=price_limit_tradable_mask is not None,
        price_limit_source_missing_date_count=price_limit_missing_dates,
        price_limit_source_missing_security_count=price_limit_missing_securities,
        name_panel_used_for_identity=False,
        name_identity_disagreement_cell_count=(
            int(name_identity_disagreement_cell_count)
        ),
    )
    contract = SecurityDataContract(
        instrument_identity_mask=identity_mask.astype(bool),
        signal_universe_mask=signal,
        execution_tradability_mask=execution.astype(bool),
        suspension_marker_mask=suspension_marker.astype(bool),
        confirmed_suspension_mask=confirmed_suspension.astype(bool),
        audit=audit,
    )
    if require_production_ready and not production_ready:
        raise ProductionDataContractError(audit)
    return contract


def _normalize_instrument_events(events: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(events).copy()
    missing = _INSTRUMENT_EVENT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "instrument events missing columns:" + ",".join(sorted(missing))
        )
    frame = frame.loc[:, sorted(_INSTRUMENT_EVENT_COLUMNS)].copy()
    for column in _INSTRUMENT_EVENT_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"instrument events contain missing {column}")
    frame["instrument_id"] = frame["instrument_id"].map(_normalize_instrument_id)
    frame["instrument_type"] = (
        frame["instrument_type"].astype(str).str.strip().str.lower()
    )
    frame["exchange"] = frame["exchange"].astype(str).str.strip().str.upper()
    frame["status"] = frame["status"].astype(str).str.strip().str.upper()
    frame["effective_date"] = frame["effective_date"].map(_normalize_date)
    frame["known_at"] = frame["known_at"].map(_normalize_date)
    if frame.empty:
        raise ValueError("instrument events must not be empty")
    if (frame["instrument_id"] == "").any():
        raise ValueError("instrument events contain an empty instrument_id")
    if (frame["instrument_type"] == "").any():
        raise ValueError("instrument events contain an empty instrument_type")
    if (frame["exchange"] == "").any():
        raise ValueError("instrument events contain an empty exchange")
    invalid_status = sorted(set(frame["status"]) - _ALLOWED_STATUS)
    if invalid_status:
        raise ValueError(
            "instrument events contain invalid status:" + ",".join(invalid_status)
        )
    if frame.duplicated(
        ["instrument_id", "effective_date", "known_at"], keep=False
    ).any():
        raise ValueError("instrument events contain ambiguous duplicate events")
    for column in ("instrument_type", "exchange"):
        if int(frame.groupby("instrument_id")[column].nunique().max()) > 1:
            raise ValueError(
                f"instrument_id maps to multiple {column} values; use a namespaced id"
            )
    return frame


def _normalize_listing_dates(
    list_dates: pd.DataFrame | pd.Series,
) -> pd.Series:
    if isinstance(list_dates, pd.DataFrame):
        if "list_date" not in list_dates.columns:
            raise ValueError("list_dates DataFrame missing list_date column")
        values = list_dates["list_date"].copy()
    else:
        values = pd.Series(list_dates).copy()
    values.index = pd.Index(
        [_normalize_instrument_id(item) for item in values.index],
        name="instrument_id",
    )
    if not values.index.is_unique:
        raise ValueError(
            "listing-date instrument ids are not unique after normalization"
        )
    normalized = values.map(_normalize_date)
    normalized.name = "list_date"
    return normalized


def _normalize_target_axes(
    dates: Sequence[object] | pd.Index,
    securities: Sequence[object] | pd.Index,
) -> tuple[pd.Index, pd.Index]:
    target_dates = pd.Index([_normalize_date(item) for item in dates], name="date")
    target_securities = pd.Index(
        [_normalize_instrument_id(item) for item in securities],
        name="instrument_id",
    )
    if not target_dates.is_unique:
        raise ValueError("target dates are not unique after normalization")
    if not target_securities.is_unique:
        raise ValueError("target securities are not unique after normalization")
    return target_dates, target_securities


def _align_daily_bool(
    mask: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
    *,
    source_name: str,
) -> tuple[pd.DataFrame, int, int]:
    source = _normalize_raw_panel(mask, source_name=source_name)
    missing_dates = len(set(dates) - set(source.index))
    missing_securities = len(set(securities) - set(source.columns))
    aligned = source.reindex(index=dates, columns=securities)
    result = aligned.astype("boolean").fillna(False).astype(bool)
    result.index.name = "date"
    result.columns.name = "instrument_id"
    return result, missing_dates, missing_securities


def _align_suspension_markers(
    events: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
) -> tuple[pd.DataFrame, int, int]:
    source = _normalize_raw_panel(events, source_name="suspension events")
    missing_dates = len(set(dates) - set(source.index))
    missing_securities = len(set(securities) - set(source.columns))
    aligned = source.reindex(index=dates, columns=securities)

    result = _false_mask(dates, securities)
    stacked = aligned.stack(future_stack=True).dropna()
    if not stacked.empty:
        normalized = stacked.map(_normalize_suspension_marker)
        positions = normalized[normalized].index
        for date, instrument_id in positions:
            result.loc[date, instrument_id] = True
    return result, missing_dates, missing_securities


def _normalize_suspension_marker(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) in (0.0, 1.0):
            return bool(value)
        raise ValueError(f"unknown numeric suspension marker:{value!r}")
    marker = str(value).strip().upper()
    if marker in _TRUE_SUSPENSION_MARKERS:
        return True
    if marker in _FALSE_SUSPENSION_MARKERS:
        return False
    raise ValueError(f"unknown suspension marker:{value!r}")


def _normalize_numeric_panel(
    frame: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    source = _normalize_raw_panel(frame, source_name=source_name)
    values = source.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return values.astype(float)


def _align_numeric_panel(
    frame: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
    *,
    source_name: str,
) -> pd.DataFrame:
    source = _normalize_raw_panel(frame, source_name=source_name)
    aligned = source.reindex(index=dates, columns=securities)
    values = aligned.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    values.index.name = "date"
    values.columns.name = "instrument_id"
    return values.astype(float)


def _align_raw_panel(
    frame: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
    *,
    source_name: str,
) -> pd.DataFrame:
    source = _normalize_raw_panel(frame, source_name=source_name)
    result = source.reindex(index=dates, columns=securities)
    result.index.name = "date"
    result.columns.name = "instrument_id"
    return result


def _normalize_raw_panel(frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy(deep=False)
    out.index = pd.Index([_normalize_date(item) for item in out.index], name="date")
    out.columns = pd.Index(
        [_normalize_instrument_id(item) for item in out.columns],
        name="instrument_id",
    )
    if not out.index.is_unique:
        raise ValueError(f"{source_name} dates are not unique after normalization")
    if not out.columns.is_unique:
        raise ValueError(
            f"{source_name} instruments are not unique after normalization"
        )
    return out.sort_index()


def _normalized_panel_columns(
    frame: pd.DataFrame,
    *,
    source_name: str,
) -> pd.Index:
    columns = pd.Index(
        [_normalize_instrument_id(item) for item in pd.DataFrame(frame).columns]
    )
    if not columns.is_unique:
        raise ValueError(
            f"{source_name} instruments are not unique after normalization"
        )
    return columns


def _false_mask(dates: pd.Index, securities: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(False, index=dates, columns=securities, dtype=bool)


def _true_count(mask: pd.DataFrame) -> int:
    return int(mask.to_numpy(dtype=bool).sum())


def _normalize_instrument_id(value: object) -> str:
    text = str(value).strip()
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    return text


def _normalize_date(value: object) -> str:
    if pd.isna(value):
        raise ValueError("date value must not be missing")
    text = str(value).strip()
    compact = text.replace("-", "").replace("/", "").replace(".", "")
    if compact.isdigit() and len(compact) >= 8:
        return compact[:8]
    return pd.Timestamp(value).strftime("%Y%m%d")
