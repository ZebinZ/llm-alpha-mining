from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_alpha_mining.research.core.data import (
    DataBatch,
    DataRequest,
    DataRole,
    DatasetSnapshot,
    DatasetSpec,
    SourceAsset,
    standard_market_schema,
)
from llm_alpha_mining.research.core.frequency import AvailabilitySpec, FrequencySpec
from llm_alpha_mining.research.core.hashing import hash_file, hash_frame, hash_json
from llm_alpha_mining.research.data.views import MarketDataView
from llm_alpha_mining.research.factors import (
    AggregationSpec,
    ComplexityBudget,
    FactorDataView,
    FactorEngine,
    FactorProvenance,
    FactorRegistry,
    FactorRegistryConflict,
    FactorSpec,
    PreprocessKind,
    PreprocessStep,
    factor_spec_from_candidate,
)
from llm_alpha_mining.research.labels import LabelDataView, PriceObservationEvent
from llm_alpha_mining.mining.data.security_contract import (
    SecuritySourceAttestation,
    build_security_data_contract,
    security_data_contract_hash,
)
from llm_alpha_mining.mining.dsl import OperatorRegistry
from llm_alpha_mining.mining.domain import CandidateSpec


DATES = ("20200102", "20200103", "20200106", "20200107")
SECURITIES = ("000001", "000002", "399001")
CREATED = "2026-07-19T00:00:00+08:00"


def _full_attestation() -> SecuritySourceAttestation:
    return SecuritySourceAttestation(
        source_id="research-test-master",
        schema_version="1",
        instrument_type_authoritative=True,
        exchange_namespace_authoritative=True,
        point_in_time_status_history=True,
        delisting_history_complete=True,
        signal_universe_point_in_time=True,
        suspension_events_point_in_time=True,
        suspension_absence_means_not_suspended=True,
        market_data_point_in_time=True,
        price_limit_status_point_in_time=True,
    )


def _fixture(tmp_path: Path):
    source = tmp_path / "factor-source.bin"
    source.write_bytes(b"research-factor-source")
    schema = standard_market_schema(
        schema_id="research-factor-data",
        version="1",
        feature_fields={"Returns": "float", "Amount": "float"},
    )
    frequency = FrequencySpec.daily()
    availability = AvailabilitySpec.market_data()
    dataset = DatasetSpec(
        dataset_id="research-daily",
        version="1",
        owner="research",
        description="Research factor contract fixture",
        storage_format="memory",
        schema_hash=schema.content_hash,
        frequency=frequency,
        availability=availability,
    )
    snapshot = DatasetSnapshot.create(
        dataset=dataset,
        schema=schema,
        source_assets=(
            SourceAsset.from_path(
                source, logical_name="raw", expected_sha256=hash_file(source)
            ),
        ),
        source_vintage_at="2026-07-18T00:00:00+08:00",
        created_at=CREATED,
    )
    request = DataRequest(
        dataset_id=dataset.dataset_id,
        snapshot_id=snapshot.snapshot_id,
        fields=("Returns", "Amount"),
        start="2020-01-02T00:00:00+08:00",
        end="2020-01-07T23:59:59+08:00",
        as_of="2020-01-08T00:00:00+08:00",
        role=DataRole.RESEARCH,
        frequency=frequency,
    )
    timestamps = pd.DatetimeIndex([f"{date} 15:00:00" for date in DATES]).tz_localize(
        "Asia/Shanghai"
    )
    rows = []
    for date_index, timestamp in enumerate(timestamps):
        for security_index, security in enumerate(SECURITIES):
            rows.append(
                {
                    "timestamp": timestamp,
                    "security": security,
                    "effective_at": timestamp,
                    "known_at": timestamp,
                    "Returns": float((date_index + 1) * (security_index + 1)) / 100,
                    "Amount": float(100 + 10 * security_index),
                }
            )
    frame = pd.DataFrame(rows)
    market = MarketDataView.from_batches(
        (
            DataBatch.create(
                sequence=0,
                snapshot=snapshot,
                schema=schema,
                request=request,
                availability=availability,
                frame=frame,
            ),
        )
    )

    events = pd.DataFrame(
        [
            {
                "instrument_id": security,
                "instrument_type": "index" if security.startswith("399") else "stock",
                "exchange": "XSHE",
                "status": "ACTIVE",
                "effective_date": "20100101",
                "known_at": "20100101",
            }
            for security in SECURITIES
        ]
    )
    signal = pd.DataFrame(True, index=DATES, columns=SECURITIES)
    returns = pd.DataFrame(0.01, index=DATES, columns=SECURITIES)
    close = pd.DataFrame(10.0, index=DATES, columns=SECURITIES)
    amount = pd.DataFrame(100.0, index=DATES, columns=SECURITIES)
    suspension = pd.DataFrame(False, index=DATES, columns=SECURITIES)
    price_limit = pd.DataFrame(True, index=DATES, columns=SECURITIES)
    contract = build_security_data_contract(
        DATES,
        SECURITIES,
        instrument_events=events,
        signal_universe_mask=signal,
        stock_returns=returns,
        stock_close=close,
        stock_amount=amount,
        suspension_events=suspension,
        price_limit_tradable_mask=price_limit,
        source_attestation=_full_attestation(),
        require_production_ready=True,
    )
    view = FactorDataView.from_market_view(
        market,
        contract,
        required_fields=("Returns", "Amount"),
    )
    return dataset, snapshot, schema, market, contract, view


def _spec(dataset, snapshot, schema, view, *, expression: str, fields: tuple[str, ...]):
    registry = OperatorRegistry.dataframe_pit_v3(
        view.cross_section_mask, view.history_mask
    )
    return FactorSpec(
        factor_id="research_test_factor",
        version="1.0.0",
        family="price_action",
        hypothesis="Past returns contain a short-lived cross-sectional continuation signal.",
        economic_rationale="Gradual information diffusion can preserve recent relative strength.",
        falsification_criterion="Reject when purged validation IC is unstable or non-positive.",
        expression=expression,
        direction=1,
        required_fields=fields,
        dataset_id=dataset.dataset_id,
        snapshot_id=snapshot.snapshot_id,
        schema_hash=schema.content_hash,
        availability_hash=hash_json(dataset.availability.to_dict()),
        security_contract_hash=view.security_contract_hash,
        frequency=dataset.frequency,
        aggregation=AggregationSpec(),
        operator_registry_version=registry.version,
        operator_registry_digest=registry.digest,
        preprocessing=(PreprocessStep(PreprocessKind.ZSCORE),),
        complexity_budget=ComplexityBudget(),
        provenance=FactorProvenance(origin="human", actor_id="research-test"),
    )


def test_factor_view_excludes_index_before_cross_section_and_engine_is_bound(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, _, _, view = _fixture(tmp_path)
    spec = _spec(
        dataset,
        snapshot,
        schema,
        view,
        expression="CsRank(TsMean(Returns, 2))",
        fields=("Returns",),
    )
    result = FactorEngine().evaluate(spec, view)
    assert result.factor_spec_hash == spec.content_hash
    assert result.admission_eligible is True
    assert result.signal["399001"].isna().all()
    valid_rows = result.signal[["000001", "000002"]].dropna(how="all")
    np.testing.assert_allclose(valid_rows.mean(axis=1).to_numpy(), 0.0, atol=1e-12)


def test_factor_and_label_factories_share_canonical_security_contract_identity(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, market, contract, factor_view = _fixture(tmp_path)
    label_view = LabelDataView.from_market_view(
        market,
        contract,
        required_fields=("Returns",),
        field_observation_events={
            "Returns": PriceObservationEvent.SESSION_CLOSE,
        },
    )

    canonical_hash = security_data_contract_hash(contract)
    assert factor_view.security_contract_hash == canonical_hash
    assert label_view.security_contract_hash == canonical_hash

    changed_signal = contract.signal_universe_mask.copy(deep=True)
    changed_signal.loc[DATES[0], "000001"] = False
    changed_contract = replace(contract, signal_universe_mask=changed_signal)
    changed_hash = security_data_contract_hash(changed_contract)
    assert changed_hash != canonical_hash

    changed_factor_view = FactorDataView.from_market_view(
        market,
        changed_contract,
        required_fields=("Returns",),
    )
    changed_label_view = LabelDataView.from_market_view(
        market,
        changed_contract,
        required_fields=("Returns",),
        field_observation_events={
            "Returns": PriceObservationEvent.SESSION_CLOSE,
        },
    )
    assert changed_factor_view.security_contract_hash == changed_hash
    assert changed_label_view.security_contract_hash == changed_hash

    original_spec = _spec(
        dataset,
        snapshot,
        schema,
        factor_view,
        expression="CsRank(Returns)",
        fields=("Returns",),
    )
    with pytest.raises(ValueError, match="security contract binding"):
        FactorEngine().evaluate(original_spec, changed_factor_view)


def test_factor_spec_semantic_hash_ignores_commutative_reordering(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, _, _, view = _fixture(tmp_path)
    first = _spec(
        dataset,
        snapshot,
        schema,
        view,
        expression="Add(Returns, Amount)",
        fields=("Returns", "Amount"),
    )
    second = _spec(
        dataset,
        snapshot,
        schema,
        view,
        expression="Add(Amount, Returns)",
        fields=("Amount", "Returns"),
    )
    assert first.semantic_hash == second.semantic_hash
    assert first.content_hash != second.content_hash


def test_factor_engine_rejects_snapshot_and_field_contract_drift(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, _, _, view = _fixture(tmp_path)
    spec = _spec(
        dataset,
        snapshot,
        schema,
        view,
        expression="CsRank(Returns)",
        fields=("Returns",),
    )
    altered = replace(spec, snapshot_id="f" * 64)
    with pytest.raises(ValueError, match="snapshot binding"):
        FactorEngine().evaluate(altered, view)

    changed_universe = replace(spec, security_contract_hash="e" * 64)
    assert changed_universe.semantic_hash == spec.semantic_hash
    assert changed_universe.definition_hash != spec.definition_hash
    with pytest.raises(ValueError, match="security contract binding"):
        FactorEngine().evaluate(changed_universe, view)

    misleading = _spec(
        dataset,
        snapshot,
        schema,
        view,
        expression="CsRank(Returns)",
        fields=("Returns", "Amount"),
    )
    with pytest.raises(ValueError, match="required_fields differ"):
        FactorEngine().evaluate(misleading, view)


def test_degraded_security_contract_requires_explicit_research_override(
    tmp_path: Path,
) -> None:
    _, _, _, market, contract, _ = _fixture(tmp_path)
    degraded = replace(
        contract,
        audit=replace(
            contract.audit,
            production_ready=False,
            mode="degraded_research_only",
            degraded_reasons=("test_unattested",),
        ),
    )
    with pytest.raises(PermissionError, match="not_production_ready"):
        FactorDataView.from_market_view(market, degraded, required_fields=("Returns",))
    view = FactorDataView.from_market_view(
        market,
        degraded,
        required_fields=("Returns",),
        allow_degraded_research=True,
    )
    assert view.production_ready is False
    assert view.degraded_reasons == ("test_unattested",)


def test_factor_view_structural_domain_can_only_narrow_signal_universe(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, market, contract, base = _fixture(tmp_path)
    domain = base.cross_section_mask.copy(deep=True)
    domain.loc[domain.index[0], "000002"] = False
    narrowed = FactorDataView.from_market_view_with_domain(
        market,
        contract,
        required_fields=("Returns",),
        domain_mask=domain,
    )
    spec = replace(
        _spec(
            dataset,
            snapshot,
            schema,
            narrowed,
            expression="Returns",
            fields=("Returns",),
        ),
        preprocessing=(),
    )
    result = FactorEngine().evaluate(spec, narrowed)
    assert pd.isna(result.signal.loc[domain.index[0], "000002"])
    assert narrowed.security_contract_hash == base.security_contract_hash
    assert narrowed.view_hash != base.view_hash

    expanded = domain.copy(deep=True)
    expanded.loc[expanded.index[0], "399001"] = True
    with pytest.raises(ValueError, match="exceeds"):
        FactorDataView.from_market_view_with_domain(
            market,
            contract,
            required_fields=("Returns",),
            domain_mask=expanded,
        )
    with pytest.raises(TypeError, match="complete boolean"):
        FactorDataView.from_market_view_with_domain(
            market,
            contract,
            required_fields=("Returns",),
            domain_mask=domain.astype(int),
        )


def test_factor_view_mutation_is_detected_before_execution(tmp_path: Path) -> None:
    dataset, snapshot, schema, _, _, view = _fixture(tmp_path)
    spec = _spec(
        dataset,
        snapshot,
        schema,
        view,
        expression="CsRank(Returns)",
        fields=("Returns",),
    )
    view.fields["Returns"].iloc[0, 0] = 999.0
    with pytest.raises(RuntimeError, match="content changed"):
        FactorEngine().evaluate(spec, view)


def test_factor_engine_masks_security_level_values_not_known_at_signal_time(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, market, contract, _ = _fixture(tmp_path)
    frame = market.frame.copy(deep=True)
    first_timestamp = pd.Timestamp(frame["timestamp"].min())
    late = (frame["timestamp"] == first_timestamp) & (frame["security"] == "000002")
    frame.loc[late, "known_at"] = first_timestamp + pd.Timedelta("1d")
    late_market = replace(market, frame=frame)
    late_view = FactorDataView.from_market_view(
        late_market,
        contract,
        required_fields=("Returns",),
    )
    spec = replace(
        _spec(
            dataset,
            snapshot,
            schema,
            late_view,
            expression="Returns",
            fields=("Returns",),
        ),
        preprocessing=(),
    )

    result = FactorEngine().evaluate(spec, late_view)
    assert result.signal.loc[first_timestamp, "000001"] == pytest.approx(0.01)
    assert pd.isna(result.signal.loc[first_timestamp, "000002"])
    assert result.admission_eligible is True

    # A value from a not-yet-known revision cannot change the early signal.
    revised_frame = frame.copy(deep=True)
    revised_frame.loc[late, "Returns"] = 999_999.0
    revised_view = FactorDataView.from_market_view(
        replace(market, frame=revised_frame),
        contract,
        required_fields=("Returns",),
    )
    revised_result = FactorEngine().evaluate(spec, revised_view)
    pd.testing.assert_frame_equal(result.signal, revised_result.signal)
    assert late_view.view_hash != revised_view.view_hash

    missing_frame = market.frame.copy(deep=True)
    missing_frame.loc[late, "known_at"] = pd.NaT
    missing_view = FactorDataView.from_market_view(
        replace(market, frame=missing_frame),
        contract,
        required_fields=("Returns",),
    )
    missing_result = FactorEngine().evaluate(spec, missing_view)
    assert pd.isna(missing_result.signal.loc[first_timestamp, "000002"])


def test_strict_engine_rejects_legacy_view_without_row_availability(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, _, _, strict_view = _fixture(tmp_path)
    fields = {"Returns": strict_view.field_panel("Returns")}
    payload = {
        "snapshot_id": strict_view.snapshot_id,
        "schema_hash": strict_view.schema_hash,
        "frequency_hash": strict_view.frequency_hash,
        "availability_hash": strict_view.availability_hash,
        "security_contract_hash": strict_view.security_contract_hash,
        "production_ready": True,
        "degraded_reasons": [],
        "fields": {"Returns": hash_frame(fields["Returns"])},
        "history_mask": hash_frame(strict_view.history_mask),
        "cross_section_mask": hash_frame(strict_view.cross_section_mask),
    }
    legacy_view = FactorDataView(
        snapshot_id=strict_view.snapshot_id,
        schema_hash=strict_view.schema_hash,
        frequency_hash=strict_view.frequency_hash,
        availability_hash=strict_view.availability_hash,
        security_contract_hash=strict_view.security_contract_hash,
        production_ready=True,
        degraded_reasons=(),
        fields=fields,
        history_mask=strict_view.history_mask,
        cross_section_mask=strict_view.cross_section_mask,
        view_hash=hash_json(payload),
    )
    spec = replace(
        _spec(
            dataset,
            snapshot,
            schema,
            legacy_view,
            expression="Returns",
            fields=("Returns",),
        ),
        preprocessing=(),
    )
    with pytest.raises(PermissionError, match="row_availability_required"):
        FactorEngine(require_point_in_time=True).evaluate(spec, legacy_view)

    # Compatibility replay remains possible, but cannot be admitted as a
    # production result without row-level PIT evidence.
    replay = FactorEngine().evaluate(spec, legacy_view)
    assert replay.admission_eligible is False


def test_future_label_field_and_complex_expression_are_rejected(tmp_path: Path) -> None:
    dataset, snapshot, schema, _, _, view = _fixture(tmp_path)
    with pytest.raises(ValueError, match="future_or_label_field_forbidden"):
        _spec(
            dataset,
            snapshot,
            schema,
            view,
            expression="CsRank(NextReturns)",
            fields=("NextReturns",),
        )

    registry = OperatorRegistry.dataframe_pit_v3(
        view.cross_section_mask, view.history_mask
    )
    complex_spec = FactorSpec(
        factor_id="too_complex",
        version="1",
        family="test",
        hypothesis="A deliberately excessive nested expression.",
        economic_rationale="Used to verify complexity admission controls.",
        falsification_criterion="Must be rejected before execution.",
        expression="Neg(Neg(Neg(Returns)))",
        direction=1,
        required_fields=("Returns",),
        dataset_id=dataset.dataset_id,
        snapshot_id=snapshot.snapshot_id,
        schema_hash=schema.content_hash,
        availability_hash=hash_json(dataset.availability.to_dict()),
        security_contract_hash=view.security_contract_hash,
        frequency=dataset.frequency,
        aggregation=AggregationSpec(),
        operator_registry_version=registry.version,
        operator_registry_digest=registry.digest,
        preprocessing=(),
        complexity_budget=ComplexityBudget(maximum_call_depth=2),
        provenance=FactorProvenance(origin="human", actor_id="research-test"),
    )
    with pytest.raises(ValueError, match="max_call_depth_exceeded"):
        FactorEngine().evaluate(complex_spec, view)


def test_minute_signal_calendar_daily_aggregation_is_executed_not_just_declared() -> (
    None
):
    timestamps = pd.DatetimeIndex(
        [
            f"2020-01-{day:02d} {clock}"
            for day in (2, 3)
            for clock in ("09:30:00", "10:00:00", "14:30:00", "15:00:00")
        ],
        tz="Asia/Shanghai",
        name="timestamp",
    )
    values = pd.DataFrame(
        np.arange(len(timestamps) * 3, dtype=float).reshape(-1, 3),
        index=timestamps,
        columns=SECURITIES,
    )
    history = pd.DataFrame(True, index=timestamps, columns=SECURITIES)
    cross_section = history.copy()
    history["399001"] = False
    cross_section["399001"] = False
    fields = {"Returns": values.where(history)}
    frequency = FrequencySpec.minute()
    security_contract_hash = "a" * 64
    payload = {
        "snapshot_id": "b" * 64,
        "schema_hash": "c" * 64,
        "frequency_hash": frequency.content_hash,
        "availability_hash": "d" * 64,
        "security_contract_hash": security_contract_hash,
        "production_ready": True,
        "degraded_reasons": [],
        "fields": {"Returns": hash_frame(fields["Returns"])},
        "history_mask": hash_frame(history),
        "cross_section_mask": hash_frame(cross_section),
    }
    view = FactorDataView(
        snapshot_id="b" * 64,
        schema_hash="c" * 64,
        frequency_hash=frequency.content_hash,
        availability_hash="d" * 64,
        security_contract_hash=security_contract_hash,
        production_ready=True,
        degraded_reasons=(),
        fields=fields,
        history_mask=history,
        cross_section_mask=cross_section,
        view_hash=hash_json(payload),
    )
    registry = OperatorRegistry.dataframe_pit_v3(
        view.cross_section_mask, view.history_mask
    )
    spec = FactorSpec(
        factor_id="minute_close30_mean",
        version="1",
        family="microstructure",
        hypothesis="Late-session intraday returns summarize closing information flow.",
        economic_rationale="Closing activity can aggregate institutional order pressure.",
        falsification_criterion="Reject if purged daily validation IC is unstable.",
        expression="Returns",
        direction=1,
        required_fields=("Returns",),
        dataset_id="minute-test",
        snapshot_id=view.snapshot_id,
        schema_hash=view.schema_hash,
        availability_hash=view.availability_hash,
        security_contract_hash=view.security_contract_hash,
        frequency=frequency,
        aggregation=AggregationSpec(
            method="calendar_daily",
            window="close30",
            smoothing_span=0,
            reducer="mean",
        ),
        operator_registry_version=registry.version,
        operator_registry_digest=registry.digest,
        preprocessing=(),
        complexity_budget=ComplexityBudget(),
        provenance=FactorProvenance(origin="human", actor_id="research-test"),
    )
    result = FactorEngine().evaluate(spec, view)
    assert list(result.signal.index) == [timestamps[3], timestamps[7]]
    expected_first = values.loc[[timestamps[2], timestamps[3]], "000001"].mean()
    assert result.signal.iloc[0, 0] == pytest.approx(expected_first)
    assert result.signal["399001"].isna().all()


def test_v5_candidate_requires_explicit_data_and_research_bindings(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, _, _, view = _fixture(tmp_path)
    registry = OperatorRegistry.dataframe_pit_v3(
        view.cross_section_mask, view.history_mask
    )
    candidate = CandidateSpec(
        candidate_id="candidate_binding_test",
        hypothesis="Recent relative strength may persist briefly.",
        expression="CsRank(Returns)",
        direction=1,
        frequency="daily",
        generation=0,
        family="price_volume",
        required_fields=("Returns",),
        protocol_hash="a" * 64,
        provider="offline-structured-provider",
        aggregation={"method": "none", "window": "full_day", "smoothing_span": 0},
    )
    spec = factor_spec_from_candidate(
        candidate,
        economic_rationale="Gradual information diffusion can preserve relative moves.",
        falsification_criterion="Reject on unstable purged validation Rank IC.",
        dataset_id=dataset.dataset_id,
        snapshot_id=snapshot.snapshot_id,
        schema_hash=schema.content_hash,
        availability_hash=hash_json(dataset.availability.to_dict()),
        security_contract_hash=view.security_contract_hash,
        frequency=dataset.frequency,
        operator_registry_version=registry.version,
        operator_registry_digest=registry.digest,
    )
    assert spec.expression == candidate.expression
    assert spec.provenance.prompt_hash == candidate.protocol_hash
    assert FactorEngine().evaluate(spec, view).definition_hash == spec.definition_hash

    with pytest.raises(ValueError, match="research rationale"):
        factor_spec_from_candidate(
            candidate,
            economic_rationale="",
            falsification_criterion="Reject.",
            dataset_id=dataset.dataset_id,
            snapshot_id=snapshot.snapshot_id,
            schema_hash=schema.content_hash,
            availability_hash=hash_json(dataset.availability.to_dict()),
            security_contract_hash=view.security_contract_hash,
            frequency=dataset.frequency,
            operator_registry_version=registry.version,
            operator_registry_digest=registry.digest,
        )


def test_explicit_ols_neutralization_removes_bound_cross_sectional_exposure() -> None:
    timestamps = pd.date_range(
        "2020-02-03 15:00:00",
        periods=3,
        freq="B",
        tz="Asia/Shanghai",
        name="timestamp",
    )
    securities = pd.Index([f"{number:06d}" for number in range(1, 7)])
    size = pd.DataFrame(
        np.tile(np.arange(1, 7, dtype=float), (3, 1)),
        index=timestamps,
        columns=securities,
    )
    residual = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    returns = 3.0 * size + pd.DataFrame(
        np.tile(residual, (3, 1)), index=timestamps, columns=securities
    )
    mask = pd.DataFrame(True, index=timestamps, columns=securities)
    fields = {"Returns": returns, "Size": size}
    frequency = FrequencySpec.daily()
    payload = {
        "snapshot_id": "1" * 64,
        "schema_hash": "2" * 64,
        "frequency_hash": frequency.content_hash,
        "availability_hash": "3" * 64,
        "security_contract_hash": "4" * 64,
        "production_ready": True,
        "degraded_reasons": [],
        "fields": {name: hash_frame(value) for name, value in sorted(fields.items())},
        "history_mask": hash_frame(mask),
        "cross_section_mask": hash_frame(mask),
    }
    view = FactorDataView(
        snapshot_id="1" * 64,
        schema_hash="2" * 64,
        frequency_hash=frequency.content_hash,
        availability_hash="3" * 64,
        security_contract_hash="4" * 64,
        production_ready=True,
        degraded_reasons=(),
        fields=fields,
        history_mask=mask,
        cross_section_mask=mask,
        view_hash=hash_json(payload),
    )
    registry = OperatorRegistry.dataframe_pit_v3(mask, mask)
    spec = FactorSpec(
        factor_id="neutralized_returns",
        version="1",
        family="cross_sectional",
        hypothesis="The return residual after size removal contains distinct information.",
        economic_rationale="Linear size exposure is a risk component rather than stock alpha.",
        falsification_criterion="Reject if residual IC is not incrementally positive.",
        expression="Returns",
        direction=1,
        required_fields=("Returns", "Size"),
        dataset_id="neutralization-test",
        snapshot_id=view.snapshot_id,
        schema_hash=view.schema_hash,
        availability_hash=view.availability_hash,
        security_contract_hash=view.security_contract_hash,
        frequency=frequency,
        aggregation=AggregationSpec(),
        operator_registry_version=registry.version,
        operator_registry_digest=registry.digest,
        preprocessing=(
            PreprocessStep(
                PreprocessKind.NEUTRALIZE_OLS,
                exposure_fields=("Size",),
            ),
        ),
        complexity_budget=ComplexityBudget(),
        provenance=FactorProvenance(origin="human", actor_id="research-test"),
    )
    result = FactorEngine().evaluate(spec, view)
    for timestamp in timestamps:
        assert result.signal.loc[timestamp].mean() == pytest.approx(0.0, abs=1e-12)
        assert result.signal.loc[timestamp].corr(size.loc[timestamp]) == pytest.approx(
            0.0, abs=1e-12
        )


def test_factor_registry_is_append_only_semantically_deduplicated_and_lineaged(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, _, _, view = _fixture(tmp_path)
    parent = _spec(
        dataset,
        snapshot,
        schema,
        view,
        expression="CsRank(Returns)",
        fields=("Returns",),
    )
    with FactorRegistry(tmp_path / "factor-registry.sqlite3") as registry:
        assert registry.register(parent, registered_at=CREATED) == parent.content_hash
        assert registry.get(parent.content_hash) == parent
        assert registry.register(parent, registered_at=CREATED) == parent.content_hash

        alias = replace(parent, factor_id="same_semantics_alias", version="2")
        with pytest.raises(FactorRegistryConflict, match="executable definition"):
            registry.register(alias, registered_at=CREATED)

        # The expression-level semantic identity may be shared across a
        # genuinely different universe contract; its executable definition is
        # nevertheless distinct and receives a separate registry record.
        different_universe = replace(
            parent,
            factor_id="different_universe",
            version="1",
            security_contract_hash="e" * 64,
        )
        assert different_universe.semantic_hash == parent.semantic_hash
        assert different_universe.definition_hash != parent.definition_hash
        assert (
            registry.register(different_universe, registered_at=CREATED)
            == different_universe.content_hash
        )

        child = replace(
            parent,
            factor_id="research_child",
            version="1",
            expression="CsRank(TsMean(Returns, 2))",
            provenance=FactorProvenance(
                origin="mutation",
                actor_id="research-test",
                parent_factor_hashes=(parent.content_hash,),
            ),
        )
        registry.register(child, registered_at=CREATED)
        assert registry.list_hashes() == (
            parent.content_hash,
            different_universe.content_hash,
            child.content_hash,
        )
        assert registry.current_state(child.content_hash) == "registered"
        registry.transition(
            child.content_hash,
            to_state="validated",
            occurred_at=CREATED,
            evidence_hash="f" * 64,
        )
        assert registry.current_state(child.content_hash) == "validated"
        with pytest.raises(FactorRegistryConflict, match="invalid factor lifecycle"):
            registry.transition(
                child.content_hash,
                to_state="rejected",
                occurred_at=CREATED,
            )
        registry.record_empirical_correlation(
            parent.content_hash,
            child.content_hash,
            evaluation_hash="a" * 64,
            correlation=0.25,
            observation_count=100,
            recorded_at=CREATED,
        )

        orphan = replace(
            child,
            factor_id="research_orphan",
            expression="CsRank(TsMean(Returns, 3))",
            provenance=FactorProvenance(
                origin="mutation",
                actor_id="research-test",
                parent_factor_hashes=("e" * 64,),
            ),
        )
        with pytest.raises(FactorRegistryConflict, match="parent is not registered"):
            registry.register(orphan, registered_at=CREATED)


def test_factor_registry_batch_is_atomic_idempotent_and_supports_ordered_lineage(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, _, _, view = _fixture(tmp_path)
    parent = _spec(
        dataset,
        snapshot,
        schema,
        view,
        expression="CsRank(Returns)",
        fields=("Returns",),
    )
    child = replace(
        parent,
        factor_id="research_batch_child",
        expression="CsRank(TsMean(Returns, 2))",
        provenance=FactorProvenance(
            origin="mutation",
            actor_id="research-test",
            parent_factor_hashes=(parent.content_hash,),
        ),
    )
    path = tmp_path / "factor-registry-batch.sqlite3"
    with FactorRegistry(path) as registry:
        expected = (parent.content_hash, child.content_hash)
        assert (
            registry.register_many((parent, child), registered_at=CREATED) == expected
        )
        assert (
            registry.register_many((parent, child), registered_at=CREATED) == expected
        )
        assert registry.list_hashes() == expected
        assert registry.current_state(parent.content_hash) == "registered"
        assert registry.current_state(child.content_hash) == "registered"


def test_factor_registry_batch_rolls_back_on_conflict_and_rejects_bad_shape(
    tmp_path: Path,
) -> None:
    dataset, snapshot, schema, _, _, view = _fixture(tmp_path)
    first = _spec(
        dataset,
        snapshot,
        schema,
        view,
        expression="CsRank(Returns)",
        fields=("Returns",),
    )
    same_definition = replace(
        first,
        factor_id="research_batch_definition_conflict",
        version="2",
    )
    with FactorRegistry(tmp_path / "factor-registry-rollback.sqlite3") as registry:
        with pytest.raises(FactorRegistryConflict, match="executable definition"):
            registry.register_many((first, same_definition), registered_at=CREATED)
        assert registry.list_hashes() == ()
        with pytest.raises(TypeError, match="exact tuple"):
            registry.register_many([first], registered_at=CREATED)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must not be empty"):
            registry.register_many((), registered_at=CREATED)
        with pytest.raises(ValueError, match="duplicate"):
            registry.register_many((first, first), registered_at=CREATED)
