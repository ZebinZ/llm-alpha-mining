from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_research.core.data import (
    DataBatch,
    DataRequest,
    DataRole,
    DatasetSnapshot,
    DatasetSpec,
    SourceAsset,
    standard_market_schema,
)
from alpha_research.core.frequency import (
    AvailabilitySpec,
    FrequencySpec,
    SessionSpec,
    TradingCalendar,
)
from alpha_research.core.hashing import hash_file
from alpha_research.data.quality import (
    DataQualityEngine,
    QualityDecision,
    QuarantineStore,
)


def _fixture(tmp_path: Path):
    source = tmp_path / "raw.bin"
    source.write_bytes(b"raw")
    schema = standard_market_schema(
        schema_id="quality-test",
        version="1",
        feature_fields={"Close": "float"},
    )
    frequency = FrequencySpec.minute()
    availability = AvailabilitySpec.market_data()
    dataset = DatasetSpec(
        dataset_id="quality",
        version="1",
        owner="research",
        description="quality fixture",
        storage_format="memory",
        schema_hash=schema.content_hash,
        frequency=frequency,
        availability=availability,
    )
    asset = SourceAsset.from_path(
        source, logical_name="raw", expected_sha256=hash_file(source)
    )
    snapshot = DatasetSnapshot.create(
        dataset=dataset,
        schema=schema,
        source_assets=(asset,),
        source_vintage_at="2026-07-18T00:00:00+00:00",
        created_at="2026-07-19T00:00:00+00:00",
    )
    request = DataRequest(
        dataset_id=dataset.dataset_id,
        snapshot_id=snapshot.snapshot_id,
        fields=("Close",),
        start="2020-01-02T09:30:00+08:00",
        end="2020-01-02T15:00:00+08:00",
        as_of="2020-01-03T00:00:00+08:00",
        role=DataRole.RESEARCH,
        frequency=frequency,
    )
    calendar = TradingCalendar(
        calendar_id="SSE_SZSE",
        timezone="Asia/Shanghai",
        sessions=("20200102",),
        session=SessionSpec.cn_a_share_regular(),
    )
    timestamps = pd.DatetimeIndex(
        ["2020-01-02 09:30:00", "2020-01-02 09:30:00"]
    ).tz_localize("Asia/Shanghai")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "security": ["000001", "000002"],
            "effective_at": timestamps,
            "known_at": timestamps,
            "Close": pd.Series([10.0, 20.0], dtype=float),
        }
    )
    return schema, dataset, snapshot, request, calendar, frame


def _batch(schema, dataset, snapshot, request, frame) -> DataBatch:
    return DataBatch.create(
        sequence=0,
        snapshot=snapshot,
        schema=schema,
        request=request,
        availability=dataset.availability,
        frame=frame,
    )


def test_clean_batch_is_accepted(tmp_path: Path) -> None:
    schema, dataset, snapshot, request, calendar, frame = _fixture(tmp_path)
    report = DataQualityEngine().evaluate(
        _batch(schema, dataset, snapshot, request, frame),
        schema=schema,
        calendar=calendar,
    )
    assert report.decision is QualityDecision.ACCEPTED
    assert report.issues == ()


def test_duplicate_missing_infinity_and_wrong_session_are_quarantined(
    tmp_path: Path,
) -> None:
    schema, dataset, snapshot, request, calendar, frame = _fixture(tmp_path)
    bad = pd.concat([frame, frame.iloc[[0]], frame.iloc[[1]]], ignore_index=True)
    bad.loc[0, "Close"] = np.nan
    bad.loc[1, "Close"] = np.inf
    bad.loc[2, "Close"] = np.nan
    noon = pd.Timestamp("2020-01-02 12:00:00", tz="Asia/Shanghai")
    bad.loc[3, ["timestamp", "effective_at", "known_at"]] = noon
    bad = bad.sort_values(["timestamp", "security"], kind="stable").reset_index(drop=True)
    batch = _batch(schema, dataset, snapshot, request, bad)
    report = DataQualityEngine().evaluate(
        batch, schema=schema, calendar=calendar
    )
    codes = {issue.code for issue in report.issues}
    assert report.decision is QualityDecision.QUARANTINED
    assert "duplicate_timestamp_security" in codes
    assert "excessive_missing_values" in codes
    assert "infinite_numeric_value" in codes
    assert "timestamp_outside_calendar_or_session" in codes

    receipt = QuarantineStore(tmp_path / "quarantine").record(batch, report)
    assert receipt.decision is QualityDecision.QUARANTINED
    assert receipt.quarantined_artifact is not None
    assert Path(receipt.quarantined_artifact).is_file()


def test_naive_timestamp_fails_before_batch_admission(tmp_path: Path) -> None:
    schema, dataset, snapshot, request, _, frame = _fixture(tmp_path)
    naive = frame.copy()
    for field in ("timestamp", "effective_at", "known_at"):
        naive[field] = naive[field].dt.tz_localize(None)
    with pytest.raises(TypeError, match="dtype_mismatch"):
        _batch(schema, dataset, snapshot, request, naive)


def test_batch_mutation_is_detected_before_quality_evaluation(tmp_path: Path) -> None:
    schema, dataset, snapshot, request, calendar, frame = _fixture(tmp_path)
    batch = _batch(schema, dataset, snapshot, request, frame)
    batch.frame.loc[0, "Close"] = 999.0
    with pytest.raises(RuntimeError, match="content changed"):
        DataQualityEngine().evaluate(batch, schema=schema, calendar=calendar)


def test_ohlc_and_corporate_action_sanity_checks_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "market.bin"
    source.write_bytes(b"market")
    schema = standard_market_schema(
        schema_id="market-sanity",
        version="1",
        feature_fields={
            "Open": "float",
            "High": "float",
            "Low": "float",
            "Close": "float",
            "Adj": "float",
        },
        bounds={"Adj": (float(np.nextafter(0.0, 1.0)), None)},
    )
    frequency = FrequencySpec.minute()
    dataset = DatasetSpec(
        dataset_id="market-sanity",
        version="1",
        owner="research",
        description="market sanity fixture",
        storage_format="memory",
        schema_hash=schema.content_hash,
        frequency=frequency,
        availability=AvailabilitySpec.market_data(),
    )
    snapshot = DatasetSnapshot.create(
        dataset=dataset,
        schema=schema,
        source_assets=(
            SourceAsset.from_path(
                source, logical_name="market", expected_sha256=hash_file(source)
            ),
        ),
        source_vintage_at="2026-07-18T00:00:00+00:00",
        created_at="2026-07-19T00:00:00+00:00",
    )
    request = DataRequest(
        dataset_id=dataset.dataset_id,
        snapshot_id=snapshot.snapshot_id,
        fields=("Open", "High", "Low", "Close", "Adj"),
        start="2020-01-02T09:30:00+08:00",
        end="2020-01-02T09:31:00+08:00",
        as_of="2020-01-03T00:00:00+08:00",
        role=DataRole.RESEARCH,
        frequency=frequency,
    )
    timestamps = pd.DatetimeIndex(
        ["2020-01-02 09:30:00", "2020-01-02 09:31:00"]
    ).tz_localize("Asia/Shanghai")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "security": ["000001", "000001"],
            "effective_at": timestamps,
            "known_at": timestamps,
            "Open": pd.Series([10.0, 10.0], dtype=float),
            "High": pd.Series([11.0, 9.0], dtype=float),
            "Low": pd.Series([9.0, 8.0], dtype=float),
            "Close": pd.Series([10.5, 10.0], dtype=float),
            "Adj": pd.Series([1.0, 101.0], dtype=float),
        }
    )
    calendar = TradingCalendar(
        calendar_id="SSE_SZSE",
        timezone="Asia/Shanghai",
        sessions=("20200102",),
        session=SessionSpec.cn_a_share_regular(),
    )
    report = DataQualityEngine().evaluate(
        DataBatch.create(
            sequence=0,
            snapshot=snapshot,
            schema=schema,
            request=request,
            availability=dataset.availability,
            frame=frame,
        ),
        schema=schema,
        calendar=calendar,
    )
    codes = {issue.code for issue in report.issues}
    assert report.decision is QualityDecision.QUARANTINED
    assert "inconsistent_ohlc_envelope" in codes
    assert "implausible_adjustment_step" in codes
