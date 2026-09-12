"""Run the real research components without market data, credentials or network."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_research.core.data import DataBatch, DataRequest, DataRole, DatasetSnapshot, DatasetSpec, SourceAsset, standard_market_schema
from alpha_research.core.frequency import AvailabilitySpec, FrequencySpec
from alpha_research.core.hashing import hash_file, hash_frame, hash_json
from alpha_research.data.views import MarketDataView
from alpha_research.evaluation.smoke import build_descriptive_smoke_diagnostics
from alpha_research.factors import FactorDataView, FactorEngine, factor_spec_from_v5_candidate
from factor_production.v5.data.security_contract import SecuritySourceAttestation, build_security_data_contract
from factor_production.v5.dsl import OperatorRegistry
from factor_production.v5.llm.agents import StructuredCallExecutor
from factor_production.v5.llm.budget import LLMBudget, LLMBudgetLimits
from factor_production.v5.llm.domain import LLMRole
from factor_production.v5.llm.ledger import CallLedger
from factor_production.v5.llm.panel import AlphaResearchPanel
from factor_production.v5.llm.policy import LLMPolicy
from factor_production.v5.llm.transport.fake import FakeTransport
from factor_production.v5.protocol import default_protocol
from factor_production.v5.providers.base import ProposalContext
from factor_production.v5.providers.llm_panel import LLMPanelProvider


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scripted_roles() -> dict:
    """Fixed teaching examples; these are not newly generated LLM hypotheses."""
    definitions = (
        ("short_reversal", "Neg(TsMean(Returns, 5))", ["Returns"]),
        ("low_variability", "Neg(TsStd(Returns, 10))", ["Returns"]),
        ("risk_blocked_example", "Div(Returns, Amount)", ["Returns", "Amount"]),
    )
    proposals = [{
        "candidate_id": name,
        "hypothesis": "Short-lived price pressure can leave a cross-sectional signal.",
        "expression": expression, "direction": 1, "frequency": "daily",
        "family": "behavioral_reversal", "required_fields": fields, "parent_ids": [],
        "aggregation": {"method": "none", "window": "full_day", "smoothing_span": 0},
        "tags": ["mechanism_first"],
    } for name, expression, fields in definitions]
    return {
        LLMRole.PROPOSER: [{"schema_version": "llm-proposer-output/v1", "proposals": proposals}],
        LLMRole.CRITIC: [{"schema_version": "llm-critic-output/v1", "reviews": [
            {"candidate_id": p["candidate_id"], "verdict": "approve", "reason_codes": ["mechanism_supported"]}
            for p in proposals]}],
        LLMRole.RISK: [{"schema_version": "llm-risk-output/v1", "assessments": [
            {"candidate_id": p["candidate_id"],
             "decision": "block" if p["candidate_id"] == "risk_blocked_example" else "allow",
             "reason_codes": ["unsafe_semantics" if p["candidate_id"] == "risk_blocked_example" else "no_blocker"]}
            for p in proposals]}],
        LLMRole.ARBITER: [{"schema_version": "llm-arbiter-output/v1", "decisions": [
            {"candidate_id": p["candidate_id"], "decision": "select",
             "reason_codes": ["balanced_mechanism"], "priority": 80}
            for p in proposals]}],
    }


def synthetic_view(out: Path, seed: int):
    rng = np.random.default_rng(seed)
    timestamps = pd.bdate_range("2020-01-02", periods=100, tz="Asia/Shanghai") + pd.Timedelta(hours=15)
    dates = tuple(timestamps.strftime("%Y%m%d"))
    securities = tuple(f"{i:06d}" for i in range(1, 41)) + ("399001",)
    returns = pd.DataFrame(rng.normal(0, .012, (100, 41)), index=dates, columns=securities)
    returns.iloc[30:33, 3] = np.nan
    amount = pd.DataFrame(rng.lognormal(14, .4, (100, 41)), index=dates, columns=securities)
    source = out / "synthetic_source.json"
    dump(source, {"schema": "synthetic-source/v1", "seed": seed, "dates": list(dates),
                  "securities": list(securities), "returns_hash": hash_frame(returns), "amount_hash": hash_frame(amount)})
    schema = standard_market_schema(schema_id="synthetic-daily", version="1", feature_fields={"Returns": "float", "Amount": "float"})
    frequency, availability = FrequencySpec.daily(), AvailabilitySpec.market_data()
    dataset = DatasetSpec(dataset_id="synthetic-only", version="1", owner="demo", description="Generated demonstration; no real market observations",
                          storage_format="memory", schema_hash=schema.content_hash, frequency=frequency, availability=availability)
    snapshot = DatasetSnapshot.create(dataset=dataset, schema=schema,
        source_assets=(SourceAsset.from_path(source, logical_name="synthetic-source", expected_sha256=hash_file(source)),),
        source_vintage_at="2020-12-30T00:00:00+08:00", created_at="2020-12-31T00:00:00+08:00")
    request = DataRequest(dataset_id=dataset.dataset_id, snapshot_id=snapshot.snapshot_id, fields=("Returns", "Amount"),
                         start=timestamps[0].isoformat(), end=timestamps[-1].isoformat(), as_of="2020-12-31T00:00:00+08:00",
                         role=DataRole.RESEARCH, frequency=frequency)
    frame = pd.DataFrame({"timestamp": timestamps.repeat(len(securities)), "security": list(securities) * len(dates),
                          "Returns": returns.to_numpy().ravel(), "Amount": amount.to_numpy().ravel()})
    frame["known_at"] = frame["timestamp"]
    frame["effective_at"] = frame["timestamp"]
    market = MarketDataView.from_batches((DataBatch.create(sequence=0, snapshot=snapshot, schema=schema, request=request, availability=availability, frame=frame),))
    events = pd.DataFrame([{"instrument_id": s, "instrument_type": "index" if s == "399001" else "stock",
                            "exchange": "XSHE", "status": "ACTIVE", "effective_date": "20100101", "known_at": "20100101"} for s in securities])
    universe = pd.DataFrame(True, index=dates, columns=securities)
    universe.iloc[45:48, 5] = False
    # The attestation describes the generated fixture only, not an external feed.
    attestation = SecuritySourceAttestation(source_id="synthetic-fixture", schema_version="1",
        instrument_type_authoritative=True, exchange_namespace_authoritative=True,
        point_in_time_status_history=True, delisting_history_complete=True, signal_universe_point_in_time=True,
        suspension_events_point_in_time=True, suspension_absence_means_not_suspended=True,
        market_data_point_in_time=True, price_limit_status_point_in_time=True)
    contract = build_security_data_contract(dates, securities, instrument_events=events, signal_universe_mask=universe,
        stock_returns=returns, stock_close=pd.DataFrame(10., index=dates, columns=securities), stock_amount=amount,
        suspension_events=pd.DataFrame(False, index=dates, columns=securities),
        price_limit_tradable_mask=pd.DataFrame(True, index=dates, columns=securities),
        source_attestation=attestation, require_production_ready=True)
    view = FactorDataView.from_market_view(market, contract, required_fields=("Returns", "Amount"))
    labels = returns.shift(-1)
    labels.index = timestamps
    return dataset, snapshot, schema, view, labels


def run_demo(output: Path, *, seed: int = 42) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = default_protocol("synthetic-offline-demo")
    fake = FakeTransport(scripted_roles())
    executor = StructuredCallExecutor(transport=fake, ledger=CallLedger(output / "llm_calls.jsonl"),
        budget=LLMBudget(LLMBudgetLimits(max_calls=4, max_input_tokens=100000, max_output_tokens=100000,
                                      max_total_tokens=200000, max_cost_microusd=1000000)), policy=LLMPolicy())
    provider = LLMPanelProvider(protocol=protocol, panel=AlphaResearchPanel(executor, max_expression_depth=protocol.max_expression_depth))
    batch = provider.propose(ProposalContext(run_id="synthetic-offline-demo", protocol_hash=protocol.content_hash, generation=0, request_count=3))
    dataset, snapshot, schema, view, labels = synthetic_view(output, seed)
    registry = OperatorRegistry.dataframe_pit_v3(view.cross_section_mask, view.history_mask)
    factors, specs = {}, []
    for candidate in batch.candidates:
        spec = factor_spec_from_v5_candidate(candidate, economic_rationale="Synthetic illustration of short-lived price pressure.",
            falsification_criterion="Reject on independent validation; the demo authorizes no admission.",
            dataset_id=dataset.dataset_id, snapshot_id=snapshot.snapshot_id, schema_hash=schema.content_hash,
            availability_hash=hash_json(dataset.availability.to_dict()), security_contract_hash=view.security_contract_hash,
            frequency=dataset.frequency, operator_registry_version=registry.version, operator_registry_digest=registry.digest)
        result = FactorEngine().evaluate(spec, view)
        factors[candidate.candidate_id] = result.signal
        specs.append(spec.to_dict())
        result.signal.to_parquet(output / f"{candidate.candidate_id}.parquet")
    labels = labels.reindex(index=next(iter(factors.values())).index, columns=next(iter(factors.values())).columns)
    domain = view.cross_section_mask.reindex(index=labels.index, columns=labels.columns)
    diagnostics = build_descriptive_smoke_diagnostics(factors=factors, labels=labels,
        factor_domain_masks={name: domain for name in factors}, full_universe_masks=domain,
        label_validity=np.isfinite(labels), minimum_observations=10)
    diagnostics.verify_content()
    diagnostics.per_date.to_parquet(output / "descriptive_metrics.parquet")
    diagnostics.pairwise_factor_correlation.to_parquet(output / "factor_correlations.parquet")
    dump(output / "candidate_specs.json", specs)
    scientific = {"seed": seed, "factor_hashes": {name: hash_frame(f) for name, f in sorted(factors.items())},
                  "diagnostics_hash": diagnostics.content_hash}
    report = {"status": "PASS", "synthetic_data": True, "transport": "fake-offline",
        "network_calls": 0, "scripted_role_calls": fake.call_count, "proposed": 3, "accepted": sorted(factors),
        "risk_veto_demonstrated": "risk_blocked_example" not in factors,
        "dates": len(labels), "synthetic_stocks": 40, "excluded_index": "399001",
        "formal_inference_permitted": False, "admission_decision_permitted": False,
        "scientific_signature": hash_json(scientific), **scientific}
    dump(output / "report.json", report)
    dump(output / "manifest.json", {"schema": "offline-demo-artifacts/v1", "files": [
        {"path": str(p.relative_to(output)), "sha256": hash_file(p), "bytes": p.stat().st_size}
        for p in sorted(output.iterdir()) if p.is_file()]})
    return report


def verify(output: Path) -> dict:
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    for row in manifest["files"]:
        path = (output / row["path"]).resolve()
        if not path.is_relative_to(output) or hash_file(path) != row["sha256"]:
            raise ValueError("artifact integrity mismatch")
    return {"status": "PASS", "verified_files": len(manifest["files"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.output) if args.verify else run_demo(args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
