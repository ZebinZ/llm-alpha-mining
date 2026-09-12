from pathlib import Path

import pandas as pd
import pytest

from alpha_demo.run import run_demo, verify


def test_relocated_demo_preserves_reproducibility_and_risk_veto(tmp_path: Path):
    first = run_demo(tmp_path / "one")
    second = run_demo(tmp_path / "two")
    assert first["scientific_signature"] == second["scientific_signature"]
    assert first["accepted"] == ["low_variability", "short_reversal"]
    assert first["risk_veto_demonstrated"]
    assert first["network_calls"] == 0
    assert first["admission_decision_permitted"] is False
    signal = pd.read_parquet(tmp_path / "one/short_reversal.parquet")
    assert signal["399001"].isna().all()
    assert signal.iloc[:4].isna().all().all()
    assert signal.iloc[45:48]["000006"].isna().all()
    assert verify(tmp_path / "one")["status"] == "PASS"


def test_artifact_verification_rejects_changed_factor(tmp_path: Path):
    run_demo(tmp_path / "demo")
    path = tmp_path / "demo/short_reversal.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity"):
        verify(tmp_path / "demo")
