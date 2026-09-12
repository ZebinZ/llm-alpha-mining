from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_alpha_mining.mining.dsl import OperatorRegistry, SafeExpressionInterpreter


def _fields() -> dict[str, pd.DataFrame]:
    index = ["20200101", "20200102", "20200103", "20200104"]
    columns = ["000001", "000002", "000003"]
    values = np.arange(1, 13, dtype=float).reshape(4, 3) / 100.0
    return {"Returns": pd.DataFrame(values, index=index, columns=columns)}


def test_safe_interpreter_matches_pandas_for_allowed_formula() -> None:
    expression = "CsRank(TsMean(Returns, 2))"
    fields = _fields()
    expected = (
        fields["Returns"].rolling(window=2, min_periods=2).mean().rank(axis=1, pct=True)
    )
    actual = SafeExpressionInterpreter().evaluate(expression, fields)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "expression,reason",
    [
        ("Returns.__class__", "syntax_node_forbidden:Attribute"),
        ("__import__(1)", "operator_not_registered:__import__"),
        ("TsMean(Returns, 0)", "window_must_be_positive:TsMean:0"),
        ("CsRank(NextReturns)", "future_or_label_field_forbidden:NextReturns"),
    ],
)
def test_safe_interpreter_rejects_code_and_causal_violations(
    expression: str,
    reason: str,
) -> None:
    validation = SafeExpressionInterpreter().validate(
        expression,
        allowed_fields={"Returns", "NextReturns"},
    )
    assert not validation.is_valid
    assert reason in validation.reasons


def test_operator_registry_digest_changes_with_semantics() -> None:
    first = OperatorRegistry.dataframe_v1()
    second = OperatorRegistry(dict(first._specs), version="dataframe_v2")
    assert first.digest != second.digest


def test_pit_registry_prevents_conditional_zero_from_reentering_csrank() -> None:
    dates = pd.Index(["20200102", "20200103"], name="date")
    columns = pd.Index(["stock_a", "stock_b", "399001"], name="stock")
    returns = pd.DataFrame(
        [[-0.02, 0.01, np.nan], [-0.01, 0.02, np.nan]],
        index=dates,
        columns=columns,
    )
    mask = pd.DataFrame(
        [[True, True, False], [True, True, False]],
        index=dates,
        columns=columns,
    )
    interpreter = SafeExpressionInterpreter(
        OperatorRegistry.dataframe_pit_v1(mask),
        max_call_depth=6,
    )

    result = interpreter.evaluate(
        "CsRank(IfElse(Less(Returns, 0), Mul(Returns, Returns), 0))",
        {"Returns": returns},
    )

    assert result["399001"].isna().all()
    assert result.loc["20200102", "stock_a"] == 1.0
    assert result.loc["20200102", "stock_b"] == 0.5


def test_pit_v2_keeps_stock_history_but_excludes_it_from_current_rank() -> None:
    dates = pd.Index(["20200102", "20200103"], name="date")
    columns = pd.Index(["stock_a", "stock_b"], name="stock")
    returns = pd.DataFrame(
        [[-0.02, -0.01], [-0.03, -0.02]],
        index=dates,
        columns=columns,
    )
    security_master = pd.DataFrame(True, index=dates, columns=columns)
    cross_section = security_master.copy()
    cross_section.loc["20200102", "stock_b"] = False
    interpreter = SafeExpressionInterpreter(
        OperatorRegistry.dataframe_pit_v2(cross_section, security_master),
        max_call_depth=6,
    )

    conditional = interpreter.evaluate(
        "IfElse(Less(Returns, 0), Mul(Returns, Returns), 0)",
        {"Returns": returns},
    )
    ranked = interpreter.evaluate(
        "CsRank(IfElse(Less(Returns, 0), Mul(Returns, Returns), 0))",
        {"Returns": returns},
    )

    assert conditional.loc["20200102", "stock_b"] == 0.0001
    assert pd.isna(ranked.loc["20200102", "stock_b"])
    assert ranked.loc["20200103", "stock_b"] == 0.5
