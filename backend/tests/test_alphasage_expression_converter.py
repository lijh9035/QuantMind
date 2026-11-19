import numpy as np
import pandas as pd

from backend.services.engine.alpha_agent.alphasage.expression_converter import (
    convert_expression,
    evaluate_semantic_check,
)


def test_converter_maps_alpha_greater_less_to_elementwise_min_max():
    greater = convert_expression("Greater($close, $open)")
    less = convert_expression("Less($close,$open)")

    assert greater.structurally_supported
    assert greater.canonical_expression == "fm:Max2($close,$open)"
    assert less.structurally_supported
    assert less.canonical_expression == "fm:Min2($close,$open)"


def test_converter_maps_only_proven_lag_and_delta_forms():
    ref = convert_expression("Ref($close, 5)")
    delta = convert_expression("TsDelta($close, 5)")

    assert ref.canonical_expression == "fm:Delay($close,5)"
    assert delta.canonical_expression == "fm:Delta($close,5)"
    future = convert_expression("Ref($close,-1)")
    assert not future.structurally_supported
    assert "future" in future.reject_reason.lower()


def test_converter_expands_documented_combinations_without_changing_operator():
    slog = convert_expression("SLog1p($close)")
    inv = convert_expression("Inv($close)")
    max_diff = convert_expression("TsMaxDiff($close, 5)")
    min_max_diff = convert_expression("TsMinMaxDiff($close, 5)")

    assert slog.canonical_expression == ("fm:Mul(Sign($close),Log(Add(Abs($close),1)))")
    assert inv.canonical_expression == "fm:Div(1,$close)"
    assert max_diff.canonical_expression == "fm:Sub($close,TsMax($close,5))"
    assert min_max_diff.canonical_expression == (
        "fm:Sub(TsMax($close,5),TsMin($close,5))"
    )


def test_converter_rejects_operators_without_exact_fm_equivalent():
    for source in (
        "Rank($close)",
        "TsRank($close,5)",
        "TsEMA($close,5)",
        "TsWMA($close,5)",
        "TsKurt($close,5)",
        "TsSum($close,5)",
        "Pow($close,0.5)",
    ):
        converted = convert_expression(source)
        assert not converted.structurally_supported, source
        assert converted.canonical_expression is None, source
        assert converted.reject_reason, source


def test_converter_rejects_python_syntax_unknown_fields_and_resource_overflow():
    for source in (
        "__import__('os').system('whoami')",
        "Rank.__call__($close)",
        "$unknown",
        "TsMean($close,1001)",
        "TsMean($close,0)",
        "TsMean($close,2.5)",
        "fm:Abs($close)",
    ):
        assert not convert_expression(source).structurally_supported, source


def _fake_runtime(values, date_values, symbol_values):
    def evaluate(_expression, *, instruments, start, end, qlib_dir, **kwargs):
        assert instruments == symbol_values
        assert start == date_values[0]
        assert end == date_values[-1]
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(date_values), symbol_values],
            names=["trade_date", "symbol"],
        )
        return pd.Series(np.asarray(values, dtype=float).reshape(-1), index=index)

    return evaluate


def test_semantic_check_aligns_reordered_runtime_grid_to_source_order():
    dates = ["2024-01-02", "2024-01-03"]
    symbols = ["sh600000", "sz000001"]

    def reversed_runtime(_expression, *, instruments, start, end, qlib_dir, **kwargs):
        assert instruments == symbols
        assert start == dates[0]
        assert end == dates[-1]
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(dates[::-1]), symbols[::-1]],
            names=["trade_date", "symbol"],
        )
        return pd.Series([4.0, 3.0, 2.0, 1.0], index=index)

    report = evaluate_semantic_check(
        "fm:Abs($close)",
        dates=dates,
        symbols=symbols,
        source_values=[[1.0, 2.0], [3.0, 4.0]],
        evaluate_window_fn=reversed_runtime,
        calendar_fn=lambda start, end: dates,
    )
    assert report["passed"]
    assert report["compared_values"] == 4


def test_semantic_check_passes_same_values_and_returns_comparison_evidence():
    dates = ["2024-01-02", "2024-01-03"]
    symbols = ["sh600000", "sz000001"]
    values = [[1.0, 2.0], [3.0, 4.0]]
    report = evaluate_semantic_check(
        "fm:Add($close,1)",
        dates=dates,
        symbols=symbols,
        source_values=values,
        evaluate_window_fn=_fake_runtime(values, dates, symbols),
        calendar_fn=lambda start, end: dates,
    )

    assert report["passed"]
    assert report["coverage"] == 1.0
    assert report["compared_values"] == 4
    assert report["mask_mismatches"] == 0
    assert report["value_mismatches"] == 0
    assert report["max_error"] == 0.0


def test_semantic_check_rejects_mask_and_value_mismatches():
    dates = ["2024-01-02", "2024-01-03"]
    symbols = ["sh600000"]
    source = [[1.0], [None]]
    target = [[1.01], [2.0]]
    report = evaluate_semantic_check(
        "fm:Abs($close)",
        dates=dates,
        symbols=symbols,
        source_values=source,
        evaluate_window_fn=_fake_runtime(target, dates, symbols),
        calendar_fn=lambda start, end: dates,
    )

    assert not report["passed"]
    assert report["mask_mismatches"] == 1
    assert report["value_mismatches"] == 1
    assert report["reject_reason"] == (
        "Source and FM NaN/positive-infinity/negative-infinity masks differ"
    )


def test_semantic_check_distinguishes_nan_positive_and_negative_infinity_masks():
    dates = ["2024-01-02", "2024-01-03"]
    symbols = ["sh600000", "sz000001"]
    source = [[1.0, None], ["inf", "-inf"]]
    same_categories = [[1.0, np.nan], [np.inf, -np.inf]]
    matching = evaluate_semantic_check(
        "fm:Abs($close)",
        dates=dates,
        symbols=symbols,
        source_values=source,
        evaluate_window_fn=_fake_runtime(same_categories, dates, symbols),
        calendar_fn=lambda start, end: dates,
    )
    assert matching["passed"]
    assert matching["mask_mismatches"] == 0
    assert matching["source_nan_values"] == 1
    assert matching["source_positive_inf_values"] == 1
    assert matching["source_negative_inf_values"] == 1

    source_inf_target_nan = evaluate_semantic_check(
        "fm:Abs($close)",
        dates=dates,
        symbols=symbols,
        source_values=[[1.0, None], ["inf", -2.0]],
        evaluate_window_fn=_fake_runtime(
            [[1.0, np.nan], [np.nan, -2.0]], dates, symbols
        ),
        calendar_fn=lambda start, end: dates,
    )
    assert not source_inf_target_nan["passed"]
    assert source_inf_target_nan["mask_mismatches"] == 1
    assert source_inf_target_nan["nan_mask_mismatches"] == 1
    assert source_inf_target_nan["positive_inf_mask_mismatches"] == 1

    opposite_infinity_sign = evaluate_semantic_check(
        "fm:Abs($close)",
        dates=dates,
        symbols=symbols,
        source_values=[[1.0, None], ["inf", -2.0]],
        evaluate_window_fn=_fake_runtime(
            [[1.0, np.nan], [-np.inf, -2.0]], dates, symbols
        ),
        calendar_fn=lambda start, end: dates,
    )
    assert not opposite_infinity_sign["passed"]
    assert opposite_infinity_sign["mask_mismatches"] == 1
    assert opposite_infinity_sign["positive_inf_mask_mismatches"] == 1
    assert opposite_infinity_sign["negative_inf_mask_mismatches"] == 1


def test_semantic_check_rejects_sparse_or_invalid_training_sample():
    dates = ["2024-01-02", "2024-01-03"]
    symbols = ["sh600000"]
    all_missing = [[None], [None]]
    report = evaluate_semantic_check(
        "fm:Abs($close)",
        dates=dates,
        symbols=symbols,
        source_values=all_missing,
        evaluate_window_fn=_fake_runtime(all_missing, dates, symbols),
        calendar_fn=lambda start, end: dates,
    )
    assert not report["passed"]
    assert report["reject_reason"] == "No comparable finite values"

    out_of_training = evaluate_semantic_check(
        "fm:Abs($close)",
        dates=dates,
        symbols=symbols,
        source_values=[[1.0], [2.0]],
        training_end="2024-01-02",
        evaluate_window_fn=_fake_runtime([[1.0], [2.0]], dates, symbols),
        calendar_fn=lambda start, end: dates,
    )
    assert not out_of_training["passed"]
    assert "training_end" in out_of_training["reject_reason"]


def test_semantic_check_requires_continuous_sessions_and_exact_grid():
    dates = ["2024-01-02", "2024-01-04"]
    symbols = ["sh600000"]
    report = evaluate_semantic_check(
        "fm:Abs($close)",
        dates=dates,
        symbols=symbols,
        source_values=[[1.0], [2.0]],
        evaluate_window_fn=_fake_runtime([[1.0], [2.0]], dates, symbols),
        calendar_fn=lambda start, end: ["2024-01-02", "2024-01-03", "2024-01-04"],
    )
    assert not report["passed"]
    assert "continuous" in report["reject_reason"]

    too_many = [f"sh{n:06d}" for n in range(33)]
    oversized = evaluate_semantic_check(
        "fm:Abs($close)",
        dates=["2024-01-02"],
        symbols=too_many,
        source_values=[[1.0] * len(too_many)],
        evaluate_window_fn=lambda *args, **kwargs: pd.Series(dtype=float),
        calendar_fn=lambda start, end: ["2024-01-02"],
    )
    assert not oversized["passed"]
    assert "32" in oversized["reject_reason"]
