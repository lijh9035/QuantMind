"""Safe AlphaSAGE expression conversion and per-candidate semantic verification.

This module never executes AlphaSAGE or Python source. It parses the small
function-call grammar emitted by the mining worker, converts only an explicit
operator allowlist to QuantMind FM-DSL, and compares candidate values against
the same Qlib-backed FM runtime before persistence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..factorminer import dsl

MAX_EXPRESSION_LENGTH = 8192
MAX_NODES = 128
MAX_DEPTH = 32
MAX_SAMPLE_DATES = 64
MAX_SAMPLE_SYMBOLS = 32
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4
DEFAULT_MIN_COVERAGE = 0.05

SOURCE_FEATURES = frozenset({"$open", "$high", "$low", "$close", "$volume", "$vwap"})


@dataclass(frozen=True)
class ConversionResult:
    """Structural conversion result; semantic acceptance is a separate gate."""

    raw_expression: str
    canonical_expression: str | None
    structurally_supported: bool
    reject_reason: str | None = None


@dataclass(frozen=True)
class _Num:
    value: float


@dataclass(frozen=True)
class _Feature:
    name: str


@dataclass(frozen=True)
class _Call:
    name: str
    args: tuple[_SourceNode, ...]


_SourceNode = _Num | _Feature | _Call


@dataclass(frozen=True)
class _OperatorSpec:
    arity: int
    window_index: int | None = None


_SOURCE_SPECS: dict[str, _OperatorSpec] = {
    "Abs": _OperatorSpec(1),
    "Sign": _OperatorSpec(1),
    "Log": _OperatorSpec(1),
    "Add": _OperatorSpec(2),
    "Sub": _OperatorSpec(2),
    "Mul": _OperatorSpec(2),
    "Div": _OperatorSpec(2),
    "Greater": _OperatorSpec(2),
    "Less": _OperatorSpec(2),
    "Ref": _OperatorSpec(2, 1),
    "TsDelta": _OperatorSpec(2, 1),
    "TsMean": _OperatorSpec(2, 1),
    "TsStd": _OperatorSpec(2, 1),
    "TsMax": _OperatorSpec(2, 1),
    "TsMin": _OperatorSpec(2, 1),
    "TsMed": _OperatorSpec(2, 1),
    "TsCorr": _OperatorSpec(3, 2),
    "TsCov": _OperatorSpec(3, 2),
    "SLog1p": _OperatorSpec(1),
    "Inv": _OperatorSpec(1),
    "TsMaxDiff": _OperatorSpec(2, 1),
    "TsMinDiff": _OperatorSpec(2, 1),
    "TsMinMaxDiff": _OperatorSpec(2, 1),
    "Rank": _OperatorSpec(1),
    "TsRank": _OperatorSpec(2, 1),
    "TsWMA": _OperatorSpec(2, 1),
    "TsEMA": _OperatorSpec(2, 1),
    "TsKurt": _OperatorSpec(2, 1),
    "TsSkew": _OperatorSpec(2, 1),
    "TsSum": _OperatorSpec(2, 1),
    "TsVar": _OperatorSpec(2, 1),
    "TsIR": _OperatorSpec(2, 1),
    "TsMad": _OperatorSpec(2, 1),
    "Pow": _OperatorSpec(2),
    "TsPctChange": _OperatorSpec(2, 1),
    "TsDiv": _OperatorSpec(3, 2),
}

# These mappings preserve the documented AlphaSAGE mathematical operation.
_DIRECT_UNARY = {"Abs": "Abs", "Sign": "Sign", "Log": "Log"}
_DIRECT_BINARY = {
    "Add": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
    # AlphaSAGE Greater/Less are elementwise max/min, not comparisons.
    "Greater": "Max2",
    "Less": "Min2",
}
_DIRECT_WINDOWS = {
    "TsMean": "Mean",
    "TsStd": "Std",
    "TsMax": "TsMax",
    "TsMin": "TsMin",
    "TsMed": "Med",
    "TsCorr": "Corr",
    "TsCov": "Cov",
}
_UNSUPPORTED_REASONS = {
    "Rank": "Rank tie handling and normalization differ from FM CsRank",
    "TsRank": "AlphaSAGE TsRank semantics are not proven equal to FM TsRank",
    "TsWMA": "AlphaSAGE TsWMA uses different weights from FM WMA",
    "TsEMA": "AlphaSAGE TsEMA uses finite-window weights unlike FM EMA",
    "TsKurt": "Kurtosis correction differs from FM Kurt",
    "TsSkew": "Skewness correction is not proven equal to FM Skew",
    "TsSum": "FM-DSL has no equivalent rolling sum operator",
    "TsVar": "Variance degrees of freedom and missing-value semantics are not proven",
    "TsIR": "Information-ratio denominator semantics are not proven equal",
    "TsMad": "FM-DSL has no proven equivalent median-absolute-deviation operator",
    "Pow": "SignedPower is not equivalent to ordinary power for negative bases",
    "TsPctChange": "Percent-change edge and denominator semantics are not proven",
    "TsDiv": "This AlphaSAGE operator has no proven equivalent FM mapping",
}

_TOKEN_RE = re.compile(
    r"\s*(?:(\$[A-Za-z][A-Za-z0-9]*)"
    r"|([A-Za-z][A-Za-z0-9]*)"
    r"|(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|([(),]))"
)
_Token = tuple[str, str]


class _UnsupportedExpression(ValueError):
    pass


class _SourceParser:
    def __init__(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Expression must be a non-empty string")
        if len(text) > MAX_EXPRESSION_LENGTH:
            raise ValueError(
                f"Expression exceeds the {MAX_EXPRESSION_LENGTH}-character limit"
            )
        self.tokens = self._tokenize(text)
        self.position = 0
        self.depth = 0
        self.node_count = 0

    @staticmethod
    def _tokenize(text: str) -> list[_Token]:
        tokens: list[_Token] = []
        position = 0
        while position < len(text):
            match = _TOKEN_RE.match(text, position)
            if match is None:
                if not text[position:].strip():
                    break
                raise ValueError(f"Unsupported token at character {position}")
            feature, identifier, number, punctuation = match.groups()
            if feature is not None:
                tokens.append(("FEATURE", feature))
            elif identifier is not None:
                tokens.append(("IDENT", identifier))
            elif number is not None:
                tokens.append(("NUMBER", number))
            else:
                assert punctuation is not None
                tokens.append(("PUNCT", punctuation))
            position = match.end()
            if len(tokens) > MAX_NODES * 4:
                raise ValueError("Expression has too many tokens")
        return tokens

    def peek(self) -> _Token | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def take(self) -> _Token:
        token = self.peek()
        if token is None:
            raise ValueError("Expression ended unexpectedly")
        self.position += 1
        return token

    def parse(self) -> _SourceNode:
        if not self.tokens:
            raise ValueError("Expression is empty")
        node = self.parse_expr()
        if self.peek() is not None:
            raise ValueError(f"Unexpected trailing token {self.peek()[1]!r}")
        return node

    def parse_expr(self) -> _SourceNode:
        token = self.take()
        kind, value = token
        self.node_count += 1
        if self.node_count > MAX_NODES:
            raise ValueError(f"Expression exceeds the {MAX_NODES}-node limit")
        if kind == "FEATURE":
            if value not in SOURCE_FEATURES:
                raise ValueError(f"Unsupported source field {value}")
            return _Feature(value)
        if kind == "NUMBER":
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Numeric constants must be finite")
            return _Num(number)
        if kind != "IDENT":
            raise ValueError(f"Unexpected token {value!r}")
        spec = _SOURCE_SPECS.get(value)
        if spec is None:
            raise ValueError(f"Unsupported operator {value}")
        if self.peek() != ("PUNCT", "("):
            raise ValueError(f"Operator {value} must be followed by '('")
        self.take()
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ValueError(f"Expression exceeds the {MAX_DEPTH}-level depth limit")
        args: list[_SourceNode] = []
        if self.peek() != ("PUNCT", ")"):
            while True:
                args.append(self.parse_expr())
                separator = self.take()
                if separator == ("PUNCT", ")"):
                    break
                if separator != ("PUNCT", ","):
                    raise ValueError(f"Expected ',' or ')' in {value}")
        else:
            self.take()
        self.depth -= 1
        if len(args) != spec.arity:
            raise ValueError(
                f"Operator {value} requires {spec.arity} arguments; got {len(args)}"
            )
        if spec.window_index is not None:
            window_node = args[spec.window_index]
            if not isinstance(window_node, _Num):
                raise ValueError(f"Operator {value} requires a literal window")
            window = window_node.value
            is_ref_lag = value == "Ref" and window < 0
            if window != int(window) or (window < 1 and not is_ref_lag):
                raise ValueError(f"Operator {value} window must be a positive integer")
            if abs(window) > dsl.MAX_WINDOW:
                raise ValueError(
                    f"Operator {value} window exceeds FM limit {dsl.MAX_WINDOW}"
                )
        return _Call(value, tuple(args))


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    return format(value, ".17g")


def _serialize(node: dsl.Node) -> str:
    if isinstance(node, dsl.Num):
        return _format_number(node.value)
    if isinstance(node, dsl.Feature):
        return node.name
    if isinstance(node, dsl.Call):
        return f"{node.op}({','.join(_serialize(arg) for arg in node.args)})"
    raise TypeError("Unsupported FM node")


def _to_target(node: _SourceNode) -> dsl.Node:
    if isinstance(node, _Num):
        return dsl.Num(node.value)
    if isinstance(node, _Feature):
        return dsl.Feature(node.name)
    if not isinstance(node, _Call):
        raise TypeError("Unsupported source node")

    if node.name in _UNSUPPORTED_REASONS:
        raise _UnsupportedExpression(f"{node.name}: {_UNSUPPORTED_REASONS[node.name]}")

    if node.name in _DIRECT_UNARY:
        return dsl.Call(_DIRECT_UNARY[node.name], (_to_target(node.args[0]),))
    if node.name in _DIRECT_BINARY:
        return dsl.Call(
            _DIRECT_BINARY[node.name],
            tuple(_to_target(arg) for arg in node.args),
        )
    if node.name == "Ref":
        lag = node.args[1]
        assert isinstance(lag, _Num)
        if lag.value < 0:
            raise _UnsupportedExpression("Ref with negative lag reads future data")
        return dsl.Call("Delay", (_to_target(node.args[0]), dsl.Num(lag.value)))
    if node.name == "TsDelta":
        window = node.args[1]
        assert isinstance(window, _Num)
        return dsl.Call("Delta", (_to_target(node.args[0]), dsl.Num(window.value)))
    if node.name in _DIRECT_WINDOWS:
        target_op = _DIRECT_WINDOWS[node.name]
        args = tuple(_to_target(arg) for arg in node.args)
        return dsl.Call(target_op, args)
    if node.name == "Inv":
        return dsl.Call("Div", (dsl.Num(1.0), _to_target(node.args[0])))
    if node.name == "SLog1p":
        value = _to_target(node.args[0])
        return dsl.Call(
            "Mul",
            (
                dsl.Call("Sign", (value,)),
                dsl.Call(
                    "Log",
                    (dsl.Call("Add", (dsl.Call("Abs", (value,)), dsl.Num(1.0))),),
                ),
            ),
        )
    if node.name in {"TsMaxDiff", "TsMinDiff", "TsMinMaxDiff"}:
        value = _to_target(node.args[0])
        window = node.args[1]
        assert isinstance(window, _Num)
        if node.name == "TsMaxDiff":
            right = dsl.Call("TsMax", (value, dsl.Num(window.value)))
            return dsl.Call("Sub", (value, right))
        if node.name == "TsMinDiff":
            right = dsl.Call("TsMin", (value, dsl.Num(window.value)))
            return dsl.Call("Sub", (value, right))
        high = dsl.Call("TsMax", (value, dsl.Num(window.value)))
        low = dsl.Call("TsMin", (value, dsl.Num(window.value)))
        return dsl.Call("Sub", (high, low))
    raise _UnsupportedExpression(
        f"{node.name}: no numerically equivalent FM operator mapping"
    )


def convert_expression(raw_expression: str) -> ConversionResult:
    """Parse an AlphaSAGE formula and produce a canonical FM expression.

    structurally_supported means only that the whitelist mapping exists. A
    caller must still run evaluate_semantic_check; this function never claims
    that a candidate passed the runtime value comparison.
    """
    if not isinstance(raw_expression, str):
        return ConversionResult(
            raw_expression="",
            canonical_expression=None,
            structurally_supported=False,
            reject_reason="Expression must be a string",
        )
    try:
        if raw_expression.startswith(dsl.FM_CODE_PREFIX):
            raise ValueError("Expected a raw AlphaSAGE expression, not an fm: formula")
        source = _SourceParser(raw_expression).parse()
        target = _to_target(source)
        canonical = f"{dsl.FM_CODE_PREFIX}{_serialize(target)}"
        dsl.parse(canonical[len(dsl.FM_CODE_PREFIX) :])
        return ConversionResult(raw_expression, canonical, True)
    except (ValueError, OverflowError, RecursionError) as exc:
        return ConversionResult(
            raw_expression=raw_expression,
            canonical_expression=None,
            structurally_supported=False,
            reject_reason=str(exc),
        )


def _date_index(values: Sequence[Any], label: str) -> pd.DatetimeIndex:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a date sequence")
    parsed = pd.to_datetime(list(values), errors="coerce")
    dates = pd.DatetimeIndex(parsed)
    if dates.hasnans or dates.tz is not None:
        raise ValueError(f"{label} contains an invalid or timezone-aware date")
    normalized = dates.normalize()
    if not dates.equals(normalized):
        raise ValueError(f"{label} must contain date-only values")
    if normalized.has_duplicates or not normalized.is_monotonic_increasing:
        raise ValueError(f"{label} must be strictly ascending without duplicates")
    return normalized


def _source_matrix(values: Any, rows: int, columns: int) -> np.ndarray:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("source_values must be a two-dimensional array")
    if len(values) != rows:
        raise ValueError("source_values row count does not match dates")
    matrix = np.full((rows, columns), np.nan, dtype=float)
    for row_index, row in enumerate(values):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("source_values must be a two-dimensional array")
        if len(row) != columns:
            raise ValueError("source_values column count does not match symbols")
        for column_index, value in enumerate(row):
            if value is None:
                # The worker's strict JSON payload uses null for source NaN.
                continue
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.number)
            ):
                if isinstance(value, str) and value.strip().lower() in {
                    "nan",
                    "inf",
                    "+inf",
                    "-inf",
                    "infinity",
                    "+infinity",
                    "-infinity",
                }:
                    matrix[row_index, column_index] = float(value)
                    continue
                raise ValueError("source_values may contain only numbers or nulls")
            matrix[row_index, column_index] = float(value)
    return matrix


def _result(
    *,
    passed: bool,
    reason: str | None,
    total_values: int,
    compared_values: int = 0,
    source_finite_values: int = 0,
    target_finite_values: int = 0,
    mask_mismatches: int = 0,
    nan_mask_mismatches: int = 0,
    positive_inf_mask_mismatches: int = 0,
    negative_inf_mask_mismatches: int = 0,
    source_nan_values: int = 0,
    target_nan_values: int = 0,
    source_positive_inf_values: int = 0,
    target_positive_inf_values: int = 0,
    source_negative_inf_values: int = 0,
    target_negative_inf_values: int = 0,
    value_mismatches: int = 0,
    max_error: float | None = None,
    atol: float,
    rtol: float,
    min_coverage: float,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "coverage": float(compared_values / total_values) if total_values else 0.0,
        "min_coverage": float(min_coverage),
        "total_values": int(total_values),
        "compared_values": int(compared_values),
        "source_finite_values": int(source_finite_values),
        "target_finite_values": int(target_finite_values),
        "mask_mismatches": int(mask_mismatches),
        "nan_mask_mismatches": int(nan_mask_mismatches),
        "positive_inf_mask_mismatches": int(positive_inf_mask_mismatches),
        "negative_inf_mask_mismatches": int(negative_inf_mask_mismatches),
        "source_nan_values": int(source_nan_values),
        "target_nan_values": int(target_nan_values),
        "source_positive_inf_values": int(source_positive_inf_values),
        "target_positive_inf_values": int(target_positive_inf_values),
        "source_negative_inf_values": int(source_negative_inf_values),
        "target_negative_inf_values": int(target_negative_inf_values),
        "value_mismatches": int(value_mismatches),
        "max_error": None if max_error is None else float(max_error),
        "atol": float(atol),
        "rtol": float(rtol),
        "reject_reason": reason,
    }


def evaluate_semantic_check(
    fm_expression: str,
    *,
    dates: Sequence[Any],
    symbols: Sequence[str],
    source_values: Any,
    qlib_dir: str | None = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    training_start: str | None = None,
    training_end: str | None = None,
    evaluate_window_fn: Callable[..., pd.Series] | None = None,
    calendar_fn: Callable[..., Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Compare native training samples with QuantMind FM values on the same grid.

    The caller supplies only training-period dates and AlphaSAGE source values.
    This function accepts no validation values and asks the FM runtime to read
    only the supplied date interval plus its formula-derived warmup.
    """
    if not isinstance(fm_expression, str) or not fm_expression.startswith(
        dsl.FM_CODE_PREFIX
    ):
        return _result(
            passed=False,
            reason="Expected a canonical fm: expression",
            total_values=0,
            atol=atol,
            rtol=rtol,
            min_coverage=min_coverage,
        )
    try:
        if not all(
            math.isfinite(float(value)) and float(value) >= 0 for value in (atol, rtol)
        ):
            raise ValueError("atol and rtol must be finite non-negative numbers")
        if not math.isfinite(float(min_coverage)) or not 0 < float(min_coverage) <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        expr = fm_expression[len(dsl.FM_CODE_PREFIX) :]
        dsl.parse(expr)
        date_index = _date_index(dates, "dates")
        if not 1 <= len(date_index) <= MAX_SAMPLE_DATES:
            raise ValueError(f"dates must contain 1 to {MAX_SAMPLE_DATES} sessions")
        if (
            not isinstance(symbols, Sequence)
            or isinstance(symbols, (str, bytes))
            or not 1 <= len(symbols) <= MAX_SAMPLE_SYMBOLS
        ):
            raise ValueError(
                f"symbols must contain 1 to {MAX_SAMPLE_SYMBOLS} instruments"
            )
        symbol_list = list(symbols)
        if any(
            not isinstance(symbol, str) or not symbol.strip() for symbol in symbol_list
        ):
            raise ValueError("symbols must be non-empty strings")
        if len(symbol_list) != len(set(symbol_list)):
            raise ValueError("symbols must not contain duplicates")
        if training_start is not None:
            start_limit = _date_index([training_start], "training_start")[0]
            if date_index[0] < start_limit:
                raise ValueError("semantic-check dates precede training_start")
        if training_end is not None:
            end_limit = _date_index([training_end], "training_end")[0]
            if date_index[-1] > end_limit:
                raise ValueError("semantic-check dates exceed training_end")
        source = _source_matrix(source_values, len(date_index), len(symbol_list))

        if calendar_fn is None and evaluate_window_fn is None:
            from ..statistical.panel import qlib_calendar

            def default_calendar(start, end):
                return qlib_calendar(start, end, qlib_dir=qlib_dir)

            calendar_fn = default_calendar
        if calendar_fn is not None:
            calendar = _date_index(
                list(
                    calendar_fn(
                        date_index[0].strftime("%Y-%m-%d"),
                        date_index[-1].strftime("%Y-%m-%d"),
                    )
                ),
                "Qlib calendar",
            )
            if not calendar.equals(date_index):
                raise ValueError(
                    "dates must be a continuous sequence of Qlib trading sessions"
                )

        if evaluate_window_fn is None:
            from ..factorminer.runtime import evaluate_window

            evaluate_window_fn = evaluate_window
        evaluated = evaluate_window_fn(
            fm_expression,
            instruments=symbol_list,
            start=date_index[0].strftime("%Y-%m-%d"),
            end=date_index[-1].strftime("%Y-%m-%d"),
            qlib_dir=qlib_dir,
            **({"calendar_fn": calendar_fn} if calendar_fn is not None else {}),
        )
        if not isinstance(evaluated, pd.Series):
            raise ValueError("FM runtime returned a non-Series result")
        if (
            not isinstance(evaluated.index, pd.MultiIndex)
            or evaluated.index.nlevels != 2
        ):
            raise ValueError("FM runtime returned an invalid date/symbol index")
        target = evaluated.rename("value").unstack(level="symbol")
        target.index = pd.DatetimeIndex(pd.to_datetime(target.index)).normalize()
        target.columns = target.columns.astype(str)
        target = target.reindex(index=date_index, columns=symbol_list)
        target_values = target.to_numpy(dtype=float, na_value=np.nan)
        source_finite = np.isfinite(source)
        target_finite = np.isfinite(target_values)
        source_nan = np.isnan(source)
        target_nan = np.isnan(target_values)
        source_positive_inf = np.isposinf(source)
        target_positive_inf = np.isposinf(target_values)
        source_negative_inf = np.isneginf(source)
        target_negative_inf = np.isneginf(target_values)
        source_category = np.zeros(source.shape, dtype=np.uint8)
        target_category = np.zeros(target_values.shape, dtype=np.uint8)
        source_category[source_nan] = 1
        target_category[target_nan] = 1
        source_category[source_positive_inf] = 2
        target_category[target_positive_inf] = 2
        source_category[source_negative_inf] = 3
        target_category[target_negative_inf] = 3
        mask_mismatches = int(np.count_nonzero(source_category != target_category))
        nan_mask_mismatches = int(np.count_nonzero(source_nan != target_nan))
        positive_inf_mask_mismatches = int(
            np.count_nonzero(source_positive_inf != target_positive_inf)
        )
        negative_inf_mask_mismatches = int(
            np.count_nonzero(source_negative_inf != target_negative_inf)
        )
        common = source_finite & target_finite
        compared_values = int(np.count_nonzero(common))
        total_values = int(source.size)
        source_finite_values = int(np.count_nonzero(source_finite))
        target_finite_values = int(np.count_nonzero(target_finite))
        max_error = None
        value_mismatches = 0
        if compared_values:
            with np.errstate(over="ignore", invalid="ignore"):
                errors = np.abs(target_values[common] - source[common])
                tolerances = float(atol) + float(rtol) * np.abs(source[common])
                matches = errors <= tolerances
            if np.all(np.isfinite(errors)):
                max_error = float(np.max(errors))
            else:
                max_error = None
                matches = np.isfinite(errors) & matches
            value_mismatches = int(np.count_nonzero(~matches))
        coverage = compared_values / total_values if total_values else 0.0
        reason = None
        if mask_mismatches:
            reason = (
                "Source and FM NaN/positive-infinity/negative-infinity masks differ"
            )
        elif compared_values == 0:
            reason = "No comparable finite values"
        elif coverage < float(min_coverage):
            reason = "Comparable finite coverage is below the required threshold"
        elif value_mismatches:
            reason = "One or more finite values exceed numeric tolerance"
        return _result(
            passed=reason is None,
            reason=reason,
            total_values=total_values,
            compared_values=compared_values,
            source_finite_values=source_finite_values,
            target_finite_values=target_finite_values,
            mask_mismatches=mask_mismatches,
            nan_mask_mismatches=nan_mask_mismatches,
            positive_inf_mask_mismatches=positive_inf_mask_mismatches,
            negative_inf_mask_mismatches=negative_inf_mask_mismatches,
            source_nan_values=int(np.count_nonzero(source_nan)),
            target_nan_values=int(np.count_nonzero(target_nan)),
            source_positive_inf_values=int(np.count_nonzero(source_positive_inf)),
            target_positive_inf_values=int(np.count_nonzero(target_positive_inf)),
            source_negative_inf_values=int(np.count_nonzero(source_negative_inf)),
            target_negative_inf_values=int(np.count_nonzero(target_negative_inf)),
            value_mismatches=value_mismatches,
            max_error=max_error,
            atol=atol,
            rtol=rtol,
            min_coverage=min_coverage,
        )
    except Exception as exc:
        return _result(
            passed=False,
            reason=str(exc)[:500],
            total_values=0,
            atol=atol,
            rtol=rtol,
            min_coverage=min_coverage,
        )


__all__ = [
    "ConversionResult",
    "convert_expression",
    "evaluate_semantic_check",
]
