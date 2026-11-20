"""Isolated entrypoint for the QuantMind factor sandbox."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("quantmind.sandbox")


def _wide(frame: pd.DataFrame, index: pd.MultiIndex, column: str) -> pd.DataFrame:
    if column not in frame.columns:
        raise ValueError(f"Factor input is missing required field {column}")
    series = pd.Series(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float),
        index=index,
    )
    return series.unstack("symbol").sort_index().sort_index(axis=1)


def _factor_series(request: dict) -> np.ndarray:
    """Evaluate an FM-DSL factor on the trusted, engine-staged Qlib frame."""
    from backend.services.engine.alpha_agent.factorminer.dsl import min_history, parse
    from backend.services.engine.alpha_agent.factorminer.evaluator import evaluate

    code = request.get("code")
    if not isinstance(code, str) or not code.startswith("fm:"):
        raise ValueError("This sandbox build supports FactorMiner series requests only")
    raw_path = request.get("factor_input_path")
    if raw_path != "/input/factor_input.pkl":
        raise ValueError("Factor input path is not allowlisted")
    source = pd.read_pickle(raw_path)
    if (
        not isinstance(source, pd.DataFrame)
        or not isinstance(source.index, pd.MultiIndex)
        or source.index.nlevels != 2
    ):
        raise ValueError("Factor input must be a two-level Qlib panel")
    if source.empty or source.index.has_duplicates:
        raise ValueError("Factor input panel is empty or contains duplicate rows")

    symbols = source.index.get_level_values(0).astype(str)
    dates = pd.to_datetime(source.index.get_level_values(1), errors="coerce")
    if dates.isna().any() or any(not symbol for symbol in symbols):
        raise ValueError("Factor input index contains invalid symbols or dates")
    index = pd.MultiIndex.from_arrays([dates, symbols], names=["trade_date", "symbol"])

    close = _wide(source, index, "$close")
    volume = _wide(source, index, "$volume")
    high = _wide(source, index, "$high")
    low = _wide(source, index, "$low")
    amount = _wide(source, index, "$amount")
    adjustment = _wide(source, index, "$factor")

    raw_vwap = (amount * 1e4 / volume.where(volume > 0)).replace(
        [np.inf, -np.inf], np.nan
    )
    adjusted_vwap = (raw_vwap * adjustment).replace([np.inf, -np.inf], np.nan)

    def coverage(vwap: pd.DataFrame) -> float:
        valid = vwap.notna() & low.notna() & high.notna()
        inside = valid & (vwap >= low - 1e-6) & (vwap <= high + 1e-6)
        return float(inside.to_numpy().sum() / max(1, valid.to_numpy().sum()))

    adjusted_coverage = coverage(adjusted_vwap)
    raw_coverage = coverage(raw_vwap)
    vwap = adjusted_vwap if adjusted_coverage >= raw_coverage else raw_vwap
    if max(adjusted_coverage, raw_coverage) < 0.99:
        raise ValueError(
            "Factor input VWAP coverage is below the validated 0.99 threshold"
        )

    close_long = pd.Series(close.stack(), name="$close")
    returns_long = close_long / close_long.groupby(level="symbol").shift(1) - 1.0
    features = {
        "$open": _wide(source, index, "$open"),
        "$high": high,
        "$low": low,
        "$close": close,
        "$volume": volume,
        "$amt": amount,
        "$vwap": vwap,
        "$returns": returns_long.unstack("symbol").sort_index().sort_index(axis=1),
    }
    node = parse(code[3:])
    values_wide = evaluate(node, features)
    values_long = values_wide.stack()
    values_long.index = values_long.index.set_names(["trade_date", "symbol"])
    values = values_long.reindex(index).to_numpy(dtype=np.float64)

    # Match the production FM runtime: each instrument's first min_history bars
    # are warm-up only, even when the expression's rolling operators allow partial windows.
    ordered = pd.DataFrame(
        {"row": np.arange(len(source)), "symbol": symbols, "date": dates}
    ).sort_values(["symbol", "date", "row"], kind="stable")
    ordered["position"] = ordered.groupby("symbol", sort=False).cumcount()
    warmup = np.empty(len(source), dtype=bool)
    warmup[ordered["row"].to_numpy()] = ordered["position"].to_numpy() < min_history(
        node
    )
    values[warmup | ~np.isfinite(values)] = np.nan
    return values


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    if len(sys.argv) != 2 or sys.argv[1] != "factor":
        logger.error("Unsupported sandbox operation")
        return 2
    try:
        request = json.loads(Path("/input/request.json").read_text(encoding="utf-8"))
        if (
            not isinstance(request, dict)
            or request.get("factor_result_mode") != "series"
        ):
            raise ValueError("Only FactorMiner series mode is supported")
        values = _factor_series(request).astype("<f8", copy=False)
        output = Path("/output/factor_series.bin")
        output.write_bytes(values.tobytes())
        logger.info("FactorMiner series completed: %d rows", len(values))
        return 0
    except Exception:
        logger.exception("Factor sandbox execution failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
