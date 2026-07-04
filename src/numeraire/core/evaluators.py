"""Native evaluators (numpy/scipy, no heavy deps) — the performance family.

Evaluators dispatch by capability and emit rows of the tidy result schema, so the metric
always matches the object (VoC's headline is *timing Sharpe*, not R²). Each
carries ``requires`` (the capabilities an OOS output must expose) and registers itself in the
open evaluator registry so external packages add peers without editing core.
"""

from __future__ import annotations

from typing import ClassVar, Protocol

import numpy as np
import pandas as pd

from numeraire.core import capabilities
from numeraire.core.engine import ForecastOutput, PanelWeightsOutput, WeightsOutput
from numeraire.core.registry import register_evaluator
from numeraire.core.schema import RESULT_COLUMNS


class _HasProvenance(Protocol):
    @property
    def run_id(self) -> str: ...
    @property
    def method(self) -> str: ...
    @property
    def capability(self) -> str: ...
    @property
    def config_hash(self) -> str: ...
    @property
    def data_vintage(self) -> str: ...
    @property
    def universe(self) -> str: ...


def _row(out: _HasProvenance, metric: str, value: float, date: object) -> dict[str, object]:
    """Build one result-schema row from an OOS output's provenance plus a (metric, value)."""
    return {
        "run_id": out.run_id,
        "method": out.method,
        "date": date,
        "metric": metric,
        "value": value,
        "universe": out.universe,
        "capability": out.capability,
        "config_hash": out.config_hash,
        "data_vintage": out.data_vintage,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble result rows into a DataFrame with the canonical column order."""
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


class SharpeEvaluator:
    """Annualized Sharpe ratio of the realized strategy returns (the timing headline)."""

    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def __init__(self, periods_per_year: int = 12) -> None:
        self.periods_per_year = periods_per_year

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, WeightsOutput | PanelWeightsOutput):
            raise TypeError("SharpeEvaluator requires a WeightsOutput or PanelWeightsOutput")
        s = oos_output.strategy_returns()
        r = s.to_numpy(dtype=np.float64)
        r = r[~np.isnan(r)]
        ann = float(np.sqrt(self.periods_per_year))
        if r.size < 2 or float(np.std(r, ddof=1)) == 0.0:
            sharpe = float("nan")
        else:
            sharpe = float(np.mean(r) / np.std(r, ddof=1)) * ann
        return _frame([_row(oos_output, "sharpe", sharpe, s.index[-1])])


class MeanReturnEvaluator:
    """Annualized mean of the realized strategy returns."""

    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def __init__(self, periods_per_year: int = 12) -> None:
        self.periods_per_year = periods_per_year

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, WeightsOutput | PanelWeightsOutput):
            raise TypeError("MeanReturnEvaluator requires a WeightsOutput or PanelWeightsOutput")
        s = oos_output.strategy_returns()
        r = s.to_numpy(dtype=np.float64)
        r = r[~np.isnan(r)]
        mean = float(np.mean(r)) * self.periods_per_year if r.size else float("nan")
        return _frame([_row(oos_output, "mean_return", mean, s.index[-1])])


class OOSR2Evaluator:
    """Goyal-Welch (2008) out-of-sample R^2 of a forecast vs the historical-mean benchmark.

    ``R^2_oos = 1 - SSE_model / SSE_benchmark``, pooled across all origins and assets, reported
    in percent. Positive => the model beats the prevailing mean OOS. This is the *right* metric
    for predictive-regression methods — e.g. 1/A's published dp Table 3.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_FORECAST}

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, ForecastOutput):
            raise TypeError("OOSR2Evaluator requires a ForecastOutput")
        r = oos_output.realized.to_numpy(dtype=np.float64)
        f = oos_output.forecasts.to_numpy(dtype=np.float64)
        b = oos_output.benchmark.to_numpy(dtype=np.float64)
        sse_model = float(np.nansum((r - f) ** 2))
        sse_bench = float(np.nansum((r - b) ** 2))
        r2 = float("nan") if sse_bench == 0.0 else (1.0 - sse_model / sse_bench) * 100.0
        date = oos_output.forecasts.index[-1]
        return _frame([_row(oos_output, "oos_r2_pct", r2, date)])


class StrategyReturnEvaluator:
    """Per-period (time-indexed) realized strategy return — one result row **per date**.

    Where the summary evaluators collapse a whole sample to one scalar, this emits the time
    series (``metric="strategy_return"``, ``date=t``), so downstream can plot cumulative
    performance / drawdowns. The result schema's ``date`` column carries the time dimension.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, WeightsOutput | PanelWeightsOutput):
            raise TypeError(
                "StrategyReturnEvaluator requires a WeightsOutput or PanelWeightsOutput"
            )
        s = oos_output.strategy_returns()
        rows = [_row(oos_output, "strategy_return", float(v), t) for t, v in s.items()]
        return _frame(rows)


class SquaredErrorDiffEvaluator:
    """Per-origin squared-error difference (benchmark minus model), one row **per date**.

    ``value_t = sum_assets[(r-b)^2 - (r-f)^2]`` at origin ``t``; its cumulative sum is the
    CDSPE curve (positive & rising ⇒ the model beats the prevailing mean over time). The
    time-series companion to the scalar :class:`OOSR2Evaluator`.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_FORECAST}

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, ForecastOutput):
            raise TypeError("SquaredErrorDiffEvaluator requires a ForecastOutput")
        r = oos_output.realized.to_numpy(dtype=np.float64)
        f = oos_output.forecasts.to_numpy(dtype=np.float64)
        b = oos_output.benchmark.to_numpy(dtype=np.float64)
        sed = np.nansum((r - b) ** 2 - (r - f) ** 2, axis=1)
        idx = oos_output.forecasts.index
        rows = [_row(oos_output, "sed", float(v), t) for t, v in zip(idx, sed, strict=True)]
        return _frame(rows)


# Bundled native evaluators register on import (open registry).
register_evaluator("sharpe", SharpeEvaluator(), overwrite=True)
register_evaluator("mean_return", MeanReturnEvaluator(), overwrite=True)
register_evaluator("strategy_return", StrategyReturnEvaluator(), overwrite=True)
register_evaluator("oos_r2", OOSR2Evaluator(), overwrite=True)
register_evaluator("sed", SquaredErrorDiffEvaluator(), overwrite=True)
