"""Native evaluators (numpy/scipy, no heavy deps) — the performance family.

Evaluators dispatch by capability and emit rows of the tidy result schema, so the metric
always matches the object (VoC's headline is *timing Sharpe*, not R²). Each
carries ``requires`` (the capabilities an OOS output must expose) and registers itself in the
open evaluator registry so external packages add peers without editing core.
"""

from __future__ import annotations

import warnings
from typing import ClassVar, Protocol

import numpy as np
import pandas as pd
from scipy.stats import norm

from numeraire.core import capabilities
from numeraire.core.engine import (
    ForecastOutput,
    PanelWeightsOutput,
    PricingOutput,
    WeightsOutput,
)
from numeraire.core.registry import register_evaluator
from numeraire.core.schema import RESULT_COLUMNS
from numeraire.core.stats import alpha_regression, certainty_equivalent, newey_west_lrv


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
    """Build one result-schema row from an OOS output's provenance plus a (metric, value).

    ``protocol`` is read from the output when present (a :class:`PricingOutput` carries its
    ``"walk_forward"`` / ``"in_sample"`` label) and defaults to ``"walk_forward"`` otherwise — every
    weights/forecast output is produced by a walk-forward driver, so that is its intrinsic protocol.
    """
    return {
        "run_id": out.run_id,
        "method": out.method,
        "date": date,
        "metric": metric,
        "value": value,
        "universe": out.universe,
        "capability": out.capability,
        "protocol": getattr(out, "protocol", "walk_forward"),
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
    """Out-of-sample R^2 of a forecast vs a benchmark, ``1 - SSE_model / SSE_benchmark`` (percent).

    Pooled across all origins and assets; positive => the model beats the benchmark OOS.

    ``benchmark`` selects the yardstick:

    - ``"historical"`` (default) — the prevailing-mean benchmark carried in the output
      (Goyal-Welch 2008): the *right* metric for predictive-regression methods (e.g. 1/A's dp).
    - ``"zero"`` — a zero forecast, ``SSE_benchmark = sum r^2``. This is the Gu-Kelly-Xiu (2020)
      convention for the machine-learning cross-section (return predictability is measured against
      "no signal", not against a fitted mean), and it materially changes the number.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_FORECAST}
    _BENCHMARKS: ClassVar[tuple[str, ...]] = ("historical", "zero")

    def __init__(self, benchmark: str = "historical") -> None:
        if benchmark not in self._BENCHMARKS:
            raise ValueError(f"benchmark must be one of {self._BENCHMARKS}; got {benchmark!r}")
        self.benchmark = benchmark

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, ForecastOutput):
            raise TypeError("OOSR2Evaluator requires a ForecastOutput")
        r = oos_output.realized.to_numpy(dtype=np.float64)
        f = oos_output.forecasts.to_numpy(dtype=np.float64)
        if self.benchmark == "zero":
            b = np.zeros_like(r)
        else:
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


class ClarkWestEvaluator:
    """Clark-West (2007) MSPE-adjusted test of the forecast against its nested benchmark.

    The right significance test to pair with :class:`OOSR2Evaluator` — plain Diebold-Mariano is
    undersized against a nested benchmark (the historical mean). Multi-asset outputs aggregate the
    per-origin adjusted loss difference across assets (the pooled companion of
    :class:`SquaredErrorDiffEvaluator`); one asset is the textbook statistic. Emits two rows:
    ``cw_t`` and ``cw_p`` (one-sided). Use ``nw_lags = horizon - 1`` for multi-step forecasts.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_FORECAST}

    def __init__(self, nw_lags: int = 0) -> None:
        self.nw_lags = nw_lags

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, ForecastOutput):
            raise TypeError("ClarkWestEvaluator requires a ForecastOutput")
        r = oos_output.realized.to_numpy(dtype=np.float64)
        f = oos_output.forecasts.to_numpy(dtype=np.float64)
        b = oos_output.benchmark.to_numpy(dtype=np.float64)
        adj = np.nansum((r - b) ** 2 - ((r - f) ** 2 - (b - f) ** 2), axis=1)
        n = len(adj)
        se = float(np.sqrt(newey_west_lrv(adj, self.nw_lags) / n)) if n else float("nan")
        t_stat = float(adj.mean() / se) if n and se > 0 else float("nan")
        p = float(norm.sf(t_stat)) if np.isfinite(t_stat) else float("nan")
        date = oos_output.forecasts.index[-1]
        return _frame([_row(oos_output, "cw_t", t_stat, date), _row(oos_output, "cw_p", p, date)])


class AlphaEvaluator:
    """Time-series alpha of the strategy vs a factor benchmark (HAC t-stat).

    ``factors`` are per-period factor (excess) returns on the strategy's calendar; rows are
    inner-joined. Emits ``alpha_ann`` (per-period alpha x ``periods_per_year``) and ``alpha_t``.
    The volatility-managed-portfolio-style headline regression; ``nw_lags=0`` = White errors.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def __init__(
        self, factors: pd.DataFrame, *, nw_lags: int = 0, periods_per_year: int = 12
    ) -> None:
        self.factors = factors
        self.nw_lags = nw_lags
        self.periods_per_year = periods_per_year

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, WeightsOutput | PanelWeightsOutput):
            raise TypeError("AlphaEvaluator requires a WeightsOutput or PanelWeightsOutput")
        s = oos_output.strategy_returns()
        res = alpha_regression(s, self.factors, nw_lags=self.nw_lags)
        date = s.index[-1]
        return _frame(
            [
                _row(oos_output, "alpha_ann", res.alpha * self.periods_per_year, date),
                _row(oos_output, "alpha_t", res.alpha_t, date),
            ]
        )


class CEQEvaluator:
    """DGU (2009) certainty-equivalent return of the realized strategy returns (economic value).

    ``ceq = mean - gamma/2 * var`` of the per-period strategy returns (``gamma`` = risk aversion,
    DGU report ``gamma=1``). Emitted per-period, in the input's units — DGU's Table 4 CEQ figures
    are monthly — so it is *not* annualized (unlike :class:`SharpeEvaluator`). The economic-value
    companion to the risk-adjusted :class:`SharpeEvaluator` in a 1/N-style horse race.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, WeightsOutput | PanelWeightsOutput):
            raise TypeError("CEQEvaluator requires a WeightsOutput or PanelWeightsOutput")
        s = oos_output.strategy_returns()
        ceq = certainty_equivalent(s.to_numpy(dtype=np.float64), self.gamma)
        return _frame([_row(oos_output, "ceq", ceq, s.index[-1])])


def _pricing_means(out: PricingOutput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-asset time-mean predicted and realized returns + a finite mask (assets with both)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN asset columns -> NaN mean
        mp = np.nanmean(out.predicted.to_numpy(dtype=np.float64), axis=0)
        mr = np.nanmean(out.realized.to_numpy(dtype=np.float64), axis=0)
    finite = np.isfinite(mp) & np.isfinite(mr)
    return mp, mr, finite


def _pricing_date(out: PricingOutput) -> object:
    """The output's last prediction date (``NaT`` for an empty panel), for the result row."""
    idx = out.predicted.index
    return idx[-1] if len(idx) else pd.NaT


class CrossSectionalR2Evaluator:
    """Cross-sectional R^2 of mean realized returns on mean predicted expected returns (OLS).

    The pricing headline (the classic average-realized-vs-average-predicted plot): time-average each
    asset's realized and predicted returns, then OLS-regress mean realized on mean predicted across
    assets and report the R^2. Assets missing either mean are dropped. Read against the output's
    ``protocol`` — an ``"in_sample"`` R^2 is explanatory, a ``"walk_forward"`` R^2 is out-of-sample.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_PRICING}

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, PricingOutput):
            raise TypeError("CrossSectionalR2Evaluator requires a PricingOutput")
        mp, mr, finite = _pricing_means(oos_output)
        mp, mr = mp[finite], mr[finite]
        if mp.size < 2:
            r2 = float("nan")
        else:
            x = np.column_stack([np.ones(mp.size), mp])
            coef, *_ = np.linalg.lstsq(x, mr, rcond=None)
            resid = mr - x @ coef
            ss_res = float(resid @ resid)
            ss_tot = float(((mr - mr.mean()) ** 2).sum())
            r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        return _frame([_row(oos_output, "xs_r2", r2, _pricing_date(oos_output))])


class AverageAbsAlphaEvaluator:
    """Average absolute pricing error (mean over assets of ``|mean realized - mean predicted|``).

    Each asset's alpha is its mean realized return minus its mean predicted expected return; the
    metric is the cross-sectional mean of the absolute alphas (in the input's return units). The
    magnitude companion to :class:`CrossSectionalR2Evaluator`. (Factor-model joint zero-alpha
    inference stays in :func:`numeraire.core.stats.grs_test`, which needs the factor returns this
    generic pricing surface deliberately does not assume.)
    """

    requires: ClassVar[set[str]] = {capabilities.TO_PRICING}

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, PricingOutput):
            raise TypeError("AverageAbsAlphaEvaluator requires a PricingOutput")
        mp, mr, finite = _pricing_means(oos_output)
        alpha = (mr - mp)[finite]
        value = float(np.mean(np.abs(alpha))) if alpha.size else float("nan")
        return _frame([_row(oos_output, "avg_abs_alpha", value, _pricing_date(oos_output))])


# Bundled native evaluators register on import (open registry).
register_evaluator("sharpe", SharpeEvaluator(), overwrite=True)
register_evaluator("ceq", CEQEvaluator(), overwrite=True)
register_evaluator("mean_return", MeanReturnEvaluator(), overwrite=True)
register_evaluator("strategy_return", StrategyReturnEvaluator(), overwrite=True)
register_evaluator("oos_r2", OOSR2Evaluator(), overwrite=True)
register_evaluator("sed", SquaredErrorDiffEvaluator(), overwrite=True)
register_evaluator("clark_west", ClarkWestEvaluator(), overwrite=True)
register_evaluator("xs_r2", CrossSectionalR2Evaluator(), overwrite=True)
register_evaluator("avg_abs_alpha", AverageAbsAlphaEvaluator(), overwrite=True)
