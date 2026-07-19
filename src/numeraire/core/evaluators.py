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
from scipy.stats import norm, spearmanr

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


def _row(
    out: _HasProvenance,
    metric: str,
    value: float,
    date: object,
    *,
    n_obs: int | None = None,
    n_dropped: int | None = None,
) -> dict[str, object]:
    """Build one result-schema row from an OOS output's provenance plus a (metric, value).

    ``protocol`` is read from the output when present (a :class:`PricingOutput` carries its
    ``"walk_forward"`` / ``"in_sample"`` label) and defaults to ``"walk_forward"`` otherwise — every
    weights/forecast output is produced by a walk-forward driver, so that is its intrinsic protocol.

    ``n_obs`` / ``n_dropped`` are the optional attrition counts (:data:`ATTRITION_COLUMNS`): the
    size of the joint finite sample the metric was computed on and the count of candidate
    observations the joint mask excluded. Evaluators that compare a model against a benchmark or
    realized target attach them so a selectively-missing input can never quietly change the
    denominator; the rest omit them.
    """
    row: dict[str, object] = {
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
    if n_obs is not None:
        row["n_obs"] = n_obs
    if n_dropped is not None:
        row["n_dropped"] = n_dropped
    return row


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble result rows into a DataFrame with the canonical columns first, extras appended.

    Any keys beyond :data:`RESULT_COLUMNS` (e.g. the optional attrition columns) are kept, in first
    appearance order, after the canonical columns — the result schema is additive, so a downstream
    consumer sees the standard columns unchanged and the extras alongside.
    """
    extra: list[str] = []
    for r in rows:
        for key in r:
            if key not in RESULT_COLUMNS and key not in extra:
                extra.append(key)
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS) + extra)


# Fail-closed missingness threshold: a comparison whose joint finite mask drops more than this
# fraction of its candidate observations is refused (raises), rather than scored on a rump sample or
# flagged with a warning that a batch run would swallow. Below the threshold the attrition counts on
# each row keep the drop auditable.
_MAX_DROP_FRACTION: float = 0.5


def _joint_finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Element-wise ``finite(a0) & finite(a1) & ...`` over identically-shaped arrays.

    One mask for the whole comparison, so the model, its target, and the benchmark are always scored
    on the *same* observations — never on separate ``nansum`` / ``nanmean`` denominators (which lets
    a selectively-missing model manufacture apparent skill against a fully-observed benchmark).
    """
    mask = np.isfinite(arrays[0])
    for a in arrays[1:]:
        mask = mask & np.isfinite(a)
    return mask


def _attrition(
    mask: np.ndarray, evaluator: str, *, candidates: np.ndarray | None = None
) -> tuple[int, int]:
    """Return ``(n_obs, n_dropped)`` for ``mask``; raise if it drops a majority of candidates.

    ``n_obs`` is the joint finite sample size, ``n_dropped`` the number of candidate observations
    the joint mask excluded. ``candidates`` (optional, same shape) narrows what counts as a
    candidate: cells outside it are *structural* — absent on every side of the comparison (e.g. an
    asset not yet in a ragged pricing universe) — and count neither as observed nor as dropped.
    The default is every cell (dense engine outputs). An empty candidate set raises ``ValueError``
    (there is nothing to score — e.g. an empty output from a view too short to produce any
    evaluation window), and dropping more than :data:`_MAX_DROP_FRACTION` of the candidates raises
    as well (fail closed — a majority-missing comparison is not scored).
    """
    total = int(mask.size) if candidates is None else int(np.count_nonzero(candidates))
    n_obs = int(np.count_nonzero(mask))
    n_dropped = total - n_obs
    if total == 0:
        raise ValueError(
            f"{evaluator}: no candidate observations to score (empty comparison output)"
        )
    if n_dropped > _MAX_DROP_FRACTION * total:
        raise ValueError(
            f"{evaluator}: joint finite sample drops {n_dropped}/{total} candidate observations "
            f"(> {_MAX_DROP_FRACTION:.0%}); refusing to score a majority-missing comparison"
        )
    return n_obs, n_dropped


def _dated_weights(
    out: WeightsOutput | PanelWeightsOutput,
) -> list[tuple[object, dict[str, float]]]:
    """Per-date ``{asset: target weight}`` maps in calendar order.

    Normalizes both the wide :class:`WeightsOutput` (a ``date x asset`` frame) and the long
    :class:`PanelWeightsOutput` (a ``(date, asset)`` Series over a ragged universe) to the same
    per-date mapping, so exposure diagnostics align turnover across an entering/exiting universe.
    """
    dated: list[tuple[object, dict[str, float]]] = []
    if isinstance(out, WeightsOutput):
        assets = [str(c) for c in out.weights.columns]
        mat = out.weights.to_numpy(dtype=np.float64)
        if not bool(np.isfinite(mat).all()):
            raise ValueError("target weights must all be finite")
        for i, t in enumerate(out.weights.index):
            dated.append((t, {a: float(v) for a, v in zip(assets, mat[i], strict=True)}))
        return dated
    for t, sub in out.weights.groupby(level="date"):
        names = [str(a) for a in sub.index.get_level_values("asset")]
        vals = sub.to_numpy(dtype=np.float64)
        if not bool(np.isfinite(vals).all()):
            raise ValueError("panel target weights must all be finite")
        dated.append((t, {a: float(v) for a, v in zip(names, vals, strict=True)}))
    return dated


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


class OutOfSampleR2Evaluator:
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
            raise TypeError("OutOfSampleR2Evaluator requires a ForecastOutput")
        r = oos_output.realized.to_numpy(dtype=np.float64)
        f = oos_output.forecasts.to_numpy(dtype=np.float64)
        if self.benchmark == "zero":
            b = np.zeros_like(r)
        else:
            b = oos_output.benchmark.to_numpy(dtype=np.float64)
        # One joint finite sample for model, target, and benchmark: both SSEs share the same
        # observations, so a forecast that is missing where the benchmark is present can no longer
        # shrink only its own denominator and manufacture skill.
        mask = _joint_finite_mask(r, f, b)
        n_obs, n_dropped = _attrition(mask, "OutOfSampleR2Evaluator")
        rm, fm, bm = r[mask], f[mask], b[mask]
        sse_model = float(np.sum((rm - fm) ** 2))
        sse_bench = float(np.sum((rm - bm) ** 2))
        r2 = float("nan") if sse_bench == 0.0 else (1.0 - sse_model / sse_bench) * 100.0
        date = oos_output.forecasts.index[-1]
        return _frame([_row(oos_output, "oos_r2_pct", r2, date, n_obs=n_obs, n_dropped=n_dropped)])


class OOSR2Evaluator(OutOfSampleR2Evaluator):
    """Deprecated alias for :class:`OutOfSampleR2Evaluator` (kept for one release).

    Constructing it warns; behaviour is identical (subclass). The registry key ``"oos_r2"`` and the
    ``oos_r2_pct`` metric string are unchanged, so registered lookups are unaffected.
    """

    def __init__(self, benchmark: str = "historical") -> None:
        warnings.warn(
            "OOSR2Evaluator is deprecated and will be removed in a future release; "
            "use OutOfSampleR2Evaluator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(benchmark)


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


class ExposureEvaluator:
    """Per-date portfolio-construction diagnostics flattened into per-date result rows.

    Emits **one scalar row per date per metric** (like :class:`StrategyReturnEvaluator`), never the
    per-date x asset weight matrix — the tidy schema has no asset axis, and the weights heatmap
    consumes the ``WeightsOutput`` object directly downstream, not this result table. For the weight
    vector ``w_t`` on date ``t``:

    - ``gross_leverage`` = ``sum_a |w_{t,a}|`` (leverage; 1.0 for a fully-invested long-only book);
    - ``net_exposure``   = ``sum_a w_{t,a}`` (directional tilt; 0 for a dollar-neutral book);
    - ``turnover``       = ``sum_a |w_{t,a} - w_{t-1,a}|`` (one-sided L1 rebalancing volume vs the
      previous rebalance, asset-aligned over the union universe; the opening rebalance is measured
      from an all-cash book, so the first date's turnover equals its gross leverage);
    - ``hhi``            = ``sum_a w_{t,a}^2`` (Herfindahl-Hirschman concentration; ``1/N`` for an
      equal-weight book of ``N`` names, 1.0 for a single-name bet).

    Handles the wide :class:`WeightsOutput` and the long :class:`PanelWeightsOutput` (turnover is
    aligned across an entering/exiting universe). Outputs reject non-finite target weights.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, WeightsOutput | PanelWeightsOutput):
            raise TypeError("ExposureEvaluator requires a WeightsOutput or PanelWeightsOutput")
        dated = _dated_weights(oos_output)
        rows: list[dict[str, object]] = []
        prev: dict[str, float] | None = None
        for t, cur in dated:
            vals = np.array(list(cur.values()), dtype=np.float64)
            gross = float(np.sum(np.abs(vals)))
            net = float(np.sum(vals))
            hhi = float(np.sum(vals**2))
            if prev is None:
                turnover = gross  # opening trade from an all-cash book
            else:
                keys = set(cur) | set(prev)
                turnover = float(sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys))
            for metric, val in (
                ("gross_leverage", gross),
                ("net_exposure", net),
                ("turnover", turnover),
                ("hhi", hhi),
            ):
                rows.append(_row(oos_output, metric, val, t))
            prev = cur
        return _frame(rows)


class SquaredErrorDiffEvaluator:
    """Per-origin squared-error difference (benchmark minus model), one row **per date**.

    ``value_t = sum_assets[(r-b)^2 - (r-f)^2]`` at origin ``t``; its cumulative sum is the
    CDSPE curve (positive & rising ⇒ the model beats the prevailing mean over time). The
    time-series companion to the scalar :class:`OutOfSampleR2Evaluator`.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_FORECAST}

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, ForecastOutput):
            raise TypeError("SquaredErrorDiffEvaluator requires a ForecastOutput")
        r = oos_output.realized.to_numpy(dtype=np.float64)
        f = oos_output.forecasts.to_numpy(dtype=np.float64)
        b = oos_output.benchmark.to_numpy(dtype=np.float64)
        # One joint finite mask over the (origin x asset) cells: each cell enters both squared-error
        # terms or neither, and the overall drop is bounded. A row with no joint-finite cell scores
        # ``nan`` (n_obs=0) rather than a spurious 0 that would flatten the CDSPE curve.
        mask = _joint_finite_mask(r, f, b)
        _attrition(mask, "SquaredErrorDiffEvaluator")
        diff = (r - b) ** 2 - (r - f) ** 2
        idx = oos_output.forecasts.index
        rows: list[dict[str, object]] = []
        for i, t in enumerate(idx):
            row_mask = mask[i]
            k = int(np.count_nonzero(row_mask))
            value = float(diff[i][row_mask].sum()) if k else float("nan")
            rows.append(
                _row(oos_output, "sed", value, t, n_obs=k, n_dropped=int(row_mask.size - k))
            )
        return _frame(rows)


class ClarkWestEvaluator:
    """Clark-West (2007) MSPE-adjusted test of the forecast against its nested benchmark.

    The right significance test to pair with :class:`OutOfSampleR2Evaluator` — plain
    Diebold-Mariano is undersized against a nested benchmark (the historical mean). Multi-asset
    outputs aggregate the per-origin adjusted loss difference across assets (the pooled companion of
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
        # One joint finite mask over the (origin x asset) cells, bounded attrition. The per-origin
        # adjusted loss difference is summed over its joint-finite cells only, and an origin with no
        # joint-finite cell is excluded from the mean and the variance (not carried as a phantom 0
        # that would bias the mean and inflate the effective sample). The HAC variance is computed
        # on the ORIGINAL origin axis — lag-l autocovariances pair only observed origins exactly l
        # positions apart — so an internal gap does not make origins two periods apart look
        # adjacent (which compacting the observed origins together would).
        mask = _joint_finite_mask(r, f, b)
        n_obs, n_dropped = _attrition(mask, "ClarkWestEvaluator")
        per_cell = (r - b) ** 2 - ((r - f) ** 2 - (b - f) ** 2)
        observed = mask.any(axis=1)
        adj = np.where(mask, per_cell, 0.0).sum(axis=1)
        n = int(np.count_nonzero(observed))
        se = (
            float(np.sqrt(newey_west_lrv(adj, self.nw_lags, valid=observed) / n))
            if n
            else float("nan")
        )
        t_stat = float(adj[observed].mean() / se) if n and se > 0 else float("nan")
        p = float(norm.sf(t_stat)) if np.isfinite(t_stat) else float("nan")
        date = oos_output.forecasts.index[-1]
        return _frame(
            [
                _row(oos_output, "cw_t", t_stat, date, n_obs=n_obs, n_dropped=n_dropped),
                _row(oos_output, "cw_p", p, date, n_obs=n_obs, n_dropped=n_dropped),
            ]
        )


class ICEvaluator:
    """Information coefficient of a forecast: the per-period cross-sectional rank correlation.

    For each origin ``t`` the (Spearman) rank correlation ``ic_t`` between the forecast
    cross-section and the realized-return cross-section across assets (finite pairs with variation
    only). Emits three summary rows dated at the last origin:

    - ``ic``    = ``mean_t ic_t`` (the average information coefficient);
    - ``ic_ir`` = ``mean_t ic_t / std_t ic_t`` (the IC information ratio — the signal consistency);
    - ``ic_t``  = ``ic_ir * sqrt(n_periods)`` (the t-statistic of a non-zero mean IC).

    This is the *rank* IC at the output's single horizon; an IC-decay-vs-horizon curve is assembled
    by the caller running forecasts at several horizons (one :class:`ForecastOutput` has one
    horizon) and stacking the ``ic`` rows. A single-asset forecast has no cross-section to rank, so
    every ``ic_t`` is undefined and the metrics are ``nan``.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_FORECAST}

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, ForecastOutput):
            raise TypeError("ICEvaluator requires a ForecastOutput")
        f = oos_output.forecasts.to_numpy(dtype=np.float64)
        r = oos_output.realized.to_numpy(dtype=np.float64)
        ics: list[float] = []
        for i in range(f.shape[0]):
            m = np.isfinite(f[i]) & np.isfinite(r[i])
            if int(m.sum()) < 2:
                continue
            fr, rr = f[i][m], r[i][m]
            if np.ptp(fr) == 0.0 or np.ptp(rr) == 0.0:
                continue  # a constant cross-section has no rank ordering
            rho, _ = spearmanr(fr, rr)
            if np.isfinite(rho):
                ics.append(float(rho))
        arr = np.asarray(ics, dtype=np.float64)
        ic_mean = float(arr.mean()) if arr.size else float("nan")
        if arr.size >= 2 and float(arr.std(ddof=1)) > 0.0:
            ic_ir = float(arr.mean() / arr.std(ddof=1))
            ic_t = ic_ir * float(np.sqrt(arr.size))
        else:
            ic_ir = ic_t = float("nan")
        date = oos_output.forecasts.index[-1] if len(oos_output.forecasts.index) else pd.NaT
        return _frame(
            [
                _row(oos_output, "ic", ic_mean, date),
                _row(oos_output, "ic_ir", ic_ir, date),
                _row(oos_output, "ic_t", ic_t, date),
            ]
        )


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


class TreynorEvaluator:
    """Treynor (1965) ratio: annualized mean excess return per unit of *systematic* (market) risk.

    ``treynor = periods_per_year * mean(r_p) / beta_market`` — where :class:`SharpeEvaluator`
    divides the reward by *total* volatility, Treynor divides by the CAPM market beta, rewarding a
    book whose idiosyncratic risk is already diversified away. ``beta_market`` is the strategy's
    loading on the market factor from a (HAC) time-series regression of its realized returns on
    ``factors[[market]]`` (reusing :func:`~numeraire.core.stats.alpha_regression`); the mean is
    taken over the same overlapping sample. ``factors`` mirrors :class:`AlphaEvaluator`; ``market``
    names the systematic column (default: the first). Numerator and beta are in return units.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def __init__(
        self,
        factors: pd.DataFrame,
        *,
        market: str | None = None,
        nw_lags: int = 0,
        periods_per_year: int = 12,
    ) -> None:
        self.factors = factors
        self.market = market
        self.nw_lags = nw_lags
        self.periods_per_year = periods_per_year

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, WeightsOutput | PanelWeightsOutput):
            raise TypeError("TreynorEvaluator requires a WeightsOutput or PanelWeightsOutput")
        s = oos_output.strategy_returns()
        col = self.market if self.market is not None else str(self.factors.columns[0])
        mkt = self.factors[[col]]
        beta = float(alpha_regression(s, mkt, nw_lags=self.nw_lags).betas[0])
        joined = pd.concat([s.rename("_p"), mkt], axis=1, join="inner").dropna()
        mean_ex = float(joined["_p"].mean()) if len(joined) else float("nan")
        treynor = float("nan") if beta == 0.0 else mean_ex * self.periods_per_year / beta
        return _frame([_row(oos_output, "treynor", treynor, s.index[-1])])


class InformationRatioEvaluator:
    """Information ratio: annualized mean active return per unit of tracking error vs a benchmark.

    ``active_t = r_p,t - r_b,t`` (inner-joined on dates); ``ir = sqrt(P) * mean(active) /
    std(active, ddof=1)``, the tracking-error-scaled active-management skill measure (Grinold-Kahn).
    ``benchmark`` is a per-period return series on the strategy's calendar (e.g. a benchmark
    method's realized :meth:`~numeraire.core.engine.WeightsOutput.strategy_returns`). Same
    annualization as :class:`SharpeEvaluator` (a ratio of a mean to a std, so ``sqrt(P)``).
    """

    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def __init__(self, benchmark: pd.Series, *, periods_per_year: int = 12) -> None:
        self.benchmark = benchmark
        self.periods_per_year = periods_per_year

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, WeightsOutput | PanelWeightsOutput):
            raise TypeError(
                "InformationRatioEvaluator requires a WeightsOutput or PanelWeightsOutput"
            )
        s = oos_output.strategy_returns()
        joined = pd.concat(
            [s.rename("_p"), self.benchmark.rename("_b")], axis=1, join="inner"
        ).dropna()
        active = (joined["_p"] - joined["_b"]).to_numpy(dtype=np.float64)
        ann = float(np.sqrt(self.periods_per_year))
        if active.size < 2 or float(np.std(active, ddof=1)) == 0.0:
            ir = float("nan")
        else:
            ir = float(np.mean(active) / np.std(active, ddof=1)) * ann
        return _frame([_row(oos_output, "information_ratio", ir, s.index[-1])])


class M2Evaluator:
    """Modigliani-Modigliani (1997) M-squared: the strategy's Sharpe expressed at benchmark risk.

    The strategy levered/de-levered to the benchmark's volatility, reported in return units:
    ``m2 = periods_per_year * (mean(r_p) / std(r_p)) * std(r_b)`` on the overlapping sample. Because
    it equals ``annualized_Sharpe(r_p) * annualized_vol(r_b)``, it ranks portfolios identically to
    the Sharpe ratio but on the intuitive scale of "what return would this earn at the benchmark's
    risk". Computed in excess-return space (the risk-free add-back cancels), so ``m2 - mean(r_b)``
    is the strategy's risk-adjusted outperformance of the benchmark. ``benchmark`` is a per-period
    return series on the strategy's calendar.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def __init__(self, benchmark: pd.Series, *, periods_per_year: int = 12) -> None:
        self.benchmark = benchmark
        self.periods_per_year = periods_per_year

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, WeightsOutput | PanelWeightsOutput):
            raise TypeError("M2Evaluator requires a WeightsOutput or PanelWeightsOutput")
        s = oos_output.strategy_returns()
        joined = pd.concat(
            [s.rename("_p"), self.benchmark.rename("_b")], axis=1, join="inner"
        ).dropna()
        p = joined["_p"].to_numpy(dtype=np.float64)
        b = joined["_b"].to_numpy(dtype=np.float64)
        sd_p = float(np.std(p, ddof=1)) if p.size >= 2 else 0.0
        if p.size < 2 or sd_p == 0.0:
            m2 = float("nan")
        else:
            m2 = float(np.mean(p) / sd_p * np.std(b, ddof=1)) * self.periods_per_year
        return _frame([_row(oos_output, "m2", m2, s.index[-1])])


class SortinoEvaluator:
    """Sortino ratio: annualized excess return over a MAR per unit of *downside* deviation.

    ``sortino = sqrt(P) * (mean(r) - mar) / DD`` with the target downside deviation
    ``DD = sqrt(mean(min(r - mar, 0)^2))`` (the full-sample denominator, so periods above the MAR
    enter as zeros). Where :class:`SharpeEvaluator` penalizes *all* volatility, Sortino penalizes
    only harmful shortfalls below the minimum acceptable return ``mar`` (per period, default 0).
    NaNs are dropped; a series that never falls below the MAR has ``DD = 0`` and a ``nan`` ratio.
    """

    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def __init__(self, mar: float = 0.0, *, periods_per_year: int = 12) -> None:
        self.mar = mar
        self.periods_per_year = periods_per_year

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        if not isinstance(oos_output, WeightsOutput | PanelWeightsOutput):
            raise TypeError("SortinoEvaluator requires a WeightsOutput or PanelWeightsOutput")
        s = oos_output.strategy_returns()
        r = s.to_numpy(dtype=np.float64)
        r = r[~np.isnan(r)]
        ann = float(np.sqrt(self.periods_per_year))
        downside = np.minimum(r - self.mar, 0.0)
        dd = float(np.sqrt(np.mean(downside**2))) if r.size else float("nan")
        if r.size < 2 or dd == 0.0:
            sortino = float("nan")
        else:
            sortino = float((np.mean(r) - self.mar) / dd) * ann
        return _frame([_row(oos_output, "sortino", sortino, s.index[-1])])


def _pricing_means(
    out: PricingOutput,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Per-asset joint time-mean predicted and realized returns + finite mask + attrition counts.

    Each asset's predicted and realized means are taken over the *same* origins — those where both
    are finite — so a period present in one series but missing in the other can never pull the two
    means off a different sample. Assets with no jointly-observed origin drop out (``finite`` is
    ``False``). ``n_obs`` / ``n_dropped`` count the jointly-finite vs. excluded *candidate* cells:
    a candidate is a cell where **either** side is finite. A cell absent on both sides is
    structural (a ragged universe legitimately leaves an entering/exiting asset unpriced and
    unrealized) and counts neither as observed nor as dropped; one-sided missingness still counts
    as dropped. A majority-missing candidate set raises (see :func:`_attrition`).
    """
    p = out.predicted.to_numpy(dtype=np.float64)
    r = out.realized.to_numpy(dtype=np.float64)
    mask = _joint_finite_mask(p, r)
    candidates = np.isfinite(p) | np.isfinite(r)
    n_obs, n_dropped = _attrition(mask, "pricing evaluator", candidates=candidates)
    pj = np.where(mask, p, np.nan)
    rj = np.where(mask, r, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # assets with no joint cell -> NaN mean
        mp = np.nanmean(pj, axis=0)
        mr = np.nanmean(rj, axis=0)
    finite = np.isfinite(mp) & np.isfinite(mr)
    return mp, mr, finite, n_obs, n_dropped


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
        mp, mr, finite, n_obs, n_dropped = _pricing_means(oos_output)
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
        return _frame(
            [
                _row(
                    oos_output,
                    "xs_r2",
                    r2,
                    _pricing_date(oos_output),
                    n_obs=n_obs,
                    n_dropped=n_dropped,
                )
            ]
        )


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
        mp, mr, finite, n_obs, n_dropped = _pricing_means(oos_output)
        alpha = (mr - mp)[finite]
        value = float(np.mean(np.abs(alpha))) if alpha.size else float("nan")
        return _frame(
            [
                _row(
                    oos_output,
                    "avg_abs_alpha",
                    value,
                    _pricing_date(oos_output),
                    n_obs=n_obs,
                    n_dropped=n_dropped,
                )
            ]
        )


# Bundled native evaluators register on import (open registry).
register_evaluator("sharpe", SharpeEvaluator(), overwrite=True)
register_evaluator("ceq", CEQEvaluator(), overwrite=True)
register_evaluator("mean_return", MeanReturnEvaluator(), overwrite=True)
register_evaluator("strategy_return", StrategyReturnEvaluator(), overwrite=True)
register_evaluator("oos_r2", OutOfSampleR2Evaluator(), overwrite=True)
register_evaluator("sed", SquaredErrorDiffEvaluator(), overwrite=True)
register_evaluator("clark_west", ClarkWestEvaluator(), overwrite=True)
register_evaluator("xs_r2", CrossSectionalR2Evaluator(), overwrite=True)
register_evaluator("avg_abs_alpha", AverageAbsAlphaEvaluator(), overwrite=True)
register_evaluator("sortino", SortinoEvaluator(), overwrite=True)
register_evaluator("ic", ICEvaluator(), overwrite=True)
register_evaluator("exposure", ExposureEvaluator(), overwrite=True)
# TreynorEvaluator / InformationRatioEvaluator / M2Evaluator take a factor frame or benchmark
# series at construction (like AlphaEvaluator), so they are exported for direct use but not
# registered with a default in the zero-argument open registry.
