"""Inference primitives for asset-pricing evaluation (pure numpy/scipy, no heavy deps).

Small, closed-form statistical tests the evaluator layer and reference-result tests build on:

- :func:`grs_test` — Gibbons-Ross-Shanken (1989) joint zero-alpha F-test of a factor model on a
  set of test assets (exact small-sample F under i.i.d. normal errors).
- :func:`sharpe_diff_test` — Jobson-Korkie (1981) paired Sharpe-ratio difference z-test with the
  Memmel (2003) variance correction (the convention of the 1/N-style horse races).
- :func:`clark_west` — Clark-West (2007) MSPE-adjusted test for nested forecast comparisons
  (the companion to the Goyal-Welch OOS R²; plain Diebold-Mariano is oversized for nested models).
- :func:`alpha_regression` — time-series alpha vs a factor benchmark with HAC (Newey-West)
  standard errors (the volatility-managed-portfolio-style headline regression).
- :func:`fama_macbeth` — Fama-MacBeth (1973) two-pass cross-sectional risk-premia estimation with
  FM t-statistics, optional Shanken (1992) errors-in-variables and Newey-West corrections.
- :func:`adjust_tests` — multiple-testing adjustments for factor-zoo sweeps (Bonferroni, Holm,
  Benjamini-Yekutieli), the Harvey-Liu-Zhu (2016) toolbox behind the "t > 3.0" hurdle.
- :func:`newey_west_lrv` — the shared Bartlett-kernel long-run variance helper.

The mean-variance *economic-value* family (the 1/N-horse-race metrics):

- :func:`certainty_equivalent` — DeMiguel-Garlappi-Uppal (2009) eq. 12 certainty-equivalent return
  of a strategy's realized returns (``mean - gamma/2 var``); their headline utility metric.
- :func:`return_loss` — DGU (2009) eq. 17 return-loss of a strategy vs a benchmark (the extra
  return the benchmark's Sharpe line delivers at the strategy's risk, net of the strategy's mean).
- :func:`performance_fee` — Fleming-Kirby-Ostdiek quadratic-utility performance fee: the per-period
  fee equating ``E[U(benchmark)]`` and ``E[U(candidate - fee)]``.

All functions take plain arrays/frames and return frozen result dataclasses (or a scalar for the
economic-value metrics); evaluator classes in :mod:`numeraire.core.evaluators` adapt them to OOS
outputs and the tidy result schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as _sps

from numeraire.core.data import Float


def newey_west_lrv(x: Float, lags: int = 0) -> float:
    """Bartlett-kernel long-run variance of a 1-D series (``lags=0`` = plain variance, MLE).

    ``lrv = g0 + 2 * sum_{l=1..lags} (1 - l/(lags+1)) * g_l`` with ``g_l`` the lag-``l``
    autocovariance (denominator ``T``).
    """
    if lags < 0:
        raise ValueError(f"lags must be >= 0; got {lags}")
    v = np.asarray(x, dtype=np.float64)
    v = v - v.mean()
    n = len(v)
    if n == 0:
        return float("nan")
    lrv = float(v @ v) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        lrv += 2.0 * w * float(v[lag:] @ v[:-lag]) / n
    return lrv


_MT_METHODS = ("bonferroni", "holm", "bhy")


@dataclass(frozen=True)
class MultipleTestResult:
    """Multiple-testing adjustment over a family of p-values (original input order)."""

    method: str
    alpha: float
    n_tests: int
    rejected: NDArray[np.bool_]
    adjusted_p: Float


def adjust_tests(
    p_values: Float, *, method: str = "bhy", alpha: float = 0.05
) -> MultipleTestResult:
    """Multiple-testing adjustment for a family of tests (Harvey-Liu-Zhu 2016 §4.4 toolbox).

    - ``bonferroni`` (single-step, FWER): reject ``p_i <= alpha / M``.
    - ``holm`` (step-down, FWER): order ascending, reject while ``p_(k) <= alpha / (M + 1 - k)``.
    - ``bhy`` (Benjamini-Yekutieli step-up, FDR under arbitrary dependence):
      ``k* = max{k : p_(k) <= k * alpha / (M * c(M))}`` with ``c(M) = sum_{j<=M} 1/j``;
      reject the ``k*`` smallest.

    Adjusted p-values follow the standard conventions (min-with-1, running max/min so rejection
    by ``adjusted_p <= alpha`` matches the sequential rule). HLZ's headline: with the factor
    zoo's family size, a new factor needs roughly ``t > 3.0`` (BHY 1%) rather than 1.96.
    """
    if method not in _MT_METHODS:
        raise ValueError(f"method must be one of {_MT_METHODS}; got {method!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p_values must be a non-empty 1-D array")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must lie in [0, 1]")
    m = len(p)
    order = np.argsort(p, kind="stable")
    ps = p[order]
    adj_sorted = np.empty(m, dtype=np.float64)
    if method == "bonferroni":
        adj_sorted = np.minimum(m * ps, 1.0)
    elif method == "holm":
        # running max of (M + 1 - k) * p_(k) preserves the step-down rejection order
        adj_sorted = np.minimum(np.maximum.accumulate((m - np.arange(m)) * ps), 1.0)
    else:  # bhy
        c_m = float((1.0 / np.arange(1, m + 1)).sum())
        raw = m * c_m / np.arange(1, m + 1) * ps
        adj_sorted = np.minimum(np.minimum.accumulate(raw[::-1])[::-1], 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adj_sorted
    rejected = adjusted <= alpha
    return MultipleTestResult(
        method=method, alpha=alpha, n_tests=m, rejected=rejected, adjusted_p=adjusted
    )


@dataclass(frozen=True)
class GRSResult:
    """GRS joint zero-alpha test: ``F ~ F(n_assets, n_obs - n_assets - n_factors)`` under H0."""

    f_stat: float
    p_value: float
    n_obs: int
    n_assets: int
    n_factors: int
    alphas: Float
    avg_abs_alpha: float
    sh2_factors: float  # max squared sample Sharpe attainable from the factors (MLE moments)


def grs_test(assets: pd.DataFrame, factors: pd.DataFrame) -> GRSResult:
    """Gibbons-Ross-Shanken (1989) test that all time-series alphas are jointly zero.

    ``assets`` are ``(date x N)`` test-asset **excess** returns, ``factors`` ``(date x K)``
    factor excess returns on the identical index. The statistic (multifactor form)::

        F = (T - N - K) / N * (a' Sigma^-1 a) / (1 + Sh(F)^2)  ~  F(N, T - N - K)

    with ``Sigma`` the MLE residual covariance and ``Sh(F)^2 = mu' Omega^-1 mu`` the factors'
    max squared sample Sharpe (MLE moments). Requires ``T > N + K``.
    """
    if not assets.index.equals(factors.index):
        raise ValueError("assets and factors must share one identical index (align upstream)")
    y = assets.to_numpy(dtype=np.float64)
    f = factors.to_numpy(dtype=np.float64)
    t_n, n = y.shape
    k = f.shape[1]
    if t_n - n - k <= 0:
        raise ValueError(f"GRS needs T > N + K; got T={t_n}, N={n}, K={k}")
    x = np.column_stack([np.ones(t_n), f])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    alphas = np.asarray(coef[0], dtype=np.float64)
    resid = y - x @ coef
    sigma = (resid.T @ resid) / t_n  # MLE
    mu = f.mean(axis=0)
    omega = ((f - mu).T @ (f - mu)) / t_n  # MLE
    sh2 = float(mu @ np.linalg.solve(omega, mu))
    quad = float(alphas @ np.linalg.solve(sigma, alphas))
    f_stat = (t_n - n - k) / n * quad / (1.0 + sh2)
    p = float(_sps.f.sf(f_stat, n, t_n - n - k))
    return GRSResult(
        f_stat=float(f_stat),
        p_value=p,
        n_obs=t_n,
        n_assets=n,
        n_factors=k,
        alphas=alphas,
        avg_abs_alpha=float(np.abs(alphas).mean()),
        sh2_factors=sh2,
    )


@dataclass(frozen=True)
class SharpeDiffResult:
    """Paired Sharpe difference: ``z`` is asymptotically standard normal under equal Sharpe."""

    sharpe_a: float
    sharpe_b: float
    z_stat: float
    p_value: float  # two-sided
    n_obs: int


def sharpe_diff_test(a: Float, b: Float) -> SharpeDiffResult:
    """Jobson-Korkie (1981) z-test of equal Sharpe ratios with the Memmel (2003) correction.

    ``a`` and ``b`` are two aligned return series (same periods). The statistic tests
    ``H0: mu_a/sigma_a = mu_b/sigma_b`` using the asymptotic variance::

        theta = (1/T) * (2 s_a^2 s_b^2 - 2 s_a s_b s_ab + mu_a^2 s_b^2 / 2 + mu_b^2 s_a^2 / 2
                          - mu_a mu_b s_ab^2 / (s_a s_b))
        z = (s_b mu_a - s_a mu_b) / sqrt(theta)
    """
    ra = np.asarray(a, dtype=np.float64)
    rb = np.asarray(b, dtype=np.float64)
    if ra.shape != rb.shape or ra.ndim != 1:
        raise ValueError("a and b must be 1-D return series of identical length")
    t_n = len(ra)
    if t_n < 3:
        raise ValueError("need at least 3 paired observations")
    mu_a, mu_b = float(ra.mean()), float(rb.mean())
    s_a, s_b = float(ra.std(ddof=1)), float(rb.std(ddof=1))
    s_ab = float(np.cov(ra, rb, ddof=1)[0, 1])
    theta = (
        2.0 * s_a**2 * s_b**2
        - 2.0 * s_a * s_b * s_ab
        + 0.5 * mu_a**2 * s_b**2
        + 0.5 * mu_b**2 * s_a**2
        - (mu_a * mu_b / (s_a * s_b)) * s_ab**2
    ) / t_n
    z = (s_b * mu_a - s_a * mu_b) / np.sqrt(theta) if theta > 0 else float("nan")
    p = 2.0 * float(_sps.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return SharpeDiffResult(
        sharpe_a=mu_a / s_a, sharpe_b=mu_b / s_b, z_stat=float(z), p_value=p, n_obs=t_n
    )


@dataclass(frozen=True)
class ClarkWestResult:
    """Clark-West MSPE-adjusted comparison of nested forecasts (one-sided: model beats bench)."""

    mspe_benchmark: float
    mspe_model: float
    t_stat: float
    p_value: float  # one-sided, H1: model improves on the nested benchmark
    n_obs: int


def clark_west(
    realized: Float, forecast: Float, benchmark: Float, *, nw_lags: int = 0
) -> ClarkWestResult:
    """Clark-West (2007) MSPE-adjusted test for nested models.

    Per-period adjusted loss difference ``f_t = e_b^2 - (e_m^2 - (bench - model)^2)``; the
    t-statistic of its mean (HAC long-run variance with ``nw_lags``; use ``horizon - 1`` for
    multi-step forecasts) is compared to one-sided standard-normal critical values. Degenerate
    when the model equals the benchmark (``t = nan``).
    """
    r = np.asarray(realized, dtype=np.float64)
    m = np.asarray(forecast, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    if not (r.shape == m.shape == b.shape) or r.ndim != 1:
        raise ValueError("realized/forecast/benchmark must be aligned 1-D series")
    e_b = r - b
    e_m = r - m
    adj = e_b**2 - (e_m**2 - (b - m) ** 2)
    t_n = len(adj)
    if t_n < 3:
        raise ValueError("need at least 3 forecast origins")
    lrv = newey_west_lrv(adj, nw_lags)
    se = float(np.sqrt(lrv / t_n))
    t_stat = float(adj.mean() / se) if se > 0 else float("nan")
    p = float(_sps.norm.sf(t_stat)) if np.isfinite(t_stat) else float("nan")
    return ClarkWestResult(
        mspe_benchmark=float(np.mean(e_b**2)),
        mspe_model=float(np.mean(e_m**2)),
        t_stat=t_stat,
        p_value=p,
        n_obs=t_n,
    )


@dataclass(frozen=True)
class AlphaResult:
    """Time-series alpha regression ``r_p = alpha + beta' F + e`` with HAC standard errors."""

    alpha: float  # per period
    alpha_t: float
    p_value: float  # two-sided, normal
    betas: Float
    r2: float
    n_obs: int


def alpha_regression(
    portfolio: pd.Series, factors: pd.DataFrame, *, nw_lags: int = 0
) -> AlphaResult:
    """OLS of portfolio (excess) returns on factor returns; HAC (Bartlett) coefficient errors.

    ``nw_lags=0`` gives White heteroskedasticity-robust standard errors; positive lags add the
    Newey-West autocorrelation correction. Rows are inner-joined on the index.
    """
    joined = pd.concat([portfolio.rename("_p"), factors], axis=1, join="inner").dropna()
    k = factors.shape[1]
    if len(joined) < k + 3:
        raise ValueError(f"need at least K+3 overlapping observations; got {len(joined)}")
    y = joined["_p"].to_numpy(dtype=np.float64)
    f = joined.drop(columns="_p").to_numpy(dtype=np.float64)
    t_n = len(y)
    x = np.column_stack([np.ones(t_n), f])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    xtx_inv = np.linalg.inv(x.T @ x)
    u = x * resid[:, None]  # (T x p) score contributions
    s = u.T @ u
    for lag in range(1, min(nw_lags, t_n - 1) + 1):
        w = 1.0 - lag / (nw_lags + 1.0)
        gamma = u[lag:].T @ u[:-lag]
        s += w * (gamma + gamma.T)
    v = xtx_inv @ s @ xtx_inv
    se_alpha = float(np.sqrt(v[0, 0]))
    alpha = float(coef[0])
    t_stat = alpha / se_alpha if se_alpha > 0 else float("nan")
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return AlphaResult(
        alpha=alpha,
        alpha_t=float(t_stat),
        p_value=2.0 * float(_sps.norm.sf(abs(t_stat))) if np.isfinite(t_stat) else float("nan"),
        betas=np.asarray(coef[1:], dtype=np.float64),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        n_obs=t_n,
    )


@dataclass(frozen=True)
class FamaMacBethResult:
    """Two-pass Fama-MacBeth (1973) cross-sectional risk-premia estimates + FM t-statistics.

    ``premia`` / ``t_stats`` / ``se`` are aligned to ``names`` (``"const"`` then the factor
    columns). ``t_stats`` carry the Shanken (1992) errors-in-variables inflation and/or the
    Newey-West autocorrelation correction exactly as requested at call time.
    """

    names: tuple[str, ...]
    premia: Float
    t_stats: Float
    se: Float
    n_periods: int
    n_assets: int
    shanken: bool
    nw_lags: int


def fama_macbeth(
    returns_panel: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    shanken: bool = False,
    nw_lags: int = 0,
) -> FamaMacBethResult:
    """Two-pass Fama-MacBeth (1973) cross-sectional regression of returns on factor exposures.

    ``returns_panel`` are ``(date x asset)`` test-asset returns, ``factors`` ``(date x K)`` factor
    returns; the two are inner-joined on their date index. The procedure::

        pass 1 (time series):  r_i = a_i + b_i' F + e_i         -> beta_i  (per asset i)
        pass 2 (cross section, each date t):
                               r_t = gamma0_t + gamma_t' beta + u_t
        premia = mean_t [gamma0_t, gamma_t] ,  FM se = std_t / sqrt(T)

    The first pass estimates each asset's factor loadings on the full sample; each period's
    cross-sectional OLS of that period's returns on those loadings gives a premia draw, and the
    Fama-MacBeth estimate is the time-series mean of the draws with the time-series-of-means
    standard error (so cross-sectional correlation is handled by construction).

    ``nw_lags > 0`` replaces the i.i.d. FM variance of the premia series with its Newey-West
    (Bartlett) long-run variance (autocorrelated premia). ``shanken=True`` multiplies the premia
    variances by the Shanken (1992) errors-in-variables factor ``1 + lambda' Sigma_f^-1 lambda``
    (``lambda`` the estimated slope premia, ``Sigma_f`` the MLE factor covariance), correcting the
    downward bias from using *estimated* betas as regressors; it inflates the standard errors. The
    two corrections compose. Per period, only assets with a finite return and finite loadings enter
    that cross-section; an asset needs enough finite observations to identify its betas.
    """
    if nw_lags < 0:
        raise ValueError(f"nw_lags must be >= 0; got {nw_lags}")
    joined = returns_panel.join(factors, how="inner", lsuffix="_ret", rsuffix="_fac")
    ret = joined.iloc[:, : returns_panel.shape[1]].to_numpy(dtype=np.float64)
    fac = joined.iloc[:, returns_panel.shape[1] :].to_numpy(dtype=np.float64)
    t_n, n_assets = ret.shape
    k = fac.shape[1]
    names = ("const", *(str(c) for c in factors.columns))
    if t_n < k + 2:
        raise ValueError(f"need at least K+2 overlapping dates; got T={t_n}, K={k}")

    # --- pass 1: per-asset time-series loadings (drop that asset's non-finite rows) ---
    betas = np.full((n_assets, k), np.nan, dtype=np.float64)
    for i in range(n_assets):
        m = np.isfinite(ret[:, i]) & np.isfinite(fac).all(axis=1)
        if int(m.sum()) < k + 1:
            continue
        x = np.column_stack([np.ones(int(m.sum())), fac[m]])
        yb = np.asarray(ret[m, i], dtype=np.float64)
        coef, *_ = np.linalg.lstsq(x, yb, rcond=None)
        betas[i] = coef[1:]

    # --- pass 2: per-date cross-sectional premia on [1, betas] ---
    beta_ok = np.isfinite(betas).all(axis=1)
    gammas: list[NDArray[np.float64]] = []
    for t in range(t_n):
        m = beta_ok & np.isfinite(ret[t])
        if int(m.sum()) < k + 1:
            continue
        x = np.column_stack([np.ones(int(m.sum())), betas[m]])
        yb = np.asarray(ret[t, m], dtype=np.float64)
        coef, *_ = np.linalg.lstsq(x, yb, rcond=None)
        gammas.append(np.asarray(coef, dtype=np.float64).ravel())
    if len(gammas) < 2:
        raise ValueError("fewer than two identifiable cross-sections — cannot form FM t-stats")
    gamma = np.vstack(gammas)  # (T_eff x (K+1))
    t_eff = gamma.shape[0]
    premia = gamma.mean(axis=0)

    var = np.array(
        [newey_west_lrv(gamma[:, j], nw_lags) / t_eff for j in range(k + 1)], dtype=np.float64
    )
    if shanken:
        fac_ok = fac[np.isfinite(fac).all(axis=1)]
        omega = np.asarray(np.cov(fac_ok, rowvar=False, ddof=0), dtype=np.float64).reshape(k, k)
        lam = np.asarray(premia[1:], dtype=np.float64)
        correction = 1.0 + float(lam @ np.linalg.solve(omega, lam))
        var = var * correction
    se = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, premia / se, np.nan)
    return FamaMacBethResult(
        names=names,
        premia=premia,
        t_stats=t_stats,
        se=se,
        n_periods=t_eff,
        n_assets=n_assets,
        shanken=shanken,
        nw_lags=nw_lags,
    )


# --------------------------------------------------------------------------------- economic value


def certainty_equivalent(returns: Float, gamma: float = 1.0, *, ddof: int = 0) -> float:
    """DGU (2009) eq. 12 certainty-equivalent return: ``mean - (gamma/2) * var``.

    The per-period CEQ of a mean-variance investor holding ``returns`` (a realized OOS strategy
    return series). ``gamma`` is risk aversion (DGU report ``gamma=1``); ``ddof=0`` matches their
    tabulated (MLE) variance. Same units/frequency as the input — DGU's tables are monthly. Higher
    is better. NaNs are dropped; fewer than two observations yields NaN.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return float("nan")
    return float(r.mean() - 0.5 * gamma * r.var(ddof=ddof))


def return_loss(candidate: Float, benchmark: Float, *, ddof: int = 0) -> float:
    """DGU (2009) eq. 17 return-loss of ``candidate`` relative to ``benchmark``.

    The additional expected return the benchmark would earn on its own risk-return line at the
    candidate's risk, minus the candidate's own mean:
    ``(mean_bench / std_bench) * std_cand - mean_cand``. **Positive => the candidate underperforms**
    the benchmark's Sharpe trade-off (the DGU sign convention). Both are aligned per-period return
    series; NaNs are dropped pairwise. ``ddof=0`` matches DGU's MLE moments.
    """
    c = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    if c.shape != b.shape or c.ndim != 1:
        raise ValueError("candidate and benchmark must be aligned 1-D return series")
    keep = ~(np.isnan(c) | np.isnan(b))
    c, b = c[keep], b[keep]
    sd_b = float(b.std(ddof=ddof))
    if c.size < 2 or sd_b == 0.0:
        return float("nan")
    return float((b.mean() / sd_b) * float(c.std(ddof=ddof)) - c.mean())


def performance_fee(candidate: Float, benchmark: Float, gamma: float) -> float:
    """Quadratic-utility performance fee (Fleming-Kirby-Ostdiek; Kirby-Ostdiek 2012 eq. 23).

    The maximum per-period fee an investor with ``U(R) = R - gamma/2 R^2`` would pay to switch from
    ``benchmark`` to ``candidate`` (both **raw** return series): the ``fee`` solving
    ``E[U(benchmark)] = E[U(candidate - fee)]``. Annualize as ``fee * periods`` (``* 1e4`` for
    bp/yr). Positive => the candidate is worth paying for; NaN when no real fee equates the two
    utilities (a deeply dominated candidate). NaNs are dropped pairwise.
    """
    ri = np.asarray(benchmark, dtype=np.float64)
    rj = np.asarray(candidate, dtype=np.float64)
    if ri.shape != rj.shape or ri.ndim != 1:
        raise ValueError("candidate and benchmark must be aligned 1-D raw return series")
    keep = ~(np.isnan(ri) | np.isnan(rj))
    ri, rj = ri[keep], rj[keep]
    if ri.size == 0:
        return float("nan")

    def eu(r: Float) -> float:
        return float(r.mean() - gamma / 2.0 * (r**2).mean())

    a = 1.0 - gamma * float(rj.mean())
    disc = a**2 - 2.0 * gamma * (eu(ri) - eu(rj))
    if disc < 0:
        return float("nan")
    return (-a + float(np.sqrt(disc))) / gamma
