"""Refit-cadence decoupling (refit_every) and the validation_split tuning helper.

Two seams the ML-cross-section protocols demand: annual refits with monthly predictions
(stale parameters, never stale information), and hyperparameter selection strictly inside
the train fold (fit window / validation window, both PIT).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire import (
    TimeSeriesView,
    WalkForwardSplitter,
    backtest_forecast,
    validation_split,
)
from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView


def _tsv(n: int = 60, seed: int = 0) -> TimeSeriesView:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    returns = pd.DataFrame({"mkt": rng.normal(0.01, 0.04, size=n)}, index=idx)
    features = pd.DataFrame({"x": rng.normal(size=n)}, index=idx)
    return TimeSeriesView(returns, features)


class _MeanModel:
    """Forecast = mean return of the train window it was FIT on (frozen at fit time)."""

    def __init__(self, mu: float, n_train: int) -> None:
        self.mu = mu
        self.n_train = n_train

    def capabilities(self) -> set[str]:
        return {capabilities.TO_FORECAST}

    def forecast(self, view: TimeSeriesView) -> pd.Series:
        return pd.Series([self.mu], index=view.assets)


class _CountingEstimator:
    """Historical-mean estimator that counts fits (to pin the refit cadence).

    The engine fits an isolated ``copy.deepcopy`` per refit block, so the counter is a shared
    mutable sink threaded through ``__deepcopy__``; otherwise each block would increment a throwaway
    copy and the original instance would report zero fits.
    """

    def __init__(self) -> None:
        self._fits = [0]  # one-element mutable sink, shared across per-block deepcopies

    def __deepcopy__(self, memo: dict[int, object]) -> _CountingEstimator:
        clone = _CountingEstimator()
        clone._fits = self._fits
        return clone

    @property
    def n_fits(self) -> int:
        return self._fits[0]

    def fit(self, view: TimeSeriesView) -> _MeanModel:
        self._fits[0] += 1
        r = view.returns_frame().to_numpy(dtype=np.float64)
        return _MeanModel(float(r.mean()), len(r))


def test_refit_every_one_is_the_default_path() -> None:
    v = _tsv()
    a = backtest_forecast(_CountingEstimator(), v, min_train=20, method="hm")
    b = backtest_forecast(_CountingEstimator(), v, min_train=20, refit_every=1, method="hm")
    pd.testing.assert_frame_equal(a.forecasts, b.forecasts)
    pd.testing.assert_frame_equal(a.benchmark, b.benchmark)


def test_refit_every_counts_fits_and_freezes_parameters() -> None:
    v = _tsv()
    est = _CountingEstimator()
    out = backtest_forecast(est, v, min_train=20, refit_every=12, method="hm")
    n_origins = len(out.forecasts)
    assert est.n_fits == int(np.ceil(n_origins / 12))
    # within a refit block the forecast (frozen mean) is constant; it jumps only at refits
    f = out.forecasts.to_numpy(dtype=np.float64).ravel()
    assert np.allclose(f[:12], f[0])  # first block frozen
    assert f[12] != f[11]  # refit boundary re-estimates
    # the benchmark stays the per-origin prevailing mean (moves every origin, cadence-free)
    b = out.benchmark.to_numpy(dtype=np.float64).ravel()
    assert not np.allclose(b[:12], b[0])


def test_refit_every_parallel_equals_serial() -> None:
    v = _tsv(seed=1)
    a = backtest_forecast(
        _CountingEstimator(), v, min_train=20, refit_every=6, method="hm", n_jobs=1
    )
    b = backtest_forecast(
        _CountingEstimator(), v, min_train=20, refit_every=6, method="hm", n_jobs=-1
    )
    pd.testing.assert_frame_equal(a.forecasts, b.forecasts)
    pd.testing.assert_frame_equal(a.realized, b.realized)


def test_refit_every_validates() -> None:
    with pytest.raises(ValueError, match="refit_every"):
        backtest_forecast(_CountingEstimator(), _tsv(), refit_every=0, method="hm")


def test_validation_split_partitions_the_fold_calendar() -> None:
    v = _tsv(n=50)
    train, _test = next(iter(WalkForwardSplitter(min_train=36, test_size=6).split(v)))
    fit, valid = validation_split(train, valid_size=12)
    assert len(valid.calendar) == 12
    assert len(fit.calendar) == len(train.calendar) - 12
    assert fit.calendar.max() < valid.calendar.min()
    joined = fit.calendar.append(valid.calendar)
    assert joined.equals(train.calendar)


def test_validation_split_fit_is_pit_at_the_seam() -> None:
    # fit's supervised pairs must not consume returns realized in the valid period: with h=1 the
    # last aligned fit feature date is strictly before the fit cutoff
    v = _tsv(n=50)
    train, _ = next(iter(WalkForwardSplitter(min_train=36, test_size=6).split(v)))
    fit, valid = validation_split(train, valid_size=12)
    dates, _x, _y = fit.aligned()
    assert dates.max() < fit.calendar.max() or len(dates) == 0
    assert dates.max() < valid.calendar.min()


def test_validation_split_preserves_rolling_fold_start() -> None:
    # on a ROLLING train fold, fit must not resurrect dates before the fold's window
    v = _tsv(n=60)
    folds = list(WalkForwardSplitter(min_train=24, test_size=6, expanding=False).split(v))
    train = folds[-1][0]  # a late rolling fold, its calendar starts well after the data start
    fit, _valid = validation_split(train, valid_size=6)
    assert fit.calendar.min() == train.calendar.min()


def test_validation_split_works_on_panels() -> None:
    rng = np.random.default_rng(2)
    cal = pd.date_range("2001-01-31", periods=30, freq="ME")
    rows = [
        {"date": t, "asset": a, "c": float(rng.normal()), "ret": float(rng.normal(0.01, 0.05))}
        for t in cal
        for a in ("A", "B", "C")
    ]
    v = CrossSectionView(pd.DataFrame(rows), chars=["c"])
    fit, valid = validation_split(v, valid_size=10)
    assert len(valid.calendar) == 10
    assert fit.calendar.max() < valid.calendar.min()
    keys, _x, _y = fit.aligned()
    assert keys.get_level_values("date").max() < valid.calendar.min()


def test_validation_split_guards() -> None:
    v = _tsv(n=10)
    with pytest.raises(ValueError, match="valid_size"):
        validation_split(v, valid_size=0)
    with pytest.raises(ValueError, match="fit dates"):
        validation_split(v, valid_size=9)
