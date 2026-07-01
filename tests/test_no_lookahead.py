"""The canonical no-look-ahead property test.

A one-period contemporaneous leak once flipped a VoC OOS R^2 from -6% to a "significant
+1.87%". The spine must make that structurally impossible: across *any* walk-forward split,
no information realized at or after the first test date may enter the training fold. These
properties pin that invariant over randomized data, horizons, and split parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.splitter import WalkForwardSplitter


def _view(n: int, horizon: int, seed: int) -> TimeSeriesView:
    rng = np.random.default_rng(seed)
    index = pd.date_range("1980-01-31", periods=n, freq="ME")
    returns = pd.DataFrame(rng.normal(0, 0.05, (n, 1)), index=index, columns=["r0"])
    features = pd.DataFrame(rng.normal(0, 1, (n, 2)), index=index, columns=["x0", "x1"])
    return TimeSeriesView(returns, features, horizon=horizon)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=40, max_value=300),
    horizon=st.integers(min_value=1, max_value=6),
    min_train=st.integers(min_value=12, max_value=60),
    test_size=st.integers(min_value=1, max_value=24),
    embargo=st.integers(min_value=0, max_value=6),
    expanding=st.booleans(),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_no_lookahead_across_random_splits(
    n: int,
    horizon: int,
    min_train: int,
    test_size: int,
    embargo: int,
    expanding: bool,
    seed: int,
) -> None:
    view = _view(n, horizon, seed)
    full = view.calendar
    sp = WalkForwardSplitter(
        min_train=min_train, test_size=test_size, embargo=embargo, expanding=expanding
    )
    for train, test in sp.split(view):
        first_test = test.calendar.min()

        # (1) train and test calendars never overlap; test is strictly future.
        assert train.calendar.max() < first_test

        dates_tr, _, y_tr = train.aligned()
        if len(dates_tr) == 0:
            continue

        # (2) every training target is fully realized strictly before the first test date.
        last_feat = dates_tr.max()
        pos_last = int(full.searchsorted(last_feat))
        realized_at = full[pos_last + horizon]  # date the last target lands on
        assert realized_at < first_test

        # (3) horizon purge: the last training feature sits >= horizon steps before the cutoff.
        pos_cut = int(full.searchsorted(train.calendar.max()))
        assert pos_cut - pos_last >= horizon

        # (4) no NaNs leaked in as targets (all training pairs are realized).
        assert not np.isnan(y_tr).any()


def _rand_panel(n_dates: int, n_assets: int, seed: int) -> pd.DataFrame:
    """A random ragged panel; asset A00 spans the full range so the calendar has every date."""
    rng = np.random.default_rng(seed)
    cal = pd.date_range("1990-01-31", periods=n_dates, freq="ME")
    rows: list[dict[str, object]] = []
    for j in range(n_assets):
        start, end = (0, n_dates) if j == 0 else _rand_span(rng, n_dates)
        for m in range(start, end):
            rows.append(
                {
                    "date": cal[m],
                    "asset": f"A{j:02d}",
                    "x0": float(rng.normal()),
                    "x1": float(rng.normal()),
                    "ret": float(rng.normal(0, 0.05)),
                }
            )
    return pd.DataFrame(rows)


def _rand_span(rng: np.random.Generator, n: int) -> tuple[int, int]:
    start = int(rng.integers(0, n - 2))
    return start, int(rng.integers(start + 2, n + 1))


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_dates=st.integers(min_value=24, max_value=80),
    n_assets=st.integers(min_value=2, max_value=6),
    horizon=st.integers(min_value=1, max_value=4),
    min_train=st.integers(min_value=6, max_value=30),
    test_size=st.integers(min_value=1, max_value=12),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_panel_no_lookahead_across_random_splits(
    n_dates: int, n_assets: int, horizon: int, min_train: int, test_size: int, seed: int
) -> None:
    cv = CrossSectionView(_rand_panel(n_dates, n_assets, seed), chars=["x0", "x1"], horizon=horizon)
    full = cv.calendar
    sp = WalkForwardSplitter(min_train=min_train, test_size=test_size)
    for train, test in sp.split(cv):
        first_test = test.calendar.min()
        assert train.calendar.max() < first_test  # test strictly future of train

        keys, x, y = train.aligned()
        if len(keys) == 0:
            continue
        # no NaN leaked into the panel design (missing-char / unrealized rows already dropped)
        assert not np.isnan(y).any()
        assert not np.isnan(x).any()
        # every training obs's target is realized by the cutoff, strictly before the first test date
        last = keys.get_level_values("date").max()
        pos = int(full.searchsorted(last))
        assert full[pos + horizon] <= train.calendar.max()


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=10, max_value=200),
    horizon=st.integers(min_value=1, max_value=8),
    end_frac=st.floats(min_value=0.2, max_value=1.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_window_never_exposes_future(n: int, horizon: int, end_frac: float, seed: int) -> None:
    view = _view(n, horizon, seed)
    cal = view.calendar
    end = cal[min(n - 1, int(end_frac * (n - 1)))]
    w = view.window(end)
    # window exposes nothing after `end`
    assert w.calendar.max() <= end
    dates, _, _ = w.aligned()
    if len(dates):
        pos_last = int(cal.searchsorted(dates.max()))
        # the last realized target still lands on or before the window end (no peeking past it)
        assert cal[pos_last + horizon] <= end
