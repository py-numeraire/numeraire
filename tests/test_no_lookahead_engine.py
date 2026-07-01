"""End-to-end no-look-ahead: corrupting *future* data must not change *past* decisions.

The strongest PIT guard through the actual engines: a decision at date t is fit on data <= its
fold's cutoff and formed from features_asof(t) <= t, so scrambling everything after a cut must leave
every weight before the cut bit-for-bit identical (only the realized P&L, which uses future returns,
may change). Covers both walk_forward (time-series) and walk_forward_panel (ragged cross-section).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import toy_market, toy_panel_wide, toy_predictors
from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.engine import walk_forward, walk_forward_panel
from numeraire.core.splitter import WalkForwardSplitter


class _SignModel:
    def __init__(self, beta: np.ndarray) -> None:
        self._beta = beta

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        rows = [
            np.sign(np.concatenate([[1.0], view.features_asof(t)]) @ self._beta)
            for t in view.calendar
        ]
        return pd.DataFrame(np.vstack(rows), index=view.calendar, columns=view.assets)


class _OLS:
    def fit(self, view: TimeSeriesView) -> _SignModel:
        _, x, y = view.aligned()
        xi = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(xi, y, rcond=None)
        return _SignModel(beta)


class _XSModel:
    def __init__(self, beta: np.ndarray) -> None:
        self._beta = beta

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: CrossSectionView) -> pd.Series:
        d: list[pd.Timestamp] = []
        a: list[object] = []
        v: list[float] = []
        for t in view.calendar:
            ids, x = view.features_asof(t)
            w = x @ self._beta
            w = w - w.mean()
            n = float(np.abs(w).sum())
            if n > 0:
                w = w / n
            for aid, wi in zip(ids, w, strict=True):
                d.append(t)
                a.append(aid)
                v.append(float(wi))
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(d), a], names=["date", "asset"])
        return pd.Series(v, index=idx, name="weight")


class _XS:
    def fit(self, view: CrossSectionView) -> _XSModel:
        _, x, y = view.aligned()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return _XSModel(beta)


def test_walk_forward_weights_invariant_to_future_corruption() -> None:
    mkt, rf = toy_market()
    pred = toy_predictors()
    sp = WalkForwardSplitter(min_train=24, test_size=6)
    base = walk_forward(_OLS(), TimeSeriesView(mkt, pred, risk_free=rf), sp, method="x").weights

    cut = pd.Timestamp("2004-01-31")
    rng = np.random.default_rng(999)
    mkt2, pred2 = mkt.copy(), pred.copy()
    fut = mkt2.index >= cut
    mkt2.loc[fut, "mkt"] = rng.normal(0.0, 5.0, int(fut.sum()))
    pred2.loc[fut, :] = rng.normal(0.0, 50.0, (int(fut.sum()), pred2.shape[1]))
    corrupt = walk_forward(
        _OLS(), TimeSeriesView(mkt2, pred2, risk_free=rf), sp, method="x"
    ).weights

    common = base.index[base.index < cut]
    assert len(common) > 0
    np.testing.assert_array_equal(base.loc[common].to_numpy(), corrupt.loc[common].to_numpy())


def test_walk_forward_panel_weights_invariant_to_future_corruption() -> None:
    pan = toy_panel_wide()
    sp = WalkForwardSplitter(min_train=24, test_size=6)
    base = walk_forward_panel(
        _XS(), CrossSectionView(pan, chars=["size", "bm", "mom"]), sp, method="p"
    ).weights

    cut = pd.Timestamp("1992-06-30")
    rng = np.random.default_rng(7)
    pan2 = pan.copy()
    fut = pan2["date"] >= cut
    for col in ("size", "bm", "mom", "ret"):
        pan2.loc[fut, col] = rng.normal(0.0, 50.0, int(fut.sum()))
    corrupt = walk_forward_panel(
        _XS(), CrossSectionView(pan2, chars=["size", "bm", "mom"]), sp, method="p"
    ).weights

    common = base.index[base.index.get_level_values("date") < cut]
    assert len(common) > 0
    np.testing.assert_allclose(
        base.loc[common].to_numpy(), corrupt.reindex(common).to_numpy(), equal_nan=True
    )
