"""walk_forward_panel: OOS backtest of a cross-sectional to_weights method over a ragged panel.

A toy Fama-MacBeth-style estimator (pooled cross-sectional OLS of forward return on characteristics)
emits dollar-neutral long weights; the engine aligns realized forward returns by ``(date, asset)``
key across an entering/exiting universe and scores the portfolio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import toy_panel_wide
from numeraire.core import capabilities
from numeraire.core.data import CharBlock, CrossSectionView
from numeraire.core.engine import PanelWeightsOutput, backtest_panel
from numeraire.core.splitter import WalkForwardSplitter


class _XSModel:
    """Dollar-neutral cross-sectional tilt on the fitted characteristic slope (toy)."""

    def __init__(self, beta: np.ndarray) -> None:
        self._beta = beta

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: CrossSectionView) -> pd.Series:
        dates: list[pd.Timestamp] = []
        assets: list[object] = []
        vals: list[float] = []
        for t in view.calendar:
            ids, x = view.features_asof(t)
            score = x @ self._beta
            w = score - score.mean()  # cross-sectionally demeaned -> dollar-neutral
            norm = float(np.abs(w).sum())
            if norm > 0:
                w = w / norm
            for a, wi in zip(ids, w, strict=True):
                dates.append(t)
                assets.append(a)
                vals.append(float(wi))
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(dates), assets], names=["date", "asset"])
        return pd.Series(vals, index=idx, name="weight")


class _XSEstimator:
    """Pooled cross-sectional OLS of the (t, t+h] return on characteristics over the train panel."""

    def fit(self, view: CrossSectionView) -> _XSModel:
        _keys, x, y = view.aligned()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return _XSModel(beta)


def _view() -> CrossSectionView:
    return CrossSectionView(toy_panel_wide(), chars=["size", "bm", "mom"], horizon=1)


def test_walk_forward_panel_plumbing() -> None:
    v = _view()
    out = backtest_panel(
        _XSEstimator(), v, WalkForwardSplitter(min_train=24, test_size=6), method="toy_fm"
    )
    assert isinstance(out, PanelWeightsOutput)
    assert out.capability == capabilities.TO_WEIGHTS
    assert out.run_id == f"toy_fm-{out.config_hash}"
    # weights and realized share the (date, asset) key index; nothing unrealized survives
    assert out.weights.index.equals(out.realized.index)
    assert list(out.weights.index.names) == ["date", "asset"]
    assert not out.realized.isna().to_numpy().any()


def test_strategy_returns_are_per_date() -> None:
    v = _view()
    out = backtest_panel(
        _XSEstimator(), v, WalkForwardSplitter(min_train=24, test_size=6), method="toy_fm"
    )
    sr = out.strategy_returns()
    # one portfolio return per rebalance date, matching the manual per-date sum of w * r
    manual = (out.weights * out.realized).groupby(level="date").sum()
    assert sr.index.equals(manual.index)
    np.testing.assert_allclose(sr.to_numpy(), manual.to_numpy())
    assert sr.index.is_monotonic_increasing


def test_model_weights_are_dollar_neutral_at_formation() -> None:
    # Neutrality holds on the model's raw cross-section; the engine may later drop a delisted name's
    # weight (its forward return is unknowable), so post-purge sums need not be zero (correct).
    v = _view()
    model = _XSEstimator().fit(v)
    per_date = model.to_weights(v).groupby(level="date").sum()
    np.testing.assert_allclose(per_date.to_numpy(), 0.0, atol=1e-12)


def test_panel_backtest_is_deterministic() -> None:
    v = _view()
    sp = WalkForwardSplitter(min_train=24, test_size=6)
    a = backtest_panel(_XSEstimator(), v, sp, method="toy_fm").strategy_returns()
    b = backtest_panel(_XSEstimator(), v, sp, method="toy_fm").strategy_returns()
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_walk_forward_panel_with_char_block_end_to_end() -> None:
    # a char_block (lagged per-asset) through the panel engine: it resolves into the design matrix,
    # its lag warm-up rows drop in aligned(), and the fit sees the concatenated characteristic
    pan = toy_panel_wide()
    extra = pan[["date", "asset", "size"]].rename(columns={"size": "lagsize"})
    v = CrossSectionView(
        pan, chars=["size", "bm", "mom"], char_blocks=[CharBlock(extra, ["lagsize"], lag=1)]
    )
    out = backtest_panel(
        _XSEstimator(), v, WalkForwardSplitter(min_train=24, test_size=6), method="toy_fm_cb"
    )
    assert isinstance(out, PanelWeightsOutput)
    assert not out.weights.empty
    assert not out.realized.isna().to_numpy().any()
