"""The skfolio adapter: a skfolio optimizer runs as a leak-free numeraire to_weights citizen.

skfolio is an optional dependency; these tests self-skip when it is absent (they run in the
optional CI job that installs the ``[skfolio]`` extra). They check the broadcast contract, meta
provenance, determinism, the conformance suite, an engine round-trip, and that weights come from
``.weights_`` (never ``.predict`` on the test window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire import WalkForwardSplitter, walk_forward
from numeraire.core.data import TimeSeriesView
from numeraire.core.schema import validate_result
from numeraire.testing import check_estimator

pytest.importorskip("skfolio")
from numeraire.adapters.skfolio_adapter import SkfolioWeights


def _view(n: int = 60, n_assets: int = 5, seed: int = 0) -> TimeSeriesView:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    cols = [f"a{i}" for i in range(n_assets)]
    ret = pd.DataFrame(rng.normal(0.01, 0.04, (n, n_assets)), index=idx, columns=cols)
    return TimeSeriesView(ret, horizon=1)


def _estimators():
    from skfolio.optimization import HierarchicalRiskParity, MeanRisk, RiskBudgeting

    return {
        "MeanRisk": MeanRisk(),
        "RiskBudgeting": RiskBudgeting(),
        "HRP": HierarchicalRiskParity(),
    }


@pytest.mark.parametrize("name", ["MeanRisk", "RiskBudgeting", "HRP"])
def test_fit_broadcasts_weights_that_sum_to_one(name: str) -> None:
    view = _view()
    model = SkfolioWeights(_estimators()[name]).fit(view)
    w = model.to_weights(view)
    assert list(w.columns) == view.assets
    assert w.index.equals(view.calendar)
    # a single optimal vector broadcast across every date (all rows identical, sum to 1)
    assert np.allclose(w.to_numpy().sum(axis=1), 1.0)
    assert np.allclose(w.to_numpy(), w.to_numpy()[0])


def test_meta_records_provenance() -> None:
    model = SkfolioWeights(window=48).fit(_view())
    assert model.meta["adapter"] == "skfolio"
    assert model.meta["estimator"] == "MeanRisk"
    assert isinstance(model.meta["skfolio_version"], str)
    assert model.meta["n_train"] == 48  # window capped the lookback


def test_weights_come_from_weights_attr_not_predict() -> None:
    # the adapter's weights must equal skfolio's fitted .weights_ on the same window (no predict)
    from skfolio.optimization import MeanRisk

    view = _view()
    ref = MeanRisk().fit(view.returns_frame())
    model = SkfolioWeights(MeanRisk()).fit(view)
    got = model.to_weights(view).to_numpy()[0]
    np.testing.assert_allclose(got, np.asarray(ref.weights_).ravel(), atol=1e-10)


def test_determinism() -> None:
    view = _view()
    w1 = SkfolioWeights().fit(view).to_weights(view).to_numpy()
    w2 = SkfolioWeights().fit(view).to_weights(view).to_numpy()
    np.testing.assert_allclose(w1, w2)


def test_conformance_suite_passes() -> None:
    check_estimator(SkfolioWeights(), lambda: _view(60, 5, seed=3), min_train=24)


def test_engine_roundtrip_emits_valid_rows() -> None:
    view = _view(60, 4, seed=5)
    out = walk_forward(
        SkfolioWeights(window=24),
        view,
        WalkForwardSplitter(min_train=24, test_size=1, expanding=True),
        method="skfolio_mean_risk",
    )
    from numeraire import SharpeEvaluator

    rows = SharpeEvaluator().evaluate(out)
    validate_result(rows)
    assert (rows["method"] == "skfolio_mean_risk").all()


def test_missing_asset_coverage_raises() -> None:
    model = SkfolioWeights().fit(_view(60, 5, seed=0))  # fit universe a0..a4
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    ret = pd.DataFrame(0.01, index=idx, columns=["a0", "zzz"])  # zzz not in the fit universe
    with pytest.raises(ValueError, match="do not cover view assets"):
        model.to_weights(TimeSeriesView(ret, horizon=1))


def test_requires_timeseriesview() -> None:
    with pytest.raises(TypeError, match="TimeSeriesView"):
        SkfolioWeights().fit(object())  # type: ignore[arg-type]
