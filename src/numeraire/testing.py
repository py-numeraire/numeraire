"""Conformance suite — a reusable ``check_estimator`` any method self-certifies against.

Extension authors (in ``numeraire-zoo``, ``numeraire-yourlab``, or a standalone
``numeraire-<method>`` package) run :func:`check_estimator` on their estimator to prove it is a
well-behaved framework citizen *before* wiring it into the engine. It is the sklearn-style
analogue of ``sklearn.utils.estimator_checks.check_estimator``: plain functions that raise
``AssertionError`` on the first violation, no base class to inherit.

This module is **core infrastructure, not a method** — it is exempt from the boundary rule's
methods/adapters import ban (it lives in ``numeraire`` proper, not ``numeraire.core``, and imports
only ``numeraire.core`` + numpy/pandas). It knows the two concrete core view types
(:class:`~numeraire.core.data.TimeSeriesView`, :class:`~numeraire.core.data.CrossSectionView`) so
the no-look-ahead check can build a future-perturbed twin generically.

The checks
----------
Given an ``estimator`` and a **deterministic** ``view_factory`` (a zero-arg callable returning an
equivalent view each call — synthetic data with a fixed seed):

- :func:`check_capabilities` — ``fit`` returns a ``Model`` whose ``capabilities()`` intersect the
  core set ``{to_weights, to_forecast, to_pricing}``, and every *crystallized* capability it
  declares has its method (``to_weights`` / ``forecast``). ``to_pricing`` has no frozen method
  surface yet (it crystallizes with the pricing rule-of-three), so no method name is enforced there.
- :func:`check_output_shapes` — weights columns ⊆ ``view.assets`` and index ⊆ ``view.calendar``
  (wide) or a ``[date, asset]`` MultiIndex (panel); a forecast is a ``pd.Series`` indexed by
  ``view.assets``.
- :func:`check_determinism` — same estimator + same view ⇒ identical output, twice.
- :func:`check_no_lookahead` — the property test. Fit on ``view.window(t)``; the outputs on the
  calendar up to ``t`` must be **invariant to mutating data strictly after ``t``**. A leaky
  estimator that peeks past a prediction date fails here.
- :func:`check_engine_roundtrip` — the estimator runs through the matching walk-forward driver
  without error and an evaluator emits rows conforming to the result schema.

Capability-only methods (``to_pricing``, e.g. an IPCA adapter) exercise :func:`check_capabilities`
and skip the weight/forecast-specific checks — that surface is not frozen, so the suite does not
invent contracts for it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.engine import walk_forward, walk_forward_forecast, walk_forward_panel
from numeraire.core.evaluators import OOSR2Evaluator, SharpeEvaluator
from numeraire.core.schema import validate_result
from numeraire.core.splitter import WalkForwardSplitter

__all__ = [
    "check_capabilities",
    "check_determinism",
    "check_engine_roundtrip",
    "check_estimator",
    "check_no_lookahead",
]

ViewFactory = Callable[[], Any]

# The capabilities whose method surface has crystallized (SupportsWeights / SupportsForecast) and
# the method each mandates. ``to_pricing`` is deliberately absent — it has no frozen method yet.
_CRYSTALLIZED: dict[str, str] = {
    capabilities.TO_WEIGHTS: "to_weights",
    capabilities.TO_FORECAST: "forecast",
}
_CORE_CAPS = frozenset({capabilities.TO_WEIGHTS, capabilities.TO_FORECAST, capabilities.TO_PRICING})


# --------------------------------------------------------------------------- internals


def _fit(estimator: Any, view: Any) -> Any:
    model = estimator.fit(view)
    assert model is not None, "fit() returned None"
    assert hasattr(model, "capabilities") and callable(model.capabilities), (
        "fit() must return a Model exposing capabilities()"
    )
    return model


def _caps(model: Any) -> set[str]:
    caps = model.capabilities()
    assert isinstance(caps, set), f"capabilities() must return a set, got {type(caps).__name__}"
    return caps


def _extractable(caps: set[str]) -> set[str]:
    """The declared capabilities the suite can materialize into outputs (weights / forecasts)."""
    return caps & _CRYSTALLIZED.keys()


def _origin_index(view: Any, frac: float = 0.6) -> int:
    """A calendar position with ample history on either side (for windowed probes/splits)."""
    n = len(view.calendar)
    return min(n - 2, max(1, round(frac * (n - 1))))


def _weights(model: Any, view: Any) -> pd.DataFrame | pd.Series:
    w = model.to_weights(view)
    assert isinstance(w, pd.DataFrame | pd.Series), "to_weights must return a DataFrame or Series"
    return w


def _restrict_weights(w: pd.DataFrame | pd.Series, t: pd.Timestamp) -> pd.DataFrame | pd.Series:
    """Rows dated ``<= t`` (wide index or ``(date, asset)`` MultiIndex)."""
    if isinstance(w, pd.Series):
        dates = w.index.get_level_values("date")
        return w[dates <= t]
    return w.loc[w.index <= t]


def _assert_weights_equal_on_common(
    a: pd.DataFrame | pd.Series, b: pd.DataFrame | pd.Series, msg: str
) -> None:
    idx = a.index.intersection(b.index)
    assert len(idx) > 0, f"{msg}: no overlapping prediction dates to compare"
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        np.testing.assert_allclose(
            a.loc[idx].to_numpy(np.float64),
            b.loc[idx].to_numpy(np.float64),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
            err_msg=msg,
        )
        return
    assert isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)
    cols = a.columns.intersection(b.columns)
    np.testing.assert_allclose(
        a.loc[idx, cols].to_numpy(np.float64),
        b.loc[idx, cols].to_numpy(np.float64),
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True,
        err_msg=msg,
    )


def _perturb_after(view: Any, t: pd.Timestamp) -> Any:
    """A twin of ``view`` identical on data ``<= t`` but randomly perturbed strictly after ``t``.

    Rebuilt from the view's own ejected frames, so an estimator whose output at some date ``d <= t``
    depends on data after ``t`` (a look-ahead leak) produces different outputs on the two twins.
    Handles the two concrete core view types.
    """
    rng = np.random.default_rng(0xC0FFEE)
    if isinstance(view, TimeSeriesView):
        ret = view.returns_frame().copy()
        rmask = ret.index > t
        ret.loc[rmask] = ret.loc[rmask].to_numpy() + rng.normal(0.0, 0.1, ret.loc[rmask].shape)
        if view.feature_names:
            feat = view.features_frame().copy()
            fmask = feat.index > t
            # multiplicative bump preserves sign/positivity of features like a realized variance
            feat.loc[fmask] = feat.loc[fmask].to_numpy() * (
                1.0 + rng.normal(0.0, 0.1, feat.loc[fmask].shape)
            )
            return TimeSeriesView(ret, feat, horizon=view.horizon)
        return TimeSeriesView(ret, horizon=view.horizon)
    if isinstance(view, CrossSectionView):
        pf = view.panel_frame().reset_index()
        mask = pf["date"] > t
        n_after = int(mask.to_numpy().sum())
        for col in [*view.char_names, "ret"]:
            pf.loc[mask, col] = pf.loc[mask, col].to_numpy() + rng.normal(0.0, 0.1, n_after)
        return CrossSectionView(pf, chars=view.char_names, horizon=view.horizon)
    raise TypeError(
        f"check_no_lookahead cannot perturb a {type(view).__name__}; pass a TimeSeriesView "
        "or CrossSectionView (or run the other checks individually)"
    )


# --------------------------------------------------------------------------- public checks


def check_capabilities(estimator: Any, view_factory: ViewFactory) -> None:
    """``fit`` returns a ``Model`` with a recognized capability and its crystallized method(s)."""
    model = _fit(estimator, view_factory())
    caps = _caps(model)
    assert caps, "capabilities() is empty — a model must expose at least one capability"
    assert caps & _CORE_CAPS, (
        f"capabilities() {caps} intersect none of the core set {set(_CORE_CAPS)}"
    )
    for cap, method in _CRYSTALLIZED.items():
        if cap in caps:
            assert callable(getattr(model, method, None)), (
                f"model declares {cap!r} but has no callable {method!r} method"
            )


def check_output_shapes(estimator: Any, view_factory: ViewFactory) -> None:
    """Weights/forecast outputs obey the shape contracts (columns ⊆ assets; forecast index)."""
    view = view_factory()
    model = _fit(estimator, view)
    caps = _caps(model)
    assets = set(view.assets)
    if capabilities.TO_WEIGHTS in caps:
        w = _weights(model, view)
        if isinstance(w, pd.DataFrame):
            assert set(map(str, w.columns)) <= assets, "weights columns must be ⊆ view.assets"
            assert set(w.index) <= set(view.calendar), "weights index must be ⊆ view.calendar"
        else:  # panel: long (date, asset) Series
            assert isinstance(w.index, pd.MultiIndex), "panel weights need a MultiIndex"
            assert list(w.index.names) == ["date", "asset"], (
                "panel weights MultiIndex must be named [date, asset]"
            )
    if capabilities.TO_FORECAST in caps:
        d = view.calendar[_origin_index(view)]
        f = model.forecast(view.window(d))
        assert isinstance(f, pd.Series), "forecast must return a pd.Series"
        assert [str(i) for i in f.index] == view.assets, "forecast index must equal view.assets"


def check_determinism(estimator: Any, view_factory: ViewFactory) -> None:
    """Same estimator + same (deterministic) view ⇒ bit-identical extractable output, twice."""
    ext = _extractable(_caps(_fit(estimator, view_factory())))
    if not ext:
        return  # capability-only method (e.g. to_pricing): no crystallized surface to compare
    m1 = _fit(estimator, view_factory())
    m2 = _fit(estimator, view_factory())
    if capabilities.TO_WEIGHTS in ext:
        _assert_weights_equal_on_common(
            _weights(m1, view_factory()),
            _weights(m2, view_factory()),
            "to_weights is non-deterministic across identical fits",
        )
    if capabilities.TO_FORECAST in ext:
        v = view_factory()
        d = v.calendar[_origin_index(v)]
        np.testing.assert_allclose(
            m1.forecast(v.window(d)).to_numpy(np.float64),
            m2.forecast(v.window(d)).to_numpy(np.float64),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
            err_msg="forecast is non-deterministic across identical fits",
        )


def check_no_lookahead(estimator: Any, view_factory: ViewFactory) -> None:
    """Outputs up to ``t`` are invariant to mutating data strictly after ``t`` (no look-ahead)."""
    view = view_factory()
    cal = view.calendar
    assert len(cal) >= 8, "no-look-ahead check needs a view with >= 8 calendar dates"
    t = cal[_origin_index(view)]
    if not _extractable(_caps(_fit(estimator, view.window(t)))):
        return  # capability-only method: no weight/forecast surface to probe
    twin = _perturb_after(view, t)
    if capabilities.TO_WEIGHTS in _CRYSTALLIZED and capabilities.TO_WEIGHTS in _caps(
        _fit(estimator, view.window(t))
    ):
        # Extract over the FULL view (which contains data after t); a PIT-respecting model windows
        # internally, so its weights at dates <= t must ignore the perturbed tail.
        w_a = _weights(_fit(estimator, view.window(t)), view)
        w_b = _weights(_fit(estimator, twin.window(t)), twin)
        _assert_weights_equal_on_common(
            _restrict_weights(w_a, t),
            _restrict_weights(w_b, t),
            "to_weights at dates <= t changed when only post-t data was mutated (look-ahead)",
        )
    if capabilities.TO_FORECAST in _caps(_fit(estimator, view.window(t))):
        m_a = _fit(estimator, view.window(t))
        m_b = _fit(estimator, twin.window(t))
        k = _origin_index(view)
        for j in range(max(1, k - 2), k + 1):
            d = cal[j]
            np.testing.assert_allclose(
                m_a.forecast(view.window(d)).to_numpy(np.float64),
                m_b.forecast(twin.window(d)).to_numpy(np.float64),
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
                err_msg="forecast at origin <= t changed when only post-t data was mutated",
            )


def check_engine_roundtrip(
    estimator: Any,
    view_factory: ViewFactory,
    *,
    splitter: Any | None = None,
    min_train: int | None = None,
    forecast_kwargs: dict[str, Any] | None = None,
) -> None:
    """The estimator runs through its matching walk-forward driver; result rows validate."""
    view = view_factory()
    caps = _caps(_fit(estimator, view))
    n = len(view.calendar)
    warmup = min_train if min_train is not None else max(2, n // 2)
    if capabilities.TO_WEIGHTS in caps and isinstance(view, TimeSeriesView):
        sp = splitter or WalkForwardSplitter(
            min_train=warmup, test_size=max(1, n - warmup), expanding=True
        )
        out = walk_forward(estimator, view, sp, method="conformance")
        assert not out.weights.empty, (
            "walk_forward produced no weights (widen the fixture/splitter)"
        )
        rows = SharpeEvaluator().evaluate(out)
        validate_result(rows)
        assert len(rows) >= 1
    elif capabilities.TO_WEIGHTS in caps and isinstance(view, CrossSectionView):
        sp = splitter or WalkForwardSplitter(
            min_train=warmup, test_size=max(1, n - warmup), expanding=True
        )
        out = walk_forward_panel(estimator, view, sp, method="conformance")
        assert not out.weights.empty, (
            "walk_forward_panel produced no weights (widen the fixture/splitter)"
        )
        rows = SharpeEvaluator().evaluate(out)
        validate_result(rows)
        assert len(rows) >= 1
    elif capabilities.TO_FORECAST in caps and isinstance(view, TimeSeriesView):
        out = walk_forward_forecast(
            estimator, view, min_train=warmup, method="conformance", **(forecast_kwargs or {})
        )
        assert not out.forecasts.empty, "walk_forward_forecast produced no forecasts"
        rows = OOSR2Evaluator().evaluate(out)
        validate_result(rows)
        assert len(rows) >= 1
    # capability-only methods (to_pricing) have no walk-forward driver yet — nothing to run.


def check_estimator(
    estimator: Any,
    view_factory: ViewFactory,
    *,
    splitter: Any | None = None,
    min_train: int | None = None,
    forecast_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run the full conformance suite; raise ``AssertionError`` on the first violation.

    Parameters
    ----------
    estimator:
        Anything with ``fit(view) -> Model`` (a :class:`~numeraire.core.protocols.Estimator`).
    view_factory:
        A **deterministic** zero-argument callable returning an equivalent view each call —
        synthetic data built with a fixed seed. Determinism is required because several checks
        rebuild the view to compare outputs.
    splitter, min_train, forecast_kwargs:
        Forwarded to :func:`check_engine_roundtrip` to size the walk-forward run for the fixture
        (e.g. a small ``min_train`` for a short synthetic view).
    """
    check_capabilities(estimator, view_factory)
    check_output_shapes(estimator, view_factory)
    check_determinism(estimator, view_factory)
    check_no_lookahead(estimator, view_factory)
    check_engine_roundtrip(
        estimator,
        view_factory,
        splitter=splitter,
        min_train=min_train,
        forecast_kwargs=forecast_kwargs,
    )
