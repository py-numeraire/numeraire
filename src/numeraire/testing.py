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
  declares has its method (``to_weights`` / ``forecast`` / ``expected_returns``).
- :func:`check_output_shapes` — finite, uniquely labelled target weights with columns ⊆
  ``view.assets`` and index ⊆ ``view.calendar`` (wide), or a unique ``[date, asset]`` MultiIndex
  (panel); a forecast is a ``pd.Series`` indexed by ``view.assets``; pricing
  ``expected_returns`` is a ``(date x asset)`` frame with columns ⊆ ``view.assets`` and index ⊆
  ``view.calendar``.
- :func:`check_determinism` — same estimator + same view ⇒ identical output, twice.
- :func:`check_fit_independence` — an estimator's output on a view is invariant to an earlier fit
  on different data: fit a prefix, fit the full view (the contaminant), refit a freshly rebuilt
  content-equal prefix, and require the two prefix outputs to be bit-identical. A warm-start /
  cached-statistic leak fails here (including one keyed on view identity).
- :func:`check_fold_isolation` — the engine fits an isolated ``deepcopy`` of the estimator per fold,
  so the matching walk-forward driver produces bit-identical output under ``n_jobs=1`` and
  ``n_jobs=4`` (and on a fresh serial rerun). Where ``check_fit_independence`` probes the
  estimator's own fit purity, this probes the engine's fold isolation holds for a stateful one.
- :func:`check_no_lookahead` — the property test **for ``to_weights`` and ``expected_returns``**.
  Both hand the model a multi-date view and require it to window internally, so its rows up to ``t``
  must be **invariant to mutating data strictly after ``t``**; a model that peeks past a prediction
  date fails here (one leak channel, shared by a weight stream and a priced cross-section).
  ``to_forecast`` is deliberately **not** probed: the forecast-origin engine only ever hands
  ``forecast()`` a **prefix-truncated** ``view.window(origin)``, so a forecast at origin ``d`` is
  structurally incapable of seeing data after ``d`` — perturbing the future can't change it, so a
  probe here can never fail (it would be dead code). That PIT guarantee is exercised where a leak
  actually surfaces: the zoo's mandatory **engine ≡ vectorized** equality test, where a
  contemporaneous / off-by-one forecast leak makes the hand-written full-sample path disagree with
  the prefix-truncated engine path (see ``tests/test_testing.py`` for the mechanism).
- :func:`check_engine_roundtrip` — the estimator runs through the matching walk-forward driver
  without error and an evaluator emits rows conforming to the result schema.

A pricing method (``to_pricing``, e.g. an IPCA adapter) now exercises the full suite: its
``expected_returns`` surface is crystallized, so the shape, determinism, no-look-ahead, and
engine-round-trip checks all apply to it exactly as they do to a weights method.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from typing import Any, cast

import numpy as np
import pandas as pd

from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView, _adopt_provenance
from numeraire.core.engine import (
    backtest_forecast,
    backtest_panel,
    backtest_pricing,
    backtest_weights,
)
from numeraire.core.evaluators import (
    CrossSectionalR2Evaluator,
    OutOfSampleR2Evaluator,
    SharpeEvaluator,
)
from numeraire.core.schema import validate_result
from numeraire.core.splitter import WalkForwardSplitter

__all__ = [
    "check_capabilities",
    "check_determinism",
    "check_engine_roundtrip",
    "check_estimator",
    "check_fit_independence",
    "check_fold_isolation",
    "check_no_lookahead",
    "check_output_shapes",
]

ViewFactory = Callable[[], Any]

# The capabilities whose method surface has crystallized and the method each mandates. All three
# dispatchable capabilities are now crystallized: ``to_pricing`` mandates ``expected_returns``
# (SupportsPricing), joining ``to_weights`` and ``to_forecast`` (Supports Weights / Forecast).
_CRYSTALLIZED: dict[str, str] = {
    capabilities.TO_WEIGHTS: "to_weights",
    capabilities.TO_FORECAST: "forecast",
    capabilities.TO_PRICING: "expected_returns",
}
_CORE_CAPS = frozenset({capabilities.TO_WEIGHTS, capabilities.TO_FORECAST, capabilities.TO_PRICING})


# --------------------------------------------------------------------------- internals


class _ReplayFolds:
    """Re-iterable splitter over folds materialized once from a caller-supplied splitter.

    ``check_fold_isolation`` runs its driver three times; materializing the caller's folds once and
    replaying them keeps a one-shot ``split`` iterator from silently emptying the later runs.
    """

    def __init__(self, folds: list[tuple[Any, Any]]) -> None:
        self._folds = folds

    def split(self, view: Any) -> Iterator[tuple[Any, Any]]:
        _ = view
        return iter(self._folds)


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
    return cast("set[str]", caps)


def _extractable(caps: set[str]) -> set[str]:
    """The declared capabilities the suite can materialize into outputs (weights / forecasts)."""
    return caps & _CRYSTALLIZED.keys()


def _origin_index(view: Any, frac: float = 0.6) -> int:
    """A calendar position with ample history on either side (for windowed probes/splits)."""
    n = len(view.calendar)
    return min(n - 2, max(1, round(frac * (n - 1))))


def _weights(model: Any, view: Any) -> pd.DataFrame | pd.Series[Any]:
    w = model.to_weights(view)
    assert isinstance(w, pd.DataFrame | pd.Series), "to_weights must return a DataFrame or Series"
    return cast("pd.DataFrame | pd.Series[Any]", w)


def _expected_returns(model: Any, view: Any) -> pd.DataFrame:
    p = model.expected_returns(view)
    assert isinstance(p, pd.DataFrame), "expected_returns must return a (date x asset) DataFrame"
    return p


def _restrict_weights(w: pd.DataFrame | pd.Series, t: pd.Timestamp) -> pd.DataFrame | pd.Series:
    """Rows dated ``<= t`` (wide index or ``(date, asset)`` MultiIndex)."""
    if isinstance(w, pd.Series):
        dates = w.index.get_level_values("date")
        return w[dates <= t]
    return w.loc[w.index <= t]


def _assert_finite_present(arr: Any, msg: str) -> None:
    """Guard the ``equal_nan=True`` comparisons: an all-non-finite pair would pass as 'equal'."""
    assert arr.size > 0 and bool(np.isfinite(arr).any()), (
        f"{msg}: output is entirely non-finite (all NaN/inf) — cannot certify (false-green guard)"
    )


def _assert_weights_equal_on_common(
    a: pd.DataFrame | pd.Series, b: pd.DataFrame | pd.Series, msg: str
) -> None:
    idx = a.index.intersection(b.index)
    assert len(idx) > 0, f"{msg}: no overlapping prediction dates to compare"
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        arr_a, arr_b = a.loc[idx].to_numpy(np.float64), b.loc[idx].to_numpy(np.float64)
    else:
        assert isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)
        cols = a.columns.intersection(b.columns)
        arr_a = a.loc[idx, cols].to_numpy(np.float64)
        arr_b = b.loc[idx, cols].to_numpy(np.float64)
    _assert_finite_present(arr_a, msg)
    np.testing.assert_allclose(arr_a, arr_b, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=msg)


def _assert_bit_identical(
    a: pd.DataFrame | pd.Series, b: pd.DataFrame | pd.Series, msg: str
) -> None:
    """Exact comparison for the fit-independence contract: labels equal, values bit-identical.

    Unlike the tolerance-based common-label comparison used by the determinism check, this demands
    the second output carry **exactly** the first's index (and columns, for a frame) and exactly
    equal values (``NaN`` matching ``NaN``): a fit that is truly independent of estimator history
    recomputes the identical result, so any drift — a shifted label set or a last-bit value change —
    is evidence of state carried across fits.
    """
    assert type(a) is type(b), (
        f"{msg}: output type changed ({type(a).__name__} -> {type(b).__name__})"
    )
    assert a.index.equals(b.index), f"{msg}: output index labels changed"
    if isinstance(a, pd.DataFrame):
        assert isinstance(b, pd.DataFrame)
        assert a.columns.equals(b.columns), f"{msg}: output column labels changed"
    arr_a = a.to_numpy(np.float64)
    arr_b = b.to_numpy(np.float64)
    _assert_finite_present(arr_a, msg)
    assert np.array_equal(arr_a, arr_b, equal_nan=True), msg


def _perturb_after(view: Any, t: pd.Timestamp) -> Any:
    """A twin of ``view`` identical on data ``<= t`` but randomly perturbed strictly after ``t``.

    Rebuilt from the view's own ejected frames, so an estimator whose output at some date ``d <= t``
    depends on data after ``t`` (a look-ahead leak) produces different outputs on the two twins.
    Handles the two concrete core view types. The ejected returns are already converted to simple,
    so the twin is built with the default ``return_type`` and the source's ingestion provenance is
    carried over unchanged — an estimator reading the public ``provenance`` property sees identical
    values on both twins (no false leak signal, no double conversion).
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
            return _adopt_provenance(TimeSeriesView(ret, feat, horizon=view.horizon), view)
        return _adopt_provenance(TimeSeriesView(ret, horizon=view.horizon), view)
    if isinstance(view, CrossSectionView):
        pf = view.panel_frame().reset_index()
        mask = pf["date"] > t
        n_after = int(mask.to_numpy().sum())
        for col in [*view.char_names, "ret"]:
            pf.loc[mask, col] = pf.loc[mask, col].to_numpy() + rng.normal(0.0, 0.1, n_after)
        return _adopt_provenance(
            CrossSectionView(pf, chars=view.char_names, horizon=view.horizon), view
        )
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
        if isinstance(view, CrossSectionView):
            if isinstance(w, pd.DataFrame):
                assert w.shape[1] == 1, "panel weights must be a Series or one-column DataFrame"
                panel_weights = w.iloc[:, 0]
            else:
                panel_weights = w
            assert isinstance(panel_weights.index, pd.MultiIndex), "panel weights need a MultiIndex"
            assert list(panel_weights.index.names) == ["date", "asset"], (
                "panel weights MultiIndex must be named [date, asset]"
            )
            assert panel_weights.index.is_unique, "panel weights need unique (date, asset) keys"
            assert set(panel_weights.index.get_level_values("date")) <= set(view.calendar), (
                "panel weight dates must be ⊆ view.calendar"
            )
            assert set(map(str, panel_weights.index.get_level_values("asset"))) <= assets, (
                "panel weight assets must be ⊆ view.assets"
            )
            for date, cross in panel_weights.groupby(level="date", sort=False):
                current = set(view.universe(date))
                emitted = set(map(str, cross.index.get_level_values("asset")))
                assert emitted <= current, (
                    f"panel weights at {date} contain asset(s) outside that date's universe: "
                    f"{sorted(emitted - current)}"
                )
            assert bool(np.isfinite(panel_weights.to_numpy(dtype=np.float64)).all()), (
                "panel target weights must all be finite"
            )
        else:
            assert isinstance(w, pd.DataFrame), (
                "time-series weights must be a date x asset DataFrame"
            )
            assert set(map(str, w.columns)) <= assets, "weights columns must be ⊆ view.assets"
            # Duplicate column labels make the engine's label-based realized-return alignment
            # (``reindex(columns=view.assets)``) ambiguous — reject them up front.
            assert w.columns.is_unique, (
                "weights columns must be unique labels (duplicates break realized-return alignment)"
            )
            assert w.index.is_unique, "weights index must be unique"
            assert set(w.index) <= set(view.calendar), "weights index must be ⊆ view.calendar"
            assert bool(np.isfinite(w.to_numpy(dtype=np.float64)).all()), (
                "target weights must all be finite"
            )
    if capabilities.TO_FORECAST in caps:
        d = view.calendar[_origin_index(view)]
        f = model.forecast(view.window(d))
        assert isinstance(f, pd.Series), "forecast must return a pd.Series"
        assert f.index.is_unique, (
            "forecast index must be unique labels (duplicates break realized-return alignment)"
        )
        assert [str(i) for i in f.index] == view.assets, "forecast index must equal view.assets"
    if capabilities.TO_PRICING in caps:
        p = _expected_returns(model, view)
        assert set(map(str, p.columns)) <= assets, "expected_returns columns must be ⊆ view.assets"
        # Uniqueness must hold on the str-normalized labels the engine pools on: an int column and
        # its str twin are distinct raw labels but collapse to one asset after normalization.
        assert pd.Index([str(c) for c in p.columns]).is_unique, (
            "expected_returns columns must be unique labels after str normalization "
            "(an int label and its str twin would collide)"
        )
        # Mirror the driver's containment guard exactly (a violation the shape check accepted
        # would only resurface later as an engine error): DatetimeIndex + unique dates, with the
        # engine's zero-row "prices nothing" convention (a plain empty index) exempted.
        if len(p.index) > 0:
            assert isinstance(p.index, pd.DatetimeIndex), (
                "expected_returns index must be a DatetimeIndex (an object index of timestamps "
                "is rejected by the pricing drivers)"
            )
            assert p.index.is_unique, (
                "expected_returns index must be unique dates (duplicates break realized alignment)"
            )
            assert set(p.index) <= set(view.calendar), (
                "expected_returns index must be ⊆ view.calendar"
            )


def check_determinism(estimator: Any, view_factory: ViewFactory) -> None:
    """Same estimator + same (deterministic) view ⇒ bit-identical extractable output, twice."""
    ext = _extractable(_caps(_fit(estimator, view_factory())))
    if not ext:
        return  # capability-only method (e.g. to_density / to_surface): no crystallized surface
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
        f1 = m1.forecast(v.window(d)).to_numpy(np.float64)
        f2 = m2.forecast(v.window(d)).to_numpy(np.float64)
        _assert_finite_present(f1, "forecast determinism")
        np.testing.assert_allclose(
            f1,
            f2,
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
            err_msg="forecast is non-deterministic across identical fits",
        )
    if capabilities.TO_PRICING in ext:
        _assert_weights_equal_on_common(
            _expected_returns(m1, view_factory()),
            _expected_returns(m2, view_factory()),
            "expected_returns is non-deterministic across identical fits",
        )


def check_fit_independence(estimator: Any, view_factory: ViewFactory) -> None:
    """An estimator's fit output on a view is independent of any earlier fit on different data.

    A stateful estimator that carries information across fits — a warm start seeded from the last
    fit, a cached statistic blended into the next one — would score differently inside a
    walk-forward loop depending on the order the folds were fitted in, a subtle look-ahead-flavoured
    contamination that ordinary shape / determinism checks miss. This probes it with a single
    instance (we cannot clone an arbitrary estimator): fit a prefix sub-view and snapshot its
    extractable output, perform a *contaminating* fit on the full view, refit a **freshly rebuilt**
    but content-equal prefix sub-view (a different object, so a cache keyed on view identity cannot
    fake independence), and require the two sub-view outputs to be **bit-identical** — labels equal
    and values exactly equal, with ``NaN`` matching ``NaN``. A model that leaks state across fits
    produces a different second output and fails here.
    """
    full = view_factory()
    cal = full.calendar
    assert len(cal) >= 4, "check_fit_independence needs a view with >= 4 calendar dates"
    t = cal[_origin_index(full, 0.65)]  # a strict prefix (~65% of the calendar)
    sub = full.window(t)
    # The FIRST fit is the measured sub-view fit, on a fresh estimator — reading capabilities from
    # a preceding full-view fit would prime any across-fit state to the same value the contaminating
    # fit sets, hiding a "cache the last fit" leak. Snapshot every extractable output of this first
    # fit before any other fit runs.
    m1 = _fit(estimator, sub)
    ext = _extractable(_caps(m1))
    if not ext:
        return  # capability-only method: no crystallized surface to compare
    d = sub.calendar[_origin_index(sub)]
    w1 = _weights(m1, sub).copy() if capabilities.TO_WEIGHTS in ext else None
    f1 = m1.forecast(sub.window(d)).copy() if capabilities.TO_FORECAST in ext else None
    p1 = _expected_returns(m1, sub).copy() if capabilities.TO_PRICING in ext else None
    _fit(estimator, view_factory())  # contaminating fit on the different, longer full view
    # Refit a content-equal but freshly built sub-view: an estimator caching by view identity
    # (id(view)) would trivially reproduce the first output on the SAME object and mask the leak.
    sub2 = view_factory().window(t)
    m2 = _fit(estimator, sub2)
    msg = (
        "fit output on a sub-view changed after an intervening fit on different data — the "
        "estimator carries state across fits (warm start / cached statistic)"
    )
    if w1 is not None:
        _assert_bit_identical(w1, _weights(m2, sub2), msg)
    if f1 is not None:
        _assert_bit_identical(f1, m2.forecast(sub2.window(d)), msg)
    if p1 is not None:
        _assert_bit_identical(p1, _expected_returns(m2, sub2), msg)


def check_no_lookahead(estimator: Any, view_factory: ViewFactory) -> None:
    """``to_weights`` / ``expected_returns`` up to ``t`` are invariant to mutating data after ``t``.

    Both multi-date extraction surfaces (a weight stream and a priced cross-section) are handed a
    view spanning data after ``t`` and must window internally, so a leak — using post-``t`` data for
    a ``<= t`` row — shows up here as a changed row when the future is perturbed. ``to_forecast`` is
    **not** probed: the engine only ever passes ``forecast()`` a prefix-truncated
    ``view.window(origin)``, so a forecast at origin ``d <= t`` cannot see post-``t`` data and a
    perturbation probe could never fail — a forecast leak instead surfaces in the zoo's
    engine ≡ vectorized equality test (module docstring).
    """
    caps = _caps(_fit(estimator, view_factory()))
    probes = caps & {capabilities.TO_WEIGHTS, capabilities.TO_PRICING}
    if not probes:
        return  # forecast-only: forecast PIT is structural (see docstring)
    view = view_factory()
    cal = view.calendar
    assert len(cal) >= 8, "no-look-ahead check needs a view with >= 8 calendar dates"
    t = cal[_origin_index(view)]
    twin = _perturb_after(view, t)
    # Extract over the FULL view (which contains data after t); a PIT-respecting model windows
    # internally, so its rows at dates <= t must ignore the perturbed tail.
    model_a = _fit(estimator, view.window(t))
    model_b = _fit(estimator, twin.window(t))
    if capabilities.TO_WEIGHTS in probes:
        _assert_weights_equal_on_common(
            _restrict_weights(_weights(model_a, view), t),
            _restrict_weights(_weights(model_b, twin), t),
            "to_weights at dates <= t changed when only post-t data was mutated (look-ahead)",
        )
    if capabilities.TO_PRICING in probes:
        pa = _expected_returns(model_a, view)
        pb = _expected_returns(model_b, twin)
        _assert_weights_equal_on_common(
            cast("pd.DataFrame", pa.loc[pa.index <= t]),
            cast("pd.DataFrame", pb.loc[pb.index <= t]),
            "expected_returns at dates <= t changed when only post-t data was mutated (look-ahead)",
        )


def check_fold_isolation(
    estimator: Any,
    view_factory: ViewFactory,
    *,
    splitter: Any | None = None,
    min_train: int | None = None,
    forecast_kwargs: dict[str, Any] | None = None,
) -> None:
    """Serial and parallel walk-forward runs are bit-identical (the engine isolates each fold).

    Every driver fits an isolated ``copy.deepcopy`` of the estimator per fold, so a fold's result
    cannot depend on which other folds ran first or on the ``n_jobs`` worker schedule — even for a
    stateful estimator (a warm start seeded from the last fit, a cached statistic). This runs the
    estimator's matching driver serially (``n_jobs=1``) and in parallel (``n_jobs=4``) and requires
    the OOS panels to be bit-identical, plus a third fresh serial run identical to the first. The
    capability probe itself fits a ``deepcopy``, so the supplied estimator reaches every run in its
    pristine pre-fit state. An engine that shared one mutable estimator across parallel folds makes
    the ``n_jobs=4`` run diverge here; per-fold isolation makes the property hold. A caller-supplied
    splitter must yield **at least two folds** — with a single fold the parallel run never exercises
    the thread pool and the check would certify nothing. Complements
    :func:`check_fit_independence` — that probes the estimator's own fit purity; this probes that
    the *engine* actually isolates each fold, so a stateful estimator stays reproducible regardless
    of ``n_jobs``. (An estimator whose fit is nondeterministic also fails here; run
    :func:`check_determinism` to tell the two causes apart.)
    """
    view = view_factory()
    # Probe capabilities on a deepcopy: fitting the supplied instance here would advance any
    # across-fit state before the measured runs, breaking the pristine-estimator premise.
    caps = _caps(_fit(copy.deepcopy(estimator), view))
    n = len(view.calendar)
    warmup = min_train if min_train is not None else max(2, n // 2)

    def _multifold() -> Any:
        # Several folds so n_jobs=4 exercises real thread-level parallelism (a single fold never
        # dispatches to the pool, hiding a scheduling-dependent divergence). A caller-supplied
        # splitter is materialized once — so the requirement is checked and a one-shot ``split``
        # iterator survives the three runs — and replayed through a re-iterable stand-in.
        if splitter is not None:
            folds = list(splitter.split(view))
            if len(folds) < 2:
                raise ValueError(
                    f"check_fold_isolation needs a splitter yielding at least two folds (got "
                    f"{len(folds)}): with a single fold the parallel run never exercises the "
                    "thread pool, so serial-vs-parallel identity would be vacuous — supply a "
                    "splitter producing two or more folds"
                )
            return _ReplayFolds(folds)
        rest = max(1, n - warmup)
        return WalkForwardSplitter(min_train=warmup, test_size=max(1, rest // 3), expanding=True)

    def _compare(primary: Any, others: list[tuple[Any, str]], attrs: tuple[str, ...]) -> None:
        for other, label in others:
            for attr in attrs:
                _assert_bit_identical(
                    getattr(primary, attr),
                    getattr(other, attr),
                    f"walk-forward {attr} under {label} differs from the serial run — either a "
                    "fold's result depends on execution order (fit-relevant mutable state shared "
                    "across fold fits) or the estimator's fit is nondeterministic (an unseeded "
                    "RNG fails here too; run check_determinism to distinguish)",
                )

    if capabilities.TO_WEIGHTS in caps and isinstance(view, TimeSeriesView):
        sp = _multifold()
        runs = [
            backtest_weights(estimator, view, sp, method="conformance", n_jobs=j) for j in (1, 4, 1)
        ]
        _compare(
            runs[0],
            [(runs[1], "n_jobs=4"), (runs[2], "a fresh serial run")],
            ("weights", "realized"),
        )
    elif capabilities.TO_WEIGHTS in caps and isinstance(view, CrossSectionView):
        sp = _multifold()
        runs = [
            backtest_panel(estimator, view, sp, method="conformance", n_jobs=j) for j in (1, 4, 1)
        ]
        _compare(
            runs[0],
            [(runs[1], "n_jobs=4"), (runs[2], "a fresh serial run")],
            ("weights", "realized"),
        )
    elif capabilities.TO_FORECAST in caps and isinstance(view, TimeSeriesView):
        runs = [
            backtest_forecast(
                estimator,
                view,
                min_train=warmup,
                method="conformance",
                n_jobs=j,
                **(forecast_kwargs or {}),
            )
            for j in (1, 4, 1)
        ]
        _compare(
            runs[0],
            [(runs[1], "n_jobs=4"), (runs[2], "a fresh serial run")],
            ("forecasts", "realized", "benchmark"),
        )
    elif capabilities.TO_PRICING in caps and isinstance(view, TimeSeriesView | CrossSectionView):
        sp = _multifold()
        runs = [
            backtest_pricing(estimator, view, sp, method="conformance", n_jobs=j) for j in (1, 4, 1)
        ]
        _compare(
            runs[0],
            [(runs[1], "n_jobs=4"), (runs[2], "a fresh serial run")],
            ("predicted", "realized"),
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
        out = backtest_weights(estimator, view, sp, method="conformance")
        assert not out.weights.empty, (
            "backtest_weights produced no weights (widen the fixture/splitter)"
        )
        rows = SharpeEvaluator().evaluate(out)
        validate_result(rows)
        assert len(rows) >= 1
    elif capabilities.TO_WEIGHTS in caps and isinstance(view, CrossSectionView):
        sp = splitter or WalkForwardSplitter(
            min_train=warmup, test_size=max(1, n - warmup), expanding=True
        )
        out = backtest_panel(estimator, view, sp, method="conformance")
        assert not out.weights.empty, (
            "backtest_panel produced no weights (widen the fixture/splitter)"
        )
        rows = SharpeEvaluator().evaluate(out)
        validate_result(rows)
        assert len(rows) >= 1
    elif capabilities.TO_FORECAST in caps and isinstance(view, TimeSeriesView):
        out = backtest_forecast(
            estimator, view, min_train=warmup, method="conformance", **(forecast_kwargs or {})
        )
        assert not out.forecasts.empty, "backtest_forecast produced no forecasts"
        rows = OutOfSampleR2Evaluator().evaluate(out)
        validate_result(rows)
        assert len(rows) >= 1
    elif capabilities.TO_PRICING in caps and isinstance(view, TimeSeriesView | CrossSectionView):
        sp = splitter or WalkForwardSplitter(
            min_train=warmup, test_size=max(1, n - warmup), expanding=True
        )
        out = backtest_pricing(estimator, view, sp, method="conformance")
        assert not out.predicted.empty, (
            "backtest_pricing produced no predictions (widen the fixture/splitter)"
        )
        rows = CrossSectionalR2Evaluator().evaluate(out)
        validate_result(rows)
        assert len(rows) >= 1


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
    # Fit-independence runs first, while the estimator is still fresh: it detects across-fit state
    # by comparing two fits of the same sub-view around a contaminating fit, so a prior check that
    # already fitted the estimator could prime that state and mask the leak.
    check_fit_independence(estimator, view_factory)
    # Right after fit-independence: the engine must isolate every fold so serial ≡ parallel even for
    # a stateful estimator. It runs the driver, so it follows the estimator-property checks above.
    check_fold_isolation(
        estimator,
        view_factory,
        splitter=splitter,
        min_train=min_train,
        forecast_kwargs=forecast_kwargs,
    )
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
