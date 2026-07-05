# Extending: write your own method

A method is any object with `fit(view) -> model`, where the fitted model advertises a capability and
implements its extractor. There is no base class to inherit and no run-loop to conform to: you write
a small estimator, self-certify it against the conformance suite, and it plugs into the same engine,
evaluators, and result schema as everything else.

This page builds a complete example — a volatility-scaled (inverse-volatility) weighting strategy —
then certifies it, registers it for discovery, and pins a reproduction target.

## The two protocols

{class}`~numeraire.core.protocols.Estimator`
: Anything with `fit(view) -> Model`.

{class}`~numeraire.core.protocols.Model`
: A fitted object with `capabilities() -> set[str]` plus the extractor each declared capability
  mandates — `to_weights` for {data}`~numeraire.core.capabilities.TO_WEIGHTS`, `forecast` for
  {data}`~numeraire.core.capabilities.TO_FORECAST`, `expected_returns` for
  {data}`~numeraire.core.capabilities.TO_PRICING`.

Both are `Protocol`s checked by duck typing, so your classes need not import or subclass anything
from the framework to conform — importing the capability constants is enough.

## A complete example

The strategy: at each decision date, weight each asset inversely to its recent return volatility,
normalised to sum to one. The critical detail is that the model **windows internally** — it reads
only `view.window(t)` for the row it emits at `t` — which is what makes it point-in-time safe.

```python
import numpy as np
import pandas as pd

from numeraire.core import capabilities
from numeraire.core.data import TimeSeriesView
from numeraire.core.protocols import DataView


class _InverseVolModel:
    """Fitted model: weights inversely proportional to trailing volatility."""

    def __init__(self, lookback: int) -> None:
        self._lookback = lookback

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: DataView) -> pd.DataFrame:
        if not isinstance(view, TimeSeriesView):
            raise TypeError("InverseVol runs on a TimeSeriesView")
        rows, idx = [], []
        for t in view.calendar:
            hist = view.window(t).returns_frame().to_numpy(dtype=np.float64)  # info known at t
            if len(hist) < self._lookback:
                continue                                                       # warm-up
            vol = hist[-self._lookback:].std(axis=0)
            inv = np.where(vol > 0.0, 1.0 / vol, 0.0)
            if inv.sum() == 0.0:
                continue
            rows.append(inv / inv.sum())
            idx.append(t)
        if not rows:
            return pd.DataFrame(columns=view.assets)
        return pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(idx), columns=view.assets)


class InverseVol:
    """Inverse-volatility weighting estimator (a ``to_weights`` method)."""

    def __init__(self, *, lookback: int = 12) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback

    def fit(self, view: DataView) -> _InverseVolModel:
        if not isinstance(view, TimeSeriesView):
            raise TypeError("InverseVol runs on a TimeSeriesView")
        return _InverseVolModel(self.lookback)
```

That is the whole method. It now runs through {func}`~numeraire.core.engine.backtest` exactly
like a bundled baseline:

```python
from numeraire import SharpeEvaluator, TimeSeriesView, WalkForwardSplitter, backtest

out = backtest(InverseVol(lookback=6), view, WalkForwardSplitter(min_train=40, test_size=8),
               method="inverse_vol")
SharpeEvaluator().evaluate(out)
```

## Self-certify with `check_estimator`

Before trusting a method's numbers, run it through {func}`numeraire.testing.check_estimator`. It is
the framework's analogue of scikit-learn's estimator checks: plain functions that raise
`AssertionError` on the first violation. You supply the estimator and a **deterministic**
`view_factory` — a zero-argument callable that returns an equivalent synthetic view each call.

```python
from numeraire.testing import check_estimator


def make_view() -> TimeSeriesView:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-31", periods=80, freq="ME")
    r = pd.DataFrame(rng.normal(0.01, 0.05, size=(80, 3)), index=dates, columns=["X", "Y", "Z"])
    return TimeSeriesView(r)


check_estimator(InverseVol(lookback=6), make_view, min_train=40)   # raises on the first failure
```

What each check catches:

{func}`~numeraire.testing.check_capabilities`
: `fit` returns a model whose `capabilities()` intersect the core set, and every *crystallised*
  capability it declares actually exposes its extractor method. Catches a model that advertises
  `to_weights` but never implemented it.

{func}`~numeraire.testing.check_output_shapes`
: Weights columns are a subset of `view.assets` and the index a subset of `view.calendar` (or a
  `[date, asset]` MultiIndex for a panel); a forecast is a `pd.Series` indexed by the assets; a
  pricing surface is a `(date × asset)` frame. Catches transposed frames and stray asset labels.

{func}`~numeraire.testing.check_determinism`
: The same estimator and the same view produce identical output twice. Catches an un-seeded RNG or
  other hidden state.

{func}`~numeraire.testing.check_no_lookahead`
: The property test. The model is handed a view spanning data *after* a cut date `t` and must window
  internally, so its rows at dates `≤ t` must be **invariant to mutating data strictly after `t`**.
  The suite builds a future-perturbed twin of the view and compares; a model that peeks past a
  prediction date fails here. Our `InverseVol` passes because every row at `t` reads only
  `view.window(t)`.

  This probe runs for `to_weights` and `expected_returns`, which both hand the model a multi-date
  view and rely on it to window. It is **not** run for `to_forecast`, and deliberately so: the
  forecast-origin engine only ever passes `forecast()` a prefix-truncated `view.window(origin)`, so
  a forecast at an origin is *structurally* incapable of seeing later data — a perturbation probe
  could never fail. A forecast leak instead surfaces as a disagreement between the engine path and a
  vectorised full-sample recomputation, which is where forecasting methods should assert their
  point-in-time safety.

{func}`~numeraire.testing.check_engine_roundtrip`
: The estimator runs through its matching walk-forward driver and an evaluator emits rows that
  validate against the result schema. Catches integration breaks the isolated checks miss.

You can also call the individual checks directly during development; `check_estimator` simply runs
them in order.

## Register for discovery

Bundled methods and external packages advertise themselves through the `numeraire.methods`
entry-point group. Declaring an entry point makes your estimator discoverable without any edit to
core — the same mechanism the bundled baselines dogfood. In your package's `pyproject.toml`:

```toml
[project.entry-points."numeraire.methods"]
inverse_vol = "yourpackage.inverse_vol:InverseVol"
InverseVol = "yourpackage.inverse_vol:InverseVol"
```

(The registry accepts both `snake_case` and `CamelCase` aliases, as the baselines do.) A method
packaged this way — pinning `numeraire`, registering via the entry point, shipping a
`check_estimator` conformance test — is a first-class peer of any built-in method.

## Pin a reproduction target

When your method reproduces a published result, record the target as a first-class, tolerance-banded
record with {class}`numeraire.reference.ReferenceResult`. A reference pins an exact paper, venue, and
table to an `expected` metric dict plus a per-metric `tolerance` **band** — never bit-equality,
because data-vintage revisions move the last decimals. `check` enforces the band and rejects a
non-finite computed value (guarding against an all-NaN false green).

```python
from numeraire.reference import PUBLIC, ReferenceResult, register_reference

REF = register_reference(ReferenceResult(
    name="inverse_vol_demo",
    paper="Author (2026)",
    venue="Journal",
    year=2026,
    table="Table 1",
    expected={"sharpe": 0.85},
    tolerance={"sharpe": 0.10},   # a band, not an exact match
    tier=PUBLIC,
))

REF.check({"sharpe": 0.87})       # passes: within the band
```

### The three data-access tiers

Each reference carries a **data-access tier**. The tier is a statement about *what data is required*,
never about importance or rank — a reproduction that needs licensed data is a first-class citizen.

{data}`~numeraire.reference.PUBLIC`
: Public, redistributable, or synthetic data — the case runs unconditionally, including in CI.

{data}`~numeraire.reference.CREDENTIALED`
: Data programmatically fetchable with the user's *own* subscription credentials; the case self-skips
  when those credentials are absent.

{data}`~numeraire.reference.RESTRICTED`
: Data anyone may obtain but that is non-redistributable, so it needs a self-obtained local copy; the
  case self-skips when that copy is absent.

An optional `available` predicate plus the tier let continuous integration stay green on public data
while the *same* case runs verbatim wherever the private data is present — one code path, no forked
assertions. {func}`~numeraire.reference.reference_params` turns the whole registry into
`pytest.param` entries (unavailable tiers marked skip) so one parametrised test drives them all.
