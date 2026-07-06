# Architecture

`numeraire` is deliberately small at the centre and open at the edges. This page is the mental
model: the data views, the capability layer, the walk-forward engine family, the evaluators and the
result schema, the portfolio simulator, the open registries, and the one dependency rule that keeps
the whole thing honest.

## Views: point-in-time by construction

A **view** is a point-in-time aligned window of the data. It exposes a `calendar` (the decision
dates) and a `window(end)` that restricts the data to information available up to `end`. Two
concrete views cover the two halves of empirical asset pricing.

{class}`~numeraire.core.data.TimeSeriesView`
: The market-timing / aggregate-predictor case: a `(date × asset)` returns block (one column for a
  single market series, several for a panel) plus zero or more time-series predictor blocks. Each
  predictor enters as its own {class}`~numeraire.core.data.FeatureBlock` with its own calendar and
  an availability lag, so heterogeneous macro sources — different frequencies, different publication
  lags — coexist. Vintaged (point-in-time revised) sources enter as a
  {class}`~numeraire.core.data.VintagedBlock`, which resolves the real-time edge so no future
  revision leaks in.

{class}`~numeraire.core.data.CrossSectionView`
: The cross-sectional case (Fama–MacBeth, characteristic sorts, panel machine learning), where the
  predictor `z_{i,t}` varies by both date **and** asset. It is built from a tidy long panel, the
  universe may enter and exit (ragged), and point-in-time windows are zero-copy prefix slices of the
  date-sorted panel. For tensor and neural methods it ejects a dense `(T × N × K)`
  {class}`~numeraire.core.data.PanelTensor` with an explicit presence mask.

### The time model is a contract

Two rules keep availability unambiguous, and the framework holds to both. **Inside the decision
calendar, horizon and lags are step arithmetic** on whatever calendar you supply: `horizon=1`, a
walk-forward window, or a block's own availability lag all count *positions*, never a unit like
"month". The framework never interprets a calendar unit, so daily, weekly, and monthly data are all
first-class. **At every source boundary, availability is a timestamp comparison:** a row or vintage
stamped `s` is usable at decision time `t` exactly when `s <= t` — visible on its stamped day and
not one moment before. Consequently, publication delays and release buffers belong in the data's
timestamps: stamp the true availability date (or shift a coarse label to a conservative release
date) at the data-preparation layer, rather than expecting the framework to add unit-based
arithmetic to compensate.

### The `(t, t+h]` pairing convention

The single convention every driver and evaluator obeys: features known **as of** `t` are paired
with the return **realised over** `(t, t+h]`. A feature dated `t` is never matched to a return that
overlaps `t` itself. The view owns this pairing — `features_asof(t)` and `target_asof(t, h)` — so a
method never indexes the returns array directly and a one-period contemporaneous overlap is
structurally impossible rather than a bug waiting to happen.

### Why point-in-time discipline matters

Empirical asset pricing is unusually exposed to **look-ahead bias**: because signals and returns
share a time axis, it is trivially easy to let a single period of future information seep into a
prediction, and even a one-period contemporaneous overlap can turn a genuinely negative
out-of-sample result into an apparently significant positive one. The consequence is a number that
cannot be earned in real time — the most expensive kind of research error. `numeraire` therefore
does not rely on author discipline to avoid leakage. The engine hands a model only a windowed view;
the view's `aligned` pairing purges any feature whose target is not yet realised; and the
conformance suite (see {doc}`extending`) ships a property test that perturbs the future and asserts
the past is unchanged. Look-ahead safety is a structural property of the framework, not a checklist.

## Capabilities: what a model can produce

The core is **representation-agnostic**. It never encodes a linear-factor (α / β / λ) structure, or
any other method-specific form, into its types. Instead a fitted model declares which
**capabilities** it supports, and the framework dispatches on them. The capability names are a flat,
open registry of string constants ({mod}`numeraire.core.capabilities`), not a closed enum —
extensions may add their own.

Three capabilities have crystallised into frozen method-level protocols:

{data}`~numeraire.core.capabilities.TO_WEIGHTS` — {class}`~numeraire.core.protocols.SupportsWeights`
: `to_weights(view) -> (date × asset)` portfolio or timing weights. Tangency, SDF, timing, and
  risk-based rules all live here.

{data}`~numeraire.core.capabilities.TO_FORECAST` — {class}`~numeraire.core.protocols.SupportsForecast`
: `forecast(view) -> pd.Series` — a per-asset prediction of the return over the next horizon. The
  predictive-regression family.

{data}`~numeraire.core.capabilities.TO_PRICING` — {class}`~numeraire.core.protocols.SupportsPricing`
: `expected_returns(view) -> (date × asset)` — the cross-section of expected returns. Factor models,
  SDFs, and three-pass risk-premium estimators share this one surface; their bespoke accessors
  (loadings, latent factors, per-candidate premia) stay method-local.

A model is any object with a `capabilities()` set and whatever extractor methods those capabilities
mandate; an {class}`~numeraire.core.protocols.Estimator` is any object with `fit(view) -> Model`.
These are `Protocol`s, not base classes — a method conforms by duck typing, with nothing to inherit.

## The walk-forward engine

The engine is the most-reused, most-bug-prone, method-agnostic part of the framework, so it is kept
deliberately small and shared. For each `(train, test)` fold it fits the estimator on the train
view, asks the fitted model for its capability output on the test view, and computes realised
profit-and-loss **from the original full view** — never from anything the model returns.
{func}`~numeraire.core.engine.backtest` is the discoverable entry point: it reads the fitted model's
capability and the view type and dispatches to the right typed driver below (`in_sample=True`
selects the in-sample pricing path). One typed driver exists per capability, each returning a
frozen, provenance-stamped output container:

| Driver | Capability | Output |
| --- | --- | --- |
| {func}`~numeraire.core.engine.backtest_weights` | `to_weights` (time series) | {class}`~numeraire.core.engine.WeightsOutput` |
| {func}`~numeraire.core.engine.backtest_panel` | `to_weights` (ragged panel) | {class}`~numeraire.core.engine.PanelWeightsOutput` |
| {func}`~numeraire.core.engine.backtest_forecast` | `to_forecast` | {class}`~numeraire.core.engine.ForecastOutput` |
| {func}`~numeraire.core.engine.backtest_pricing` | `to_pricing` (out-of-sample) | {class}`~numeraire.core.engine.PricingOutput` |
| {func}`~numeraire.core.engine.backtest_pricing_in_sample` | `to_pricing` (explanatory) | {class}`~numeraire.core.engine.PricingOutput` |

Every output carries a `config_hash` (a stable hash of the preprocessing/method config, so
preprocessing is pinned as part of the method) and a `data_vintage` stamp, which flow into every
result row. The forecast driver additionally decouples the refit cadence from the prediction cadence
(`refit_every`) — annual refits with monthly predictions, for instance — with each prediction still
consuming its own up-to-date point-in-time window. All drivers accept `n_jobs` to fan the
independent folds over a thread pool; the mapping is order-preserving, so a parallel run is
identical to the serial one.

The `(train, test)` folds come from a **splitter**. The bundled
{class}`~numeraire.core.splitter.WalkForwardSplitter` yields expanding- or rolling-window folds and
supports an `embargo` gap on top of the automatic horizon purge; anything with a compatible `split`
method (including a wrapped scikit-learn splitter) works. {func}`~numeraire.core.splitter.validation_split`
carves a point-in-time `(fit, valid)` split *inside* a train fold for hyper-parameter tuning.

## Evaluators and the result schema

**Evaluators** turn an output container into rows of the standard tidy schema. They dispatch by
capability — each carries a `requires` set — so the metric always matches the object: a timing
strategy is scored by Sharpe, a forecast by out-of-sample R², a pricing model by cross-sectional R²
and average absolute alpha. The native evaluators (numpy/scipy only) cover the performance,
forecast-accuracy, and pricing families; two of them ({class}`~numeraire.core.evaluators.StrategyReturnEvaluator`,
{class}`~numeraire.core.evaluators.SquaredErrorDiffEvaluator`) emit one row **per date** for plotting
cumulative curves.

Every row conforms to {data}`~numeraire.core.schema.RESULT_COLUMNS` —
`run_id, method, date, metric, value, universe, capability, protocol, config_hash, data_vintage` —
and {func}`~numeraire.core.schema.validate_result` enforces their presence. The schema is the stable
contract between computation and everything downstream (plotting, aggregation, comparison), and its
stability is promised under semantic versioning. The `protocol` column is what keeps an explanatory
in-sample number distinguishable from an out-of-sample one at every point in the pipeline.

The lower-level statistical machinery the evaluators build on is available directly in
{mod}`numeraire.core.stats`: the Gibbons–Ross–Shanken joint zero-alpha test, the Clark–West
nested-forecast test, the Jobson–Korkie–Memmel paired-Sharpe test, HAC alpha regressions, the
Benjamini–Yekutieli / Holm / Bonferroni multiple-testing adjustments behind the factor-zoo `t > 3`
hurdle, and the certainty-equivalent / return-loss / performance-fee economic-value measures.

## The simulator

The evaluators score idealised weight streams. When trading frictions matter, the
{func}`~numeraire.core.simulate.simulate_weights` accounting simulator turns a stream of target
weights and asset returns into realised gross and net return series with per-rebalance turnover and
costs. Published papers disagree on turnover and cost conventions, so every convention here is an
explicit, named parameter — accounting mode (constant-mix target vs drifted holdings), turnover
definition, proportional cost, cash/risk-free treatment, missing-return policy, and target
normalisation — never an implicit default buried in the accounting. A
{class}`~numeraire.core.simulate.RebalanceSchedule` decouples the decision calendar from the data
frequency (month-end decisions over daily returns, say).

## Open registries

Extensibility runs through open registries rather than closed enumerations. Evaluators register in
the {mod}`evaluator registry <numeraire.core.registry>`; methods (including the bundled baselines)
register through the `numeraire.methods` **entry-point group**, so an external package is a
first-class peer discovered at install time without any edit to core; reproduction targets register
in the {mod}`reference registry <numeraire.reference>`. Adding a method, a metric, or a replication
target never requires touching the spine.

## The boundary rule

One rule holds the architecture together:

> `numeraire.core` is exactly the modules that depend on no specific method and that every method
> depends on. Dependency arrows point toward `core`; **`core` never imports a method, an adapter, or
> a reference library.**

```{eval-rst}
.. code-block:: text

    numeraire.baselines ─┐
    numeraire.adapters  ─┼──▶  numeraire.core   (spine: views, engine, evaluators, schema, ...)
    external methods    ─┘
```

The rule is enforced in continuous integration by `import-linter`, configured under
`[tool.importlinter]` in `pyproject.toml`. The lint rule *is* the architecture: if a change appears
to require breaking it, that is a signal the design is wrong, not that the rule should bend. A useful
operational test — code that would be rewritten to try a *different* algorithm does not belong in
core.

A small number of modules ({mod}`numeraire.testing`, {mod}`numeraire.reference`,
{mod}`numeraire.comparison`) live in `numeraire` proper rather than `numeraire.core`. They are core
*infrastructure* — they import only `numeraire.core` plus numpy/pandas, never a method — and are
exempt from the ban by construction, since they need to know the concrete view types to do their
job.
