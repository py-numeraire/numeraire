# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Versions are tag-driven (`hatch-vcs`).

## [Unreleased]

### Added

- `MissingReturnPolicy` and `WeightsOutput.scoring_weights()` /
  `PanelWeightsOutput.scoring_weights()` make incomplete-return scoring explicit and auditable.
  `"renormalize_legs"` preserves the original positive and negative target exposures separately;
  driver-produced output metadata records the policy, missing held observations, affected dates,
  and re-normalized dates.
- `assign_portfolio_bins` and `aggregate_assigned_portfolios` expose portfolio formation and
  holding-period aggregation as separate, testable steps. `SortAssignments` carries the frozen
  formation membership and its breakpoints; `sort_portfolios` remains the convenience wrapper.
- `numeraire.testing.check_fit_independence` — a conformance check that an estimator's output on a
  view is independent of any earlier fit on different data (fit a prefix, fit the full view, refit a
  freshly rebuilt content-equal prefix, require the two prefix outputs to be bit-identical). Catches
  warm-start / cached-statistic state that leaks across fits, including caches keyed on view
  identity. Added to the default `check_estimator` battery.
- Property-based no-look-ahead tests for the timestamp-`asof` availability layer (`VintagedBlock`
  and `CharBlock` vintaged mode), checked against an independent brute-force oracle over irregular
  calendars, long publication lags and intra-period stamps.
- Optional `n_obs` / `n_dropped` attrition columns (`schema.ATTRITION_COLUMNS`) on the tidy result
  rows of the benchmark-comparison evaluators (`OutOfSampleR2Evaluator`, `SquaredErrorDiffEvaluator`,
  `ClarkWestEvaluator`, `CrossSectionalR2Evaluator`, `AverageAbsAlphaEvaluator`). They record the
  size of the joint finite sample a metric was scored on and how many candidate observations the
  joint mask excluded, so selective missingness is auditable on the row itself. The columns are
  schema-additive: `validate_result` never requires them, but every non-null cell must be a finite,
  non-negative, integer-valued numeric (non-numeric cells are rejected rather than coerced away).
- `newey_west_lrv` accepts an optional `valid` observation mask: autocovariances then pair only
  observed positions exactly the lag apart on the original time axis, keeping HAC lags meaningful
  for a series with internal gaps. The default (no mask) is the previous dense behavior.
- `numeraire.testing.check_fold_isolation` — a conformance check that the engine isolates every
  fold: a stateful estimator's matching walk-forward driver produces bit-identical output under
  `n_jobs=1` and `n_jobs=4` (and on a fresh serial rerun). The capability probe itself fits a
  deepcopy so the supplied estimator reaches every run pristine, and a caller-supplied splitter must
  yield at least two folds (a single fold never dispatches to the thread pool, so serial-vs-parallel
  identity would be vacuous). Where `check_fit_independence` probes the estimator's own fit purity,
  this probes that the engine's per-fold isolation holds; a nondeterministic fit also fails it (the
  failure message says how to tell the causes apart). Runs in the default `check_estimator` battery
  immediately after `check_fit_independence`.

### Changed

- **Breaking — benchmark-comparison evaluators fail closed above 50% missingness.** When the joint
  finite mask (model ∩ target ∩ benchmark) drops more than half of the candidate observations,
  `OutOfSampleR2Evaluator`, `SquaredErrorDiffEvaluator`, `ClarkWestEvaluator`, and the cross-sectional
  pricing evaluators now raise `ValueError` instead of scoring a rump sample. There is no warning
  tier — a majority-missing comparison is refused outright. On the pricing side a *candidate* is a
  cell where either the predicted or the realized value is finite: cells absent on both sides are
  structural (a ragged entering/exiting universe), count neither as observed nor as dropped, and
  cannot trip the threshold. An empty comparison output — no candidate observations at all, e.g.
  from a view too short to produce any evaluation window — also raises `ValueError` instead of
  crashing or returning an empty/NaN result. Below the threshold, scoring proceeds and the
  `n_obs` / `n_dropped` columns make the drop visible.

- **Breaking — weight backtests now fail closed on a missing held return.** `backtest_weights` and
  `backtest_panel` default to `missing_returns="error"`; callers must explicitly choose `"zero"` or
  `"renormalize_legs"` when a paper's convention requires it. The policy is included in
  `config_hash`, so weight-run hashes change even with an otherwise empty method config. The engine
  now removes only a mechanically identified horizon tail, not earlier rows or assets merely because
  their realized return is unavailable. `WeightsOutput.weights` and `PanelWeightsOutput.weights`
  always remain the model's target decisions, so missingness can no longer silently alter exposure,
  turnover, HHI, or weight plots. Non-finite target weights are rejected rather than treated as zero.
- `backtest()` now performs its capability-probe fit on the selected driver's **first train
  window** — the first fold's train view (walk-forward), the warm-up prefix (forecast), or the
  whole view (in-sample) — instead of always fitting the full sample. Fitting the full view ahead
  of a walk-forward run let a stateful estimator observe post-train data while its capabilities were
  being read, a silent look-ahead channel; the probe now stays within the same information set the
  driver's first fit uses. The user splitter's `split(view)` is consulted exactly once (its folds
  are materialized and replayed to the driver), so a splitter whose `split` returns a one-shot
  iterator loses no folds. Output is unchanged for stateless estimators.
- **Breaking — every backtest fit now runs on an isolated `copy.deepcopy` of the estimator.** All
  four drivers (`backtest_weights`, `backtest_forecast`, `backtest_panel`, `backtest_pricing`), the
  in-sample pricing path, `backtest()`'s capability-probe fit, and the `compare` comparison harness
  deep-copy the estimator before fitting — uniformly, serial *and* parallel; the engine never fits
  the caller's instance directly. For estimators honoring the isolation contract, a fold's result no
  longer depends on which other folds were fitted first or on the `n_jobs` thread schedule
  (previously the drivers fitted one shared instance, so a stateful estimator's serial folds chained
  state and its parallel folds raced). Output is unchanged for stateless estimators; an estimator
  that deliberately relied on cross-fold warm-start / cached state now sees each fold fitted from
  its pristine pre-fit state. The isolation contract: estimators must be **deepcopy-able** (a
  failing deepcopy raises a contextual `TypeError` naming the method, chaining the original error;
  an un-copyable resource such as a live DB handle belongs behind a factory that builds it at `fit`
  time) and must **not share fit-relevant mutable state across copies** — `copy.deepcopy` cannot
  sever class attributes, module globals, or containers a custom `__deepcopy__` aliases, so an
  estimator routing state through such channels defeats the isolation and can still observe or
  mutate the caller's instance. Copying a pre-fit estimator is cheap next to the fit.
- **Breaking — point-in-time availability is now a real-timestamp comparison.** `VintagedBlock` and
  `CharBlock` previously decided what was "known" by comparing calendar *month ordinals*, so a row
  or release stamped later in the same month counted as already available — a silent intra-month
  look-ahead whenever the data or the decision calendar was finer than monthly (daily panels,
  month-end-stamped rows read on a daily calendar, mid-month releases). Availability is now the
  unit-free rule `stamp <= t`: a reference date, vintage, or release is visible on its stamped day
  and not before. Behavior only changes for data whose stamps are misaligned within a period, and
  always in the safe direction (a value becomes older or `NaN`, never newer). For the
  timestamp-comparison change in isolation, month-end-stamped monthly data is unaffected (the
  separate availability shift from removing the default `lag` is described in the next bullet).
  Missing (`NaT`) availability stamps and tz-aware stamps are now rejected at construction rather
  than silently mis-scaled (a `NaT` stamp used to read as "available since the beginning of time",
  a tz-aware stamp shifted the boundary by its UTC offset). Duplicate `(ref_date, vintage)` /
  `(asset, ref_date, vintage)` keys, whose real-time edge was order-dependent, now raise as well.
- **Breaking — `VintagedBlock` no longer takes a `lag` argument.** The old `lag` (whole months,
  default 1) was a coarse availability buffer that cannot be expressed under timestamp resolution.
  Bake any publication delay into the `vintage` column at the data end instead — e.g.
  `table.assign(vintage=table["vintage"] + pd.DateOffset(months=1))` before constructing the block.
  Consumers that relied on the default `lag=1` will see availability move up to ~one period earlier
  (the old default was deliberately over-conservative); this is correct real-time behavior, but a
  golden number fed by a vintaged source may shift.
- **Breaking — `CharBlock` vintaged mode rejects a non-zero `lag`.** In vintaged mode availability
  is the vintage timestamp, so a row-step lag is meaningless; passing `lag != 0` together with
  `vintage_col` now raises `ValueError`. Lagged mode is unchanged: availability is the row's own
  date and `lag` still steps back that many rows in the asset's own series.

### Fixed

- Benchmark-comparison evaluators no longer let non-finite predictions manufacture apparent skill.
  `OutOfSampleR2Evaluator`, `SquaredErrorDiffEvaluator`, and `ClarkWestEvaluator` previously scored
  the model and its benchmark with separate `nansum` denominators, and the pricing evaluators
  averaged predicted and realized returns over separate `nanmean` samples. A forecast that was
  present on some observations and missing on others could therefore be scored against a smaller
  error base than the fully-observed benchmark — a selectively-missing model reporting false skill
  (e.g. roughly +50% out-of-sample R² against a zero benchmark). Every such metric now builds one
  joint finite mask across model, target, and benchmark and scores all terms on the same
  observations; a per-origin statistic drops (rather than zero-fills) origins with no jointly-finite
  cell. The Clark-West Newey-West variance is computed on the original origin axis — lag-l
  autocovariances pair only observed origins exactly l periods apart — so an internal gap in the
  observed origins does not make observations several periods apart look adjacent.
- `ReferenceResult` now rejects a non-finite `expected` value and a non-finite or negative
  `tolerance` at construction, and snapshots `expected` / `tolerance` into read-only copies so
  mutating the caller's dicts after construction cannot bypass that validation. A `NaN`/`Inf`
  expected value or an infinite band would previously auto-pass its own `check`, letting a vacuous
  "verified" reproduction be registered; the existing exact-match guard (a zero band on an integer
  target stays legal) is unchanged.
- The forecast and pricing drivers now contain their model's output to the fold, closing the last
  two gaps left after the weights/panel guards. `backtest_pricing` /
  `backtest_pricing_in_sample` reject an `expected_returns` panel whose dates are not a unique
  `DatetimeIndex` inside the fold's calendar, or that carries an asset absent from the view —
  validated **before** the structural horizon tail is dropped, so a bad date or phantom asset hidden
  in that tail cannot slip through (previously an out-of-fold or duplicated date was pooled as a
  genuine OOS observation), and **before** any emptiness short-circuit, so a zero-row panel cannot
  smuggle a malformed column either. `backtest_forecast` rejects a forecast whose asset labels are
  non-unique or carry a label absent from the view (previously a phantom asset was silently dropped,
  scored as an abstention; a duplicate label raised a cryptic pandas error). All messages name the
  method. `check_output_shapes` mirrors the driver guard exactly for pricing outputs: prediction
  dates must be a unique `DatetimeIndex` and column labels must stay unique after string
  normalization.
- Pricing drivers pool per-fold panels on **string-normalized** asset labels, matching what they
  validate. Previously validation ran on `str(column)` but concatenation kept the original labels,
  so one fold emitting the integer column `1` and another the string column `"1"` each passed yet
  pooled into two distinct, half-empty assets — one asset silently became two.
- Portfolio sorts no longer let holding-period return availability change formation-period
  breakpoints or bin membership. Signals, returns, weights, eligibility, and breakpoint-universe
  masks are validated on unique axes and aligned by pandas labels; missing mask values mean false,
  infinities are rejected, and thin or signal-degenerate breakpoint universes now fail closed
  instead of silently falling back to all stocks or emitting collapsed quantiles. Value-weighted
  bins with no positive observed weight remain `NaN` instead of silently becoming equal-weighted.
  `SortResult.counts` now explicitly counts frozen formation members, including members whose
  realized return is missing.

## [0.2.2] - 2026-07-07

Documentation and packaging refresh only — no functional changes since 0.2.1.

### Changed

- Rebuilt the documentation site and README: an academic structure with a pain-first overview,
  grouped navigation, an ecosystem page, and a "How to cite" section. The docs now live at
  <https://numeraire.py-numeraire.org/> (Cloudflare Pages), cross-linked with the companion
  packages' sites.
- Updated the `Documentation` project URL to the new docs domain.

### Added

- `CITATION.cff` (so GitHub renders a "Cite this repository" button) and a BibTeX snippet in the
  README.

## [0.2.1] - 2026-07-06

Ecosystem release: the plotting and data companion packages are now on PyPI, and this
release adds convenience extras to pull them in. Also carries the post-0.2.0 API work
(all backward-compatible — old names keep working as deprecated aliases for one release).

### Added

- **Ecosystem extras** — `numeraire[graphics]` (pulls `numeraire-graphics`),
  `numeraire[data]` (pulls `numeraire-dataset`), and `numeraire[all]`. `pip install numeraire`
  stays the minimal spine; opt into the companions here.
- **`backtest(estimator, view, splitter, *, method, in_sample=False)`** — a discoverable
  dispatching entry point that routes by the model's capability and the view type to the typed
  drivers `backtest_weights` / `backtest_forecast` / `backtest_panel` / `backtest_pricing` /
  `backtest_pricing_in_sample`.
- **Risk-adjusted evaluators** — `TreynorEvaluator`, `InformationRatioEvaluator`, `M2Evaluator`,
  `SortinoEvaluator`; **`ICEvaluator`** (rank IC); **`ExposureEvaluator`** (per-date leverage /
  net / turnover / concentration); and **`fama_macbeth`** (two-pass cross-sectional regression
  with Shanken + Newey-West).

### Changed

- Renamed for a clearer register (old names remain as **deprecated aliases** emitting
  `DeprecationWarning`): `walk_forward*` → `backtest_*`, `adjust_tests` → `adjust_pvalues`,
  `clark_west` → `clark_west_test`, `make_sorts` → `sort_portfolios`,
  `OOSR2Evaluator` → `OutOfSampleR2Evaluator`. `WalkForwardSplitter` is unchanged.

### Fixed

- Weights/forecast backtests now align the model's output to the view's asset order by **label**
  before scoring (previously positional), so a method returning permuted/subset columns is scored
  correctly rather than silently mis-scored. Clear errors on a missing or misused splitter.

## [0.2.0] - 2026-07-05

First tagged release. The spine is capability-complete: `to_weights`, `to_forecast`,
and `to_pricing` are all crystallized protocols with walk-forward drivers, native
evaluators, and a conformance suite.

### Added

- **Pricing capability** — `SupportsPricing.expected_returns`, `walk_forward_pricing` /
  `pricing_in_sample`, cross-sectional R² and average-|α| evaluators, and
  `numeraire.comparison.compare` to score competing pricing models (factor models, SDFs,
  risk-premium estimators) on one common set of test assets. Every result row carries an
  explicit `protocol` label (`in_sample` / `walk_forward`), so explanatory numbers are
  never confusable with out-of-sample ones.
- **Conformance suite** (`numeraire.testing.check_estimator`) — capabilities, output
  shapes, determinism, a no-look-ahead property test, and an engine round-trip: the
  self-certification any extension runs before its numbers are trusted.
- **Reference registry** (`numeraire.reference.ReferenceResult`) — pinned published
  results with tolerance bands and data-access tiers (`public` / `credentialed` /
  `restricted`); CI stays green on public data while the same case runs verbatim wherever
  licensed data is present.
- **Bundled baselines** (`numeraire.baselines`) — equal weight (1/N), minimum variance,
  mean-variance, and historical mean, registered through the same entry-point mechanism as
  any external method.
- **Weight-stream simulator** — `simulate_weights` + `RebalanceSchedule` with explicit
  drift, turnover, and cost conventions.
- **Inference toolkit** (`core.stats`) — GRS, Clark-West, paired Sharpe
  (Jobson-Korkie–Memmel), HAC alpha regression, Bonferroni/Holm/BHY adjustments, and
  certainty-equivalent / return-loss / performance-fee measures.
- **Cross-sectional data layer** — `CrossSectionView` with zero-copy point-in-time
  windows, a ragged-panel walk-forward engine, parallel fold execution, refit-cadence
  control, and a validation-split helper.
- **Interop** — polars/arrow ingestion at the view boundary (narwhals-optional, zero new
  hard dependencies) and a skfolio adapter (`[skfolio]` extra) that wraps portfolio
  optimizers as `to_weights` estimators.

Python ≥ 3.11, pandas ≥ 2.2.

[Unreleased]: https://github.com/py-numeraire/numeraire/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/py-numeraire/numeraire/releases/tag/v0.2.0
