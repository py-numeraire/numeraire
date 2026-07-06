# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Versions are tag-driven (`hatch-vcs`).

## [Unreleased]

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
