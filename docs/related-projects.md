# Related projects and scope

`numeraire` occupies a deliberately narrow niche, and it is most useful when that niche is clear.
This page states what the framework is, what it is *not*, and how it relates to the mature libraries
next to it. The intent is orientation, not competition: several of these are excellent tools that
`numeraire` complements or wraps rather than replaces.

## What numeraire is

A **spine** for empirical asset pricing: point-in-time data views, a walk-forward out-of-sample
engine, capability-dispatched evaluators and statistical tests, a tidy result schema, and an
open registry through which methods plug in as first-class extensions. Its purpose is to make
**backtesting, comparison, and replication reproducible and comparable** across methods of very
different internal form.

## What numeraire is not

- **Not a portfolio-optimization library.** It does not implement constrained mean-variance
  optimizers, risk budgeting, or hierarchical allocation. When a constrained optimizer is needed,
  the `numeraire[skfolio]` adapter wraps [skfolio](https://skfolio.org/); the optimizers stay in
  skfolio.
- **Not a trading or execution system.** There is no order routing, no live market connectivity,
  no broker integration. The accounting simulator turns a target-weight stream into realised net
  returns under explicit cost conventions — an evaluation tool, not an execution engine.
- **Not a data warehouse.** The spine ships only tiny public example slices; data acquisition and
  cleaning live in the separate `numeraire-dataset` package as transparent ETL (see
  {doc}`ecosystem`).
- **Not a general econometrics package.** It does not aim to cover the breadth of statistical models
  that statsmodels or linearmodels do; it reuses that machinery where it needs it.

## How it relates to neighbouring libraries

**[statsmodels](https://www.statsmodels.org/) / [linearmodels](https://bashtage.github.io/linearmodels/)**
: The estimation and inference layer for regression, panel, IV, and system models. `numeraire`
  complements them: it supplies the point-in-time discipline, the out-of-sample protocol, and the
  reproduction harness *around* a method, and reuses established estimators and tests rather than
  re-deriving them. Their econometric depth and `numeraire`'s backtesting spine are orthogonal.

**[skfolio](https://skfolio.org/)**
: A scikit-learn-compatible portfolio-optimization and risk-management library. It answers "given
  expected returns and risk, what is the optimal portfolio?"; `numeraire` answers "how does a method
  perform out of sample, and does it reproduce a published result?". They compose — the adapter runs
  an skfolio optimizer as a `to_weights` method inside the walk-forward engine.

**[qlib](https://github.com/microsoft/qlib) / [zipline](https://github.com/quantopian/zipline)**
: Full quantitative-investment or event-driven backtesting platforms with data pipelines, model
  training, and execution modelling. `numeraire` is smaller and more academic in scope: a
  representation-agnostic spine for method comparison and replication, not an end-to-end
  alpha-to-execution platform.

**[scikit-learn](https://scikit-learn.org/)**
: The estimator/conformance idiom is a direct influence — `fit`, duck-typed protocols, and a
  `check_estimator`-style conformance suite. `numeraire` adapts it to the point-in-time, walk-forward
  setting that time-ordered financial data requires, where a plain cross-validation split would leak.

## When to reach for numeraire

Reach for it when you want to **compare methods of different internal form on the same footing**,
**reproduce a paper's headline within a tolerance band**, or **run a backtest whose out-of-sample
discipline and data provenance are structural rather than a matter of author care**. For pure
in-sample estimation, portfolio optimization, or production execution, one of the libraries above is
the better fit — often used *through* `numeraire` rather than instead of it.
