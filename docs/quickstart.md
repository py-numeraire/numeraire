# Quickstart

This walk-through runs end to end on synthetic data — no external data source required. It builds
a point-in-time view, runs a walk-forward out-of-sample backtest of a bundled baseline, evaluates
it, reads the tidy result rows, and finishes with a short forecasting example.

Everything on this page is self-contained; paste it into a session and it runs.

## 1. Synthetic returns

We start from a wide `(date × asset)` frame of returns. Any pandas frame with a sorted, unique
`DatetimeIndex` works; here we draw five assets of monthly returns.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
dates = pd.date_range("2000-01-31", periods=180, freq="ME")
returns = pd.DataFrame(
    rng.normal(0.008, 0.05, size=(180, 5)),
    index=dates,
    columns=["asset_0", "asset_1", "asset_2", "asset_3", "asset_4"],
)
```

## 2. Build a view

A {class}`~numeraire.core.data.TimeSeriesView` wraps the returns block (and, optionally, predictor
blocks) on an explicit calendar. The view owns the alignment between the information known at a
decision date `t` and the return realised over the next horizon `(t, t+h]`, so a method can never
index future returns by hand.

```python
from numeraire import TimeSeriesView

view = TimeSeriesView(returns)              # excess returns by default; pass risk_free= to convert
view.assets                                 # ['asset_0', ..., 'asset_4']
view.calendar                               # the 180 monthly decision dates
```

If your returns are *raw* rather than excess, pass `risk_free=<Series>` and they are converted
internally. The forecast horizon defaults to one period; pass `horizon=` for multi-period targets.

## 3. Backtest

{func}`~numeraire.core.engine.backtest` runs the out-of-sample loop. It is the discoverable entry
point: it inspects the fitted model's capability (`to_weights` / `to_forecast` / `to_pricing`) and
the view type and dispatches to the right typed driver, returning the matching output. A
{class}`~numeraire.core.splitter.WalkForwardSplitter` yields expanding (or rolling) `(train, test)`
folds; at each fold the estimator is fitted on the train view and asked for its weights on the test
view, and realised profit-and-loss is computed from the original view so the model never touches
future returns.

We use the bundled {class}`~numeraire.baselines.EqualWeight` (1/N) baseline.

```python
from numeraire import WalkForwardSplitter, backtest
from numeraire.baselines import EqualWeight

splitter = WalkForwardSplitter(min_train=60, test_size=12, expanding=True)
result = backtest(
    EqualWeight(), view, splitter,
    method="equal_weight",
    data_vintage="synthetic-v1",
)
```

For an explicit return type (or to skip the probe fit `backtest` uses to read capabilities —
it fits the selected driver's first train window, never the full sample), call the typed driver
directly — {func}`~numeraire.core.engine.backtest_weights`
here; {func}`~numeraire.core.engine.backtest_forecast`,
{func}`~numeraire.core.engine.backtest_panel` and {func}`~numeraire.core.engine.backtest_pricing`
are its siblings.

The return value is a {class}`~numeraire.core.engine.WeightsOutput` — a frozen container carrying
the realised `weights` and `realized` panels plus the provenance every result row needs:

```python
result.config_hash    # '44136fa355b3' — a stable hash of the (empty) config dict
result.run_id         # 'equal_weight-44136fa355b3'
result.strategy_returns()   # the realised, no-look-ahead P&L series
```

## 4. Evaluate

Evaluators dispatch by capability and emit rows of the standard tidy result schema. A weights
output is scored by, for example, {class}`~numeraire.core.evaluators.SharpeEvaluator`:

```python
from numeraire import SharpeEvaluator

rows = SharpeEvaluator(periods_per_year=12).evaluate(result)
print(rows.to_string(index=False))
```

```text
                   run_id       method metric    value universe capability     protocol  config_hash data_vintage
equal_weight-44136fa355b3 equal_weight sharpe 1.213409      n=5 to_weights walk_forward 44136fa355b3 synthetic-v1
```

### Reading the result rows

Every evaluator, everywhere in the framework, emits these columns
({data}`~numeraire.core.schema.RESULT_COLUMNS`). Downstream plotting and aggregation consume the
long format, so the choice of plotting library stays decoupled.

`run_id`, `method`
: Identify the run and the method that produced it.

`date`
: The time dimension. Summary metrics stamp the last evaluation date; per-period metrics carry one
  row per date.

`metric`, `value`
: The metric name and its scalar value — here `sharpe`.

`universe`
: A compact label for the asset set (`n=5` for a panel; the asset name for single-asset timing).

`capability`
: The capability the metric was computed against (`to_weights`), so the metric always matches the
  object being scored.

`protocol`
: The evaluation discipline — `walk_forward` (out-of-sample) or `in_sample`. This keeps an
  explanatory in-sample number from ever being mistaken for an out-of-sample one.

`config_hash`, `data_vintage`
: Provenance. `config_hash` is a stable hash of the preprocessing/method config; `data_vintage`
  is the data snapshot you stamped the run with. Together they pin exactly what produced a number.

For a time series rather than a scalar, use the per-period companion
{class}`~numeraire.core.evaluators.StrategyReturnEvaluator`, which emits one row per date (its
cumulative sum is the equity curve):

```python
from numeraire import StrategyReturnEvaluator

curve = StrategyReturnEvaluator().evaluate(result)   # one row per date, metric='strategy_return'
```

## 5. A forecast example

Forecasting methods advertise the `to_forecast` capability; `backtest` dispatches them to
{func}`~numeraire.core.engine.backtest_forecast`, which uses the forecast-origin convention: at
each origin `t` the model is fit on data up to and including `t` and predicts the return over
`(t, t+h]`. The engine records the realised return and, for free, the prevailing historical-mean
benchmark — the Goyal–Welch reference the out-of-sample R² is measured against.

Here the bundled {class}`~numeraire.baselines.HistoricalMean` forecasts a single market series;
because it *is* the benchmark, its out-of-sample R² is zero by construction.

```python
from numeraire import OutOfSampleR2Evaluator, backtest
from numeraire.baselines import HistoricalMean

market = returns[["asset_0"]]
fview = TimeSeriesView(market, horizon=1)
fout = backtest(HistoricalMean(), fview, min_train=60, method="historical_mean")

print(OutOfSampleR2Evaluator(benchmark="historical").evaluate(fout)[["metric", "value", "capability"]].to_string(index=False))
```

```text
    metric  value  capability
oos_r2_pct    0.0 to_forecast
```

Swap in a real predictive model (one that consumes a predictor block and returns a per-asset
forecast) and the same call reports whether it beats the prevailing mean out of sample. Pair it
with {class}`~numeraire.core.evaluators.ClarkWestEvaluator` for the significance test appropriate
to a nested benchmark.

## Next steps

- {doc}`architecture` — how views, capabilities, the engine family, and evaluators fit together.
- {doc}`extending` — turn your own model into a first-class, self-certified method.
