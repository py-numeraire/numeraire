---
title: numeraire
---

# numeraire

A research framework providing a **stable bedrock** for **backtesting, comparing, and
replicating** empirical asset-pricing and financial-econometrics methods.

The *numéraire* is the reference unit against which all prices are measured. The framework's
core stays **representation-agnostic**: it defines *capabilities* — what a model can produce
(portfolio weights, return forecasts, a priced cross-section) — never a specific method's
internal form. Linear-factor, nonlinear, neural, and distributional methods are therefore all
first-class citizens, and each plugs into the same point-in-time views, the same walk-forward
out-of-sample engine, and the same tidy result schema.

Three ideas organise the whole framework:

Point-in-time views
: A {class}`~numeraire.core.data.TimeSeriesView` or
  {class}`~numeraire.core.data.CrossSectionView` aligns returns and predictors on an explicit
  calendar and hands a model only the information available at each decision date. Look-ahead
  bias is made structurally difficult rather than left to author discipline.

Capabilities
: A model advertises what it can produce — {data}`~numeraire.core.capabilities.TO_WEIGHTS`,
  {data}`~numeraire.core.capabilities.TO_FORECAST`, {data}`~numeraire.core.capabilities.TO_PRICING`
  — and the engine and evaluators dispatch on those capabilities. The metric always matches the
  object.

The boundary rule
: `numeraire.core` depends on no specific method, and every method depends on core. Methods,
  adapters, and reference libraries never leak into the spine. The rule is enforced in CI by
  `import-linter`.

## Install

Until the first PyPI release, install from GitHub:

```bash
pip install "numeraire @ git+https://github.com/py-numeraire/numeraire"
```

or, in a `uv`-managed project:

```bash
uv add "numeraire @ git+https://github.com/py-numeraire/numeraire"
```

The base install is the spine plus the native evaluators; optional integrations (for example the
`skfolio` portfolio optimizers) live behind extras.

## A first backtest

```python
import numpy as np
import pandas as pd

from numeraire import SharpeEvaluator, TimeSeriesView, WalkForwardSplitter, backtest
from numeraire.baselines import EqualWeight

rng = np.random.default_rng(0)
dates = pd.date_range("2000-01-31", periods=120, freq="ME")
returns = pd.DataFrame(rng.normal(0.01, 0.05, (120, 4)), index=dates, columns=list("ABCD"))

view = TimeSeriesView(returns)
result = backtest(EqualWeight(), view, WalkForwardSplitter(min_train=60, test_size=12),
                  method="equal_weight")
print(SharpeEvaluator().evaluate(result)[["method", "metric", "value", "protocol"]])
```

## Where to go next

- {doc}`quickstart` — a runnable, end-to-end walk-through on synthetic data.
- {doc}`architecture` — the mental model: views, capabilities, the engine family, evaluators, the
  result schema, the simulator, and the boundary rule.
- {doc}`extending` — write and self-certify your own method.
- {doc}`comparison` — compare competing methods on one common set of test assets.
- {doc}`api/index` — the full API reference.

```{toctree}
:hidden:
:maxdepth: 2

quickstart
architecture
extending
comparison
api/index
changelog
```
