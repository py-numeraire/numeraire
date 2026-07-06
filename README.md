<h1 align="center">numeraire</h1>

<p align="center">
  <strong>A research framework for backtesting, comparing, and replicating empirical
  asset-pricing and financial-econometrics methods.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/numeraire/"><img alt="PyPI" src="https://img.shields.io/pypi/v/numeraire.svg"></a>
  <a href="https://pypi.org/project/numeraire/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/numeraire.svg"></a>
  <a href="https://github.com/py-numeraire/numeraire/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/py-numeraire/numeraire/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://numeraire.py-numeraire.org/"><img alt="Documentation" src="https://img.shields.io/badge/docs-latest-blue.svg"></a>
  <a href="https://github.com/py-numeraire/numeraire/blob/main/LICENSE"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg"></a>
</p>

<p align="center">
  <a href="https://numeraire.py-numeraire.org/">Documentation</a> ·
  <a href="https://numeraire.py-numeraire.org/quickstart.html">Quickstart</a> ·
  <a href="https://numeraire.py-numeraire.org/api/index.html">API Reference</a> ·
  <a href="https://pypi.org/project/numeraire/">PyPI</a> ·
  <a href="#how-to-cite">Citation</a>
</p>

---

`numeraire` provides a stable, **representation-agnostic** spine for empirical asset pricing: a
point-in-time data view, a walk-forward out-of-sample engine, a library of evaluators and
statistical tests, and a tidy result schema. Its core never encodes a specific method's internal
form — it defines *capabilities* (what a model can produce: portfolio weights, return forecasts, a
priced cross-section), and dispatches on them. Linear-factor, nonlinear, neural, and distributional
methods are therefore all first-class, and each plugs into the same views, the same engine, and the
same schema as a peer.

The *numéraire* is the reference unit against which all prices are measured. The framework is
oriented toward reproducible empirical asset pricing: results are produced out of sample, on a
recorded data vintage, and scored with a metric matched to the model.

## Why numeraire

Empirical asset-pricing results are sensitive to a few well-known failure modes: a signal that
overlaps the return it predicts, a headline metric mismatched to the model, or preprocessing that is
not recorded. numeraire is designed so these are addressed by the structure of the framework.

- **Out-of-sample by construction.** A point-in-time view exposes only the data available at each
  decision date, and the walk-forward engine computes realised returns from the original data rather
  than from a model's output. A conformance test perturbs future data and verifies that earlier
  outputs do not change.
- **The metric matches the object.** Evaluators dispatch on what a model produces — a timing strategy
  is scored by its Sharpe ratio, a return forecast by its out-of-sample R², a pricing model by its
  cross-sectional fit.
- **Every result carries its provenance.** Each result row records the data vintage, a hash of the
  method and preprocessing configuration, and whether the number is in-sample or out-of-sample.

Underlying this, the core describes a model by its *capabilities* — the outputs it can produce
(portfolio weights, return forecasts, a priced cross-section) — not by any specific internal form.
Methods as different as linear factor models, neural predictors, and stochastic discount factors run
through the same engine, evaluators, and result schema.

## Installation

```bash
pip install numeraire
```

`numeraire` requires Python 3.11+. The base install is the spine plus the native evaluators; opt into
the companion packages and integrations through extras:

```bash
pip install "numeraire[all]"        # + the plotting and data companion packages
pip install "numeraire[graphics]"   # + numeraire-graphics (grammar-of-graphics figures)
pip install "numeraire[data]"       # + numeraire-dataset (data loaders and builders)
pip install "numeraire[skfolio]"    # + the skfolio portfolio-optimizer adapter
```

Using [uv](https://docs.astral.sh/uv/):

```bash
uv add numeraire            # or: uv add "numeraire[all]"
```

## Quickstart

Build a point-in-time view, run a walk-forward out-of-sample backtest, and evaluate it — on
synthetic data, no external source required:

```python
import numpy as np
import pandas as pd

from numeraire import SharpeEvaluator, TimeSeriesView, WalkForwardSplitter, backtest
from numeraire.baselines import EqualWeight

rng = np.random.default_rng(0)
dates = pd.date_range("2000-01-31", periods=120, freq="ME")
returns = pd.DataFrame(rng.normal(0.01, 0.05, (120, 4)), index=dates, columns=list("ABCD"))

view = TimeSeriesView(returns)
splitter = WalkForwardSplitter(min_train=60, test_size=12)
result = backtest(EqualWeight(), view, splitter, method="equal_weight")

print(SharpeEvaluator().evaluate(result)[["method", "metric", "value", "protocol"]])
```

`backtest` reads the fitted model's capability and the view type and dispatches to the right typed
driver; `SharpeEvaluator` emits rows of the standard tidy schema. See the
[quickstart](https://numeraire.py-numeraire.org/quickstart.html) for the full walk-through,
including a forecasting example.

## The ecosystem

`numeraire` is the spine of a small, deliberately decoupled family of packages. The core ships only
tiny public example slices and no bundled methods; everything else is an optional, first-class peer.

| Package | Role | Install |
| --- | --- | --- |
| **[numeraire](https://numeraire.py-numeraire.org/)** | The spine: views, engine, evaluators, result schema, statistical tests. | `pip install numeraire` |
| **[numeraire-graphics](https://graphics.py-numeraire.org/)** | Grammar-of-graphics figures over the result schema and Output objects. | `pip install numeraire-graphics` |
| **[numeraire-dataset](https://dataset.py-numeraire.org/)** | Open, reproducible data loaders and point-in-time builders. | `pip install numeraire-dataset` |

Reproductions of published methods, and a lab's own unpublished methods, live in **separate
packages** that register through the `numeraire.methods` entry-point group and pin `numeraire` — they
are discovered at install time without any edit to core. A public reproduction collection is in
preparation.

## What you can do with it

- **Research** — reproduce a published result within a tolerance band, and compare competing pricing
  models on one shared panel of test assets in the Fama–French / GRS tradition, with the
  corresponding significance tests.
- **Backtesting** — construct portfolios and score them out of sample, turn a target-weight stream
  into realised net returns through an accounting simulator with explicit cost conventions, and
  evaluate with risk-adjusted, information-coefficient, and exposure measures.

## Documentation

Full documentation — installation, a runnable quickstart, the architecture, the extension guide, and
the API reference — is at **<https://numeraire.py-numeraire.org/>**.

## How to cite

If you use `numeraire` in your research, a citation is appreciated. The repository ships a
[`CITATION.cff`](CITATION.cff) (GitHub renders a **Cite this repository** button from it), and the
equivalent BibTeX is:

```bibtex
@software{wu_numeraire,
  author  = {Wu, Yuheng},
  title   = {{numeraire: a research framework for backtesting, comparing,
             and replicating empirical asset-pricing methods}},
  year    = {2026},
  version = {0.2.1},
  url     = {https://github.com/py-numeraire/numeraire},
  license = {BSD-3-Clause}
}
```

## Contributing

Contributions to the spine — evaluators, statistical tests, and engine or view improvements — are
welcome. Methods are distributed as separate packages rather than added to core: a method is any
object with `fit(view) -> model` that advertises a capability, registered through the
`numeraire.methods` entry point and self-certified with `numeraire.testing.check_estimator` (see the
[extension guide](https://numeraire.py-numeraire.org/extending.html)).

Development uses [uv](https://docs.astral.sh/uv/), `ruff`, `basedpyright` (strict on core), and
`import-linter` for the architecture boundary:

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run basedpyright src/numeraire/core     # strict types on the spine
uv run lint-imports                        # architecture boundary
uv run pytest                              # tests (public / synthetic data only)
```

## License

BSD-3-Clause. Author: Yuheng Wu. Tests use public or synthetic data only; never commit
CRSP / WRDS / proprietary data or credentials.
