# Comparing methods

Replication asks whether one method reproduces a published number. **Comparison** asks a different
question: given several competing methods, which prices a common set of test assets best?
{mod}`numeraire.comparison` answers it in the Fama–French / GRS tradition — competing pricing models
judged by how well they explain **one shared panel of test assets**.

## The wrinkle a common panel creates

Different methods want different *representations* of the same test assets. A factor model may train
on a characteristic panel (a {class}`~numeraire.core.data.CrossSectionView`); an SDF or three-pass
estimator may train on a returns block (a {class}`~numeraire.core.data.TimeSeriesView`). Each must
still be scored against the *same* realised-return panel for the numbers to be comparable.

A {class}`~numeraire.comparison.ComparisonEntry` therefore carries both a `train_view` (the method's
native representation, on which it is fitted) and an optional `test_view` (its own representation of
the shared test assets, which it prices). The `test_view` must share the calendar and asset labels
of the common panel — a different view *type* is fine, which is the whole point. It defaults to
`train_view` for a method that trains directly on the test assets.
{func}`~numeraire.comparison.compare` verifies that alignment and always pulls **realised** returns
from the canonical test-asset panel, never from a model's own view.

## Example

Two competing unconditional pricers scored on one common panel:

```python
import numpy as np
import pandas as pd

from numeraire.comparison import ComparisonEntry, compare
from numeraire.core.data import TimeSeriesView

# ... UnconditionalMean and Shrunk are to_pricing estimators (see the extending guide) ...

view = TimeSeriesView(test_assets)          # the common (date x asset) test-asset panel
entries = [
    ComparisonEntry(name="unconditional", estimator=UnconditionalMean(), train_view=view),
    ComparisonEntry(name="shrunk",        estimator=Shrunk(0.5),         train_view=view),
]
rows = compare(entries, view, data_vintage="synthetic-v1")
print(rows[["method", "metric", "value", "protocol"]].to_string(index=False))
```

```text
       method        metric    value  protocol
unconditional         xs_r2 0.999009 in_sample
unconditional avg_abs_alpha 0.000275 in_sample
       shrunk         xs_r2 0.999009 in_sample
       shrunk avg_abs_alpha 0.003141 in_sample
```

The result is a single tidy frame in the standard schema, one block of rows per entry, ready to
pivot or plot.

## `in_sample` versus `walk_forward`

This is the most important distinction to keep straight, and the schema keeps it explicit through
the `protocol` column.

{func}`~numeraire.comparison.compare` is a **single full-sample-fit, in-sample** comparison — the
cross-sectional-pricing tradition, where every model is fitted once on all the data and its expected
returns are scored against the same sample. Every row it emits is tagged `protocol="in_sample"`. An
in-sample cross-sectional R² is an *explanatory* number: it says how well the model fits, not how
well it would have predicted.

For the **out-of-sample** counterpart, run {func}`~numeraire.core.engine.walk_forward_pricing` on
each method directly. It refits at every fold on that fold's point-in-time window and pools the
per-fold cross-sections into a `PricingOutput` tagged `protocol="walk_forward"`. The same evaluators
apply; only the discipline — and therefore the meaning of the number — differs.

Because the tag rides in every result row, an explanatory in-sample R² can never be silently
compared against, or mistaken for, an out-of-sample one.

## Pricing evaluators

`compare` defaults to the two native pricing metrics, and you can pass an explicit list to add or
narrow them:

{class}`~numeraire.core.evaluators.CrossSectionalR2Evaluator`
: The pricing headline — time-average each asset's realised and predicted returns, regress mean
  realised on mean predicted across assets, and report the R². The classic
  average-realised-versus-average-predicted plot, as a scalar.

{class}`~numeraire.core.evaluators.AverageAbsAlphaEvaluator`
: The magnitude companion — the cross-sectional mean of the absolute pricing errors (each asset's
  mean realised minus mean predicted).

For factor models with observable factor returns, the joint zero-alpha F-test lives in
{func}`numeraire.core.stats.grs_test`, which needs the factor returns that this deliberately generic
pricing surface does not assume.

## Honest metrics

Comparison is where it is easiest to fool yourself, so the framework bakes in three habits.

**Bands, not bit-equality.** A reproduction asserts an *invariant plus a headline scalar within a
tolerance band*, never an exact match. Data vintages drift — a data provider revises history, a
live download differs from the snapshot a paper used — and a band absorbs that drift while still
catching a real regression. {class}`~numeraire.reference.ReferenceResult` enforces exactly this and
rejects an all-NaN false green.

**The metric must match the object.** Evaluators dispatch by capability precisely so that a timing
strategy is judged by its Sharpe ratio, a predictive regression by its out-of-sample R², and a
pricing model by its cross-sectional fit. A method whose headline is a timing Sharpe should not be
ranked on an R² that is negative by design — every result row carries its `capability` so the right
metric is unambiguous.

**Provenance travels with the number.** `config_hash`, `data_vintage`, and `protocol` are on every
row, so a comparison table records not just *what* was measured but *under what preprocessing, on
which data snapshot, and with what discipline*. A number you cannot reproduce is a number you cannot
compare.
