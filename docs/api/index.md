# API reference

The API is organised in three layers: the top-level `numeraire` namespace (the common surface,
re-exported for convenience), the `numeraire.core` spine, and the core-adjacent infrastructure
(`testing`, `reference`, `comparison`, `baselines`, `adapters`).

```{toctree}
:maxdepth: 1

numeraire
core.data
core.engine
core.evaluators
core.protocols
core.capabilities
core.schema
core.registry
core.simulate
core.splitter
core.stats
core.sorts
testing
reference
comparison
baselines
adapters.skfolio
```

## Top-level namespace

The most common classes and functions are re-exported at the top level, so
`from numeraire import TimeSeriesView, walk_forward, SharpeEvaluator` works directly.

```{eval-rst}
.. currentmodule:: numeraire

.. autosummary::
   :toctree: generated
   :nosignatures:

   TimeSeriesView
   CrossSectionView
   WalkForwardSplitter
   validation_split
   walk_forward
   walk_forward_panel
   walk_forward_forecast
   walk_forward_pricing
   pricing_in_sample
   config_hash
   WeightsOutput
   PanelWeightsOutput
   ForecastOutput
   PricingOutput
   SharpeEvaluator
   MeanReturnEvaluator
   CEQEvaluator
   AlphaEvaluator
   StrategyReturnEvaluator
   OOSR2Evaluator
   SquaredErrorDiffEvaluator
   ClarkWestEvaluator
   CrossSectionalR2Evaluator
   AverageAbsAlphaEvaluator
   DataView
   Estimator
   Model
   Splitter
   Evaluator
   SupportsWeights
   SupportsForecast
   SupportsPricing
   validate_result
   register_evaluator
   get_evaluator
   available_evaluators
   simulate_weights
   RebalanceSchedule
   SimulationResult
   make_sorts
   SortResult
   grs_test
   sharpe_diff_test
   clark_west
   alpha_regression
   adjust_tests
   newey_west_lrv
   certainty_equivalent
   return_loss
   performance_fee
```
