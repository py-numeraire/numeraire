"""numeraire — a research framework for empirical asset pricing.

The reference unit against which methods are measured. The spine (``numeraire.core``)
is method-agnostic and depended upon by everything; methods and adapters depend on it,
never the reverse (enforced by import-linter, see pyproject ``[tool.importlinter]``).
"""

from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.engine import (
    ForecastOutput,
    PanelWeightsOutput,
    WeightsOutput,
    config_hash,
    walk_forward,
    walk_forward_forecast,
    walk_forward_panel,
)
from numeraire.core.evaluators import (
    AlphaEvaluator,
    CEQEvaluator,
    ClarkWestEvaluator,
    MeanReturnEvaluator,
    OOSR2Evaluator,
    SharpeEvaluator,
    SquaredErrorDiffEvaluator,
    StrategyReturnEvaluator,
)
from numeraire.core.protocols import (
    DataView,
    Estimator,
    Evaluator,
    Model,
    Splitter,
    SupportsForecast,
    SupportsWeights,
)
from numeraire.core.registry import (
    available_evaluators,
    get_evaluator,
    register_evaluator,
)
from numeraire.core.schema import RESULT_COLUMNS, validate_result
from numeraire.core.simulate import RebalanceSchedule, SimulationResult, simulate_weights
from numeraire.core.sorts import SortResult, make_sorts
from numeraire.core.splitter import WalkForwardSplitter, validation_split
from numeraire.core.stats import (
    adjust_tests,
    alpha_regression,
    certainty_equivalent,
    clark_west,
    grs_test,
    newey_west_lrv,
    performance_fee,
    return_loss,
    sharpe_diff_test,
)

__all__ = [
    "RESULT_COLUMNS",
    "AlphaEvaluator",
    "CEQEvaluator",
    "ClarkWestEvaluator",
    "CrossSectionView",
    "DataView",
    "Estimator",
    "Evaluator",
    "ForecastOutput",
    "MeanReturnEvaluator",
    "Model",
    "OOSR2Evaluator",
    "PanelWeightsOutput",
    "RebalanceSchedule",
    "SharpeEvaluator",
    "SimulationResult",
    "SortResult",
    "Splitter",
    "SquaredErrorDiffEvaluator",
    "StrategyReturnEvaluator",
    "SupportsForecast",
    "SupportsWeights",
    "TimeSeriesView",
    "WalkForwardSplitter",
    "WeightsOutput",
    "adjust_tests",
    "alpha_regression",
    "available_evaluators",
    "capabilities",
    "certainty_equivalent",
    "clark_west",
    "config_hash",
    "get_evaluator",
    "grs_test",
    "make_sorts",
    "newey_west_lrv",
    "performance_fee",
    "register_evaluator",
    "return_loss",
    "sharpe_diff_test",
    "simulate_weights",
    "validate_result",
    "validation_split",
    "walk_forward",
    "walk_forward_forecast",
    "walk_forward_panel",
]
