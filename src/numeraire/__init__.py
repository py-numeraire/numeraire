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
from numeraire.core.splitter import WalkForwardSplitter

__all__ = [
    "RESULT_COLUMNS",
    "CrossSectionView",
    "DataView",
    "Estimator",
    "Evaluator",
    "ForecastOutput",
    "MeanReturnEvaluator",
    "Model",
    "OOSR2Evaluator",
    "PanelWeightsOutput",
    "SharpeEvaluator",
    "Splitter",
    "SquaredErrorDiffEvaluator",
    "StrategyReturnEvaluator",
    "SupportsForecast",
    "SupportsWeights",
    "TimeSeriesView",
    "WalkForwardSplitter",
    "WeightsOutput",
    "available_evaluators",
    "capabilities",
    "config_hash",
    "get_evaluator",
    "register_evaluator",
    "validate_result",
    "walk_forward",
    "walk_forward_forecast",
    "walk_forward_panel",
]
