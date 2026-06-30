"""1/A — the conservative-slope equity-premium forecast (Li, Li, Lyu & Yu, RFS 2025).

"How to Dominate the Historical Average." A one-asset timing rule that predicts the next-period
excess return with a deliberately *conservative constant* slope ``±1/A`` on a standardized
predictor::

    forecast_{t+1} = HM_t + (sign / A) * x_std,t

where ``HM_t`` is the historical mean of the excess return over the estimation window, ``x_std,t``
is the predictor standardized (mean 0, sd 1) on that same window, and ``sign`` comes from economic
theory (Campbell-Thompson). A larger ``A`` is more conservative. The shrunk slope has lower bias
than the zero-slope historical average but the same variance, so it dominates HM out-of-sample
(paper Theorems 1-2).

Exposes ``to_forecast`` (a one-asset return forecast); drive it with
:func:`numeraire.core.engine.walk_forward_forecast` and score with the GW2008
:class:`~numeraire.core.evaluators.OOSR2Evaluator`. Reproduces paper Table 3 (dp) within
<= 0.15pp on public Goyal-Welch annual data — see ``tests/test_method_conservative_slope.py``.
"""

from __future__ import annotations

import pandas as pd

from numeraire.core import capabilities
from numeraire.core.data import TimeSeriesView


class _ConservativeSlopeModel:
    """Fitted 1/A model: recomputes window stats from the view handed to ``forecast``."""

    def __init__(self, a: float, sign: float, predictor: str | None) -> None:
        self._a = a
        self._sign = sign
        self._predictor = predictor

    def capabilities(self) -> set[str]:
        return {capabilities.TO_FORECAST}

    def forecast(self, view: TimeSeriesView) -> pd.Series:
        feats = view.features_frame()
        rets = view.returns_frame()
        col = self._predictor if self._predictor is not None else str(feats.columns[0])
        x = feats[col]
        x_std = float((x.iloc[-1] - x.mean()) / x.std(ddof=1))  # last predictor in the window
        hm = rets.mean()  # historical mean per asset (= the GW benchmark)
        return hm + (self._sign / self._a) * x_std


class ConservativeSlope:
    """The 1/A estimator. Each ``fit`` window yields one forecast at its right edge.

    Parameters
    ----------
    a:
        Conservatism ``A`` (slope magnitude is ``1/A``); the paper sweeps ``A in [50, 1000]``.
    sign:
        Theory-implied slope sign (``+1`` for dividend-price, per Campbell-Thompson).
    predictor:
        Feature column to use. Defaults to the view's single/first feature.
    """

    def __init__(self, a: float, *, sign: float = 1.0, predictor: str | None = None) -> None:
        if a <= 0:
            raise ValueError("A must be positive")
        self.a = a
        self.sign = sign
        self.predictor = predictor

    def fit(self, view: TimeSeriesView) -> _ConservativeSlopeModel:
        _ = view
        # The rule is window-stateless: the model recomputes HM and standardization from the
        # window it is asked to forecast on (the engine passes the same fit/forecast view).
        return _ConservativeSlopeModel(self.a, self.sign, self.predictor)
