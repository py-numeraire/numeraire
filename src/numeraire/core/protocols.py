"""Core contracts (spine). Protocols, not base classes — methods conform by duck typing.

Spine protocols (``DataView``, ``Estimator``, ``Splitter``, ``Evaluator``, result schema)
are committed. The capability layer (``to_weights`` / ``to_pricing`` / ...) is v0 and is
expected to crystallize from the first three real adapters, so ``Model`` only
mandates ``capabilities()`` — concrete extractors are optional and dispatched by capability.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataView(Protocol):
    """A point-in-time aligned view of the data. A returns panel is one realization."""

    def window(self, end: object) -> DataView:
        """Return the view restricted to information available up to ``end`` (no look-ahead)."""
        ...

    @property
    def calendar(self) -> pd.DatetimeIndex:
        """Rebalancing / observation timestamps."""
        ...


@runtime_checkable
class Model(Protocol):
    """A fitted model. Exposes whatever capabilities it has — capabilities, not mandatory methods.

    Optional, dispatched by capability (do NOT make these mandatory):
    ``to_weights(view) -> pd.DataFrame | pd.Series``, ``expected_returns(view) -> pd.DataFrame``,
    ``to_density(view)``, ``to_surface(view)``, ...

    Capability → method-name mapping (the three crystallized surfaces follow *different* naming
    patterns, so an extension author must map deliberately, not by rote):

    - ``capabilities.TO_WEIGHTS`` → ``to_weights()``
    - ``capabilities.TO_FORECAST`` → ``forecast()`` (**not** ``to_forecast``)
    - ``capabilities.TO_PRICING`` → ``expected_returns()`` (**not** ``to_pricing``)

    A model declaring a capability must expose its mapped method; the conformance suite
    (``numeraire.testing.check_capabilities``) enforces this.
    """

    def capabilities(self) -> set[str]:
        """The set of capability names this model supports (see ``numeraire.core.capabilities``)."""
        ...


@runtime_checkable
class SupportsWeights(Protocol):
    """Capability protocol (v0): a model that emits portfolio/timing weights (``to_weights``).

    Optional and dispatched by capability — a model advertises it via
    ``capabilities() >= {capabilities.TO_WEIGHTS}``. Kept deliberately thin; the capability
    layer crystallizes once the third real adapter lands.
    """

    def to_weights(self, view: DataView) -> pd.DataFrame | pd.Series:
        """Return per-date weights for each date in ``view.calendar``.

        A fixed-universe time-series method returns a wide ``(date x asset)`` ``pd.DataFrame``; a
        cross-sectional (ragged / entering-exiting universe) method returns a long ``pd.Series`` on
        a ``(date, asset)`` MultiIndex — the panel engine (``backtest_panel``) accepts that form
        (or an equivalent one-column frame). Either way the engine aligns the returned labels to
        realized returns before scoring, so column / row order need not be canonical.
        """
        ...


@runtime_checkable
class SupportsForecast(Protocol):
    """Capability protocol (v0): a model that emits a next-horizon return forecast.

    The forecast-origin engine fits on a window ending at ``t`` and calls ``forecast`` on that
    same window; the returned value is the prediction of the return over ``(t, t+h]``, where
    ``t`` is the window's last date. Advertised via ``capabilities() >= {TO_FORECAST}``.
    """

    def forecast(self, view: DataView) -> pd.Series:
        """Per-asset forecast of the return over ``(t, t+h]`` (``t`` is the view's last date)."""
        ...


@runtime_checkable
class SupportsPricing(Protocol):
    """Capability protocol: a model that prices a cross-section of test assets (``to_pricing``).

    The single shared operation across the pricing/SDF family (factor models, SDFs, three-pass
    risk-premium estimators): the cross-section of **expected returns** on a set of assets. A
    conditional model varies its estimate by date (e.g. a characteristics-driven loading times a
    factor premium); an unconditional model returns the same row every date (broadcast). Advertised
    via ``capabilities() >= {TO_PRICING}``.

    Kept deliberately to this one method — the bespoke per-method accessors (loadings, latent
    factors, per-candidate premia) stay method-local; only the pricing surface the framework's
    evaluators and comparison harness consume is standardized here.
    """

    def expected_returns(self, view: DataView) -> pd.DataFrame:
        """Return ``(date x asset)`` expected returns for each date in ``view.calendar``."""
        ...


@runtime_checkable
class Estimator(Protocol):
    """scikit-learn-compatible. The method-specific body lives in methods/ or a lab repo."""

    def fit(self, view: DataView) -> Model:
        """Fit on a (point-in-time) view and return a fitted ``Model``."""
        ...


@runtime_checkable
class Splitter(Protocol):
    """Yields (train, test) views — purge/embargo/PIT aware. May wrap sklearn splitters."""

    def split(self, view: DataView) -> Iterator[tuple[DataView, DataView]]:
        """Yield (train, test) view pairs; the test view is strictly future of train."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores OOS output, emitting rows of the standard tidy result schema."""

    requires: ClassVar[set[str]]
    """Capabilities an OOS output must expose for this evaluator to run."""

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        """Return rows in the standard result schema (see ``numeraire.core.schema``)."""
        ...
