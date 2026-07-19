"""Within-capability comparison harness — score several methods on one common set of test assets.

The long-deferred comparison item, crystallized for the pricing capability: a cross-sectional
asset-pricing comparison (Fama-French / GRS tradition), where competing models are judged by how
well they price **one shared panel of test assets**. Each entry brings its own *training*
view — a factor-model estimator may train on a characteristic panel (a ``CrossSectionView``), an
SDF or three-pass estimator on a returns block (a ``TimeSeriesView``) — but every model's expected
returns are scored against the same canonical realized-return panel, so the numbers are comparable.

The wrinkle a common panel creates: a representation-hungry model (e.g. one driven by
characteristics) needs its *own* view of those same test assets to price them. An entry therefore
may carry a ``test_view`` — same calendar and asset labels as ``test_assets``, possibly a different
view type — that its fitted model prices. :func:`compare` verifies that alignment and always pulls
**realized** returns from the canonical ``test_assets`` panel, never from a model's own view.

This module is core-adjacent infrastructure (it lives in ``numeraire`` proper, imports only
``numeraire.core`` + numpy/pandas, and is exempt from the boundary rule's method/adapter ban like
:mod:`numeraire.testing`). ``compare`` is a single full-sample-fit, in-sample comparison (every row
is tagged ``protocol="in_sample"``); for out-of-sample per-method scoring, run
:func:`numeraire.core.engine.backtest_pricing` on each method directly. The signature is kept
capability-generic (entries + a common test set + a list of evaluators); v1 implements the pricing
capability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.engine import (
    PricingOutput,
    _finalize_pricing,  # pyright: ignore[reportPrivateUsage]  # engine-internal, shared in-package
    _fit_isolated,  # pyright: ignore[reportPrivateUsage]  # engine-internal, shared in-package
    _pricing_realized,  # pyright: ignore[reportPrivateUsage]  # engine-internal, shared in-package
    _target_contract_meta,  # pyright: ignore[reportPrivateUsage]  # engine-internal, in-package
    config_hash,
)
from numeraire.core.evaluators import AverageAbsAlphaEvaluator, CrossSectionalR2Evaluator
from numeraire.core.protocols import DataView, Estimator, Evaluator, SupportsPricing
from numeraire.core.schema import validate_result

__all__ = ["ComparisonEntry", "compare"]


@dataclass(frozen=True)
class ComparisonEntry:
    """One competitor in a comparison: a named estimator, its training view, its test-asset view.

    Parameters
    ----------
    name:
        The label carried into every result row's ``method`` (and ``run_id``).
    estimator:
        Anything with ``fit(view) -> Model`` whose fitted model prices a cross-section
        (:class:`~numeraire.core.protocols.SupportsPricing`).
    train_view:
        The view the estimator is fit on (its native representation — a characteristic panel, a
        returns block, ...).
    test_view:
        The estimator's own representation of the **common** test assets, used to price them. Must
        carry the same calendar and asset labels as the shared ``test_assets`` (a different view
        *type* is fine — that is the whole point). Defaults to ``train_view`` for a method that
        trains directly on the test assets (e.g. an SDF on the test-asset returns block).
    config:
        Optional method config, hashed into the entry's ``config_hash`` provenance.
    """

    name: str
    estimator: Estimator
    train_view: DataView
    test_view: DataView | None = None
    config: dict[str, Any] | None = None


def _canonical_returns(test_assets: Any) -> pd.DataFrame:
    """The shared realized-return panel as a ``(date x asset)`` frame (from a frame or a view)."""
    if isinstance(test_assets, pd.DataFrame):
        panel = test_assets
    elif hasattr(test_assets, "returns_frame"):
        panel = test_assets.returns_frame()
    else:
        raise TypeError(
            "test_assets must be a (date x asset) DataFrame or a view with returns_frame()"
        )
    panel = panel.copy()
    panel.columns = [str(c) for c in panel.columns]
    return panel


def _price_entry(
    entry: ComparisonEntry,
    canonical: pd.DataFrame,
    realized_source: Any,
    *,
    data_vintage: str,
) -> PricingOutput:
    """Fit the entry, price the common test assets, and pair with canonical realized returns.

    ``realized_source`` is either a concrete core view (its own horizon-aware ``target_asof``
    pairing resolves ``predicted.loc[t]`` against the return over ``(t, t+h]``, exactly like the
    pricing drivers) or the pre-shifted ``(date x asset)`` frame built by :func:`compare` for a
    bare panel (the horizon-1 convention). Either way the alignment convention of
    :class:`~numeraire.core.engine.PricingOutput` is preserved — one class, one convention.

    Like the engine drivers, the fit runs on an isolated ``copy.deepcopy`` of the entry's
    estimator — never the caller's instance — under the same contract: the estimator must be
    deepcopy-able and must not share fit-relevant mutable state across copies.
    """
    model = _fit_isolated(entry.estimator, entry.train_view, entry.name)
    if capabilities.TO_PRICING not in model.capabilities() or not isinstance(
        model, SupportsPricing
    ):
        raise TypeError(
            f"comparison entry {entry.name!r}: fitted model does not support 'to_pricing'"
        )
    view = entry.test_view if entry.test_view is not None else entry.train_view
    predicted = model.expected_returns(view)
    predicted = predicted.copy()
    predicted.columns = [str(c) for c in predicted.columns]

    labels = set(canonical.columns)
    stray_assets = [c for c in predicted.columns if c not in labels]
    if stray_assets:
        raise ValueError(
            f"comparison entry {entry.name!r}: expected_returns priced assets {stray_assets} "
            "absent from the common test_assets panel (align asset labels)"
        )
    stray_dates = predicted.index.difference(canonical.index)
    if len(stray_dates):
        raise ValueError(
            f"comparison entry {entry.name!r}: expected_returns has dates absent from the common "
            f"test_assets calendar ({list(stray_dates[:3])}...); test_view must share the calendar"
        )
    # The effective horizon: a concrete core view carries its own; the bare-frame path is the
    # documented horizon-1 (next-row) convention. Frequency is inferred from the common calendar.
    horizon = 1 if isinstance(realized_source, pd.DataFrame) else int(realized_source.horizon)
    if isinstance(realized_source, pd.DataFrame):
        realized = realized_source.reindex(index=predicted.index, columns=predicted.columns)
    else:
        realized = _pricing_realized(realized_source, predicted)
    predicted, realized = _finalize_pricing(predicted, realized)
    chash = config_hash(entry.config)
    return PricingOutput(
        predicted=predicted,
        realized=realized,
        method=entry.name,
        config_hash=chash,
        data_vintage=data_vintage,
        run_id=f"{entry.name}-{chash}",
        protocol="in_sample",
        horizon=horizon,
        meta=_target_contract_meta(canonical.index, horizon),
    )


def compare(
    entries: Sequence[ComparisonEntry],
    test_assets: Any,
    *,
    evaluators: Sequence[Evaluator] | None = None,
    data_vintage: str = "unknown",
) -> pd.DataFrame:
    """Score every entry's expected returns on one common test-asset panel; return tidy result rows.

    Each entry is fit on its own ``train_view`` and prices the shared test assets through its
    ``test_view`` (defaulting to ``train_view``); realized returns always come from the canonical
    ``test_assets`` panel, and asset-label / calendar alignment is verified before scoring. The
    default evaluators are the two native pricing metrics (:class:`CrossSectionalR2Evaluator`,
    :class:`AverageAbsAlphaEvaluator`); pass an explicit list to add or narrow them.

    Alignment convention (the :class:`~numeraire.core.engine.PricingOutput` invariant):
    ``predicted.loc[t]`` prices the return realized over ``(t, t+h]``, exactly as in the pricing
    drivers. When ``test_assets`` is a concrete core view, its own horizon-aware pairing
    (``target_asof``) supplies the realized panel. When it is a bare ``(date x asset)`` frame,
    **horizon 1 is assumed**: ``realized.loc[t]`` is the panel's next row (``panel.shift(-1)``) —
    pass a view with the right ``horizon`` for multi-period targets. Prediction dates whose realized
    cross-section is entirely unrealized (the mechanically identified horizon tail) are dropped.

    Parameters
    ----------
    entries:
        The competitors (see :class:`ComparisonEntry`).
    test_assets:
        The common realized-return panel — a ``(date x asset)`` DataFrame (horizon-1 convention) or
        a view exposing ``returns_frame()`` (its own horizon).
    evaluators:
        Evaluators to run on each entry's :class:`~numeraire.core.engine.PricingOutput`. Defaults to
        the native pricing pair.
    data_vintage:
        Provenance stamp copied into every result row.

    Returns
    -------
    A single tidy DataFrame in the result schema (``method`` = each entry's name), validated against
    :data:`~numeraire.core.schema.RESULT_COLUMNS`.
    """
    if not entries:
        raise ValueError("compare needs at least one entry")
    evals = (
        list(evaluators)
        if evaluators is not None
        else [
            CrossSectionalR2Evaluator(),
            AverageAbsAlphaEvaluator(),
        ]
    )
    canonical = _canonical_returns(test_assets)
    # Realized source, per the (t, t+h] convention: a concrete view resolves horizon-aware via the
    # engine's own helper; a bare frame (or duck-typed view) uses the next-row (horizon-1) pairing.
    realized_source: Any
    if isinstance(test_assets, TimeSeriesView | CrossSectionView):
        realized_source = test_assets
    else:
        realized_source = canonical.shift(-1)
    parts: list[pd.DataFrame] = []
    for entry in entries:
        out = _price_entry(entry, canonical, realized_source, data_vintage=data_vintage)
        for ev in evals:
            parts.append(ev.evaluate(out))
    result = pd.concat(parts, ignore_index=True)
    validate_result(result)
    return result
