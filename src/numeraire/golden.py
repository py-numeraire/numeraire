"""GoldenCase registry — reproduction targets as first-class, tiered data records.

A :class:`GoldenCase` pins a *published* result — an exact paper, venue, table, and paper
version — to an ``expected`` metric dict plus a per-metric ``tolerance`` band on a named dataset,
tagged by **data tier**:

- :data:`PUBLIC_CI` — redistributable public/synthetic data; runs in CI unconditionally.
- :data:`WRDS_CRED` — needs licensed data behind the user's own credentials (CRSP/Compustat via a
  connector); skipped unless the credentials/data are reachable.
- :data:`LAB_ONLY` — needs non-redistributable data at a local path (e.g. CC-BY-NC returns that may
  never be committed); skipped unless the file is present.

Needing licensed data does **not** disqualify a reproduction target. The tier plus an optional
``available`` predicate let CI stay green on public data while the *same* case runs verbatim
wherever the private data is present — the connector pattern, one code path, no forked assertions.

This module is **core infrastructure, not a method** — it is exempt from the boundary rule's
methods/adapters import ban (it lives in ``numeraire`` proper, not ``numeraire.core``, and imports
only the standard library). The registry is process-global and open: ``numeraire`` ships a couple of
its own goldens and any downstream package (``numeraire-zoo``, ``numeraire-yourlab``) registers its
reproduction targets the same way, then a single parametrized test drives them all
(:func:`golden_params`).

The tolerance philosophy is the framework's: a golden asserts an **invariant plus a headline scalar
within a band**, never bit-equality — bands absorb data-vintage revisions (French/GW live-data
drift). :meth:`GoldenCase.check` enforces the band and rejects a non-finite computed value (an
all-NaN false green).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from _pytest.mark.structures import ParameterSet

# --------------------------------------------------------------------------------- data tiers

PUBLIC_CI = "PUBLIC-CI"
"""Redistributable public or synthetic data — the case runs in CI unconditionally."""

WRDS_CRED = "WRDS-CRED"
"""Licensed data reachable through the user's own credentials — skipped when absent."""

LAB_ONLY = "LAB-ONLY"
"""Non-redistributable data at a local path (never committed) — skipped when absent."""

DATA_TIERS: tuple[str, ...] = (PUBLIC_CI, WRDS_CRED, LAB_ONLY)
"""The closed set of data tiers, from least to most access-restricted."""

# --------------------------------------------------------------------------------- status values

VERIFIED = "verified"
"""The headline scalar was reproduced within band against the pinned fixture."""

VERIFIED_WITH_CAVEAT = "reproduced-with-caveat"
"""The economics/invariants reproduce but an exact figure is sensitive (documented in ``notes``)."""

UNVERIFIED = "UNVERIFIED"
"""A target recorded but not yet reproduced — carried so the queue is visible."""

_STATUSES: frozenset[str] = frozenset({VERIFIED, VERIFIED_WITH_CAVEAT, UNVERIFIED})


@dataclass(frozen=True)
class GoldenCase:
    """A pinned, tiered reproduction target: a paper figure/table matched within a band.

    ``expected`` maps a metric name to the paper's value; ``tolerance`` maps a (subset of those)
    metric names to an absolute band — a metric absent from ``tolerance`` must match exactly
    (band ``0.0``, only sensible for integer counts). ``available`` is an optional zero-arg
    predicate: when it returns ``False`` the case is skipped (its data is out of reach on this
    machine). ``PUBLIC_CI`` cases normally leave it ``None`` (always available).
    """

    name: str
    paper: str
    venue: str
    year: int
    table: str
    expected: Mapping[str, float]
    tolerance: Mapping[str, float] = field(default_factory=dict[str, float])
    tier: str = PUBLIC_CI
    paper_version: str = "published"
    data: str = ""
    status: str = VERIFIED
    notes: str = ""
    available: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("GoldenCase.name must be non-empty")
        if self.tier not in DATA_TIERS:
            raise ValueError(f"unknown data tier {self.tier!r}; expected one of {DATA_TIERS}")
        if self.status not in _STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {sorted(_STATUSES)}")
        if not self.expected:
            raise ValueError(f"{self.name}: expected must name at least one metric")
        stray = set(self.tolerance) - set(self.expected)
        if stray:
            raise ValueError(f"{self.name}: tolerance names non-expected metrics {sorted(stray)}")

    def is_available(self) -> bool:
        """Whether this case's data is reachable here (``True`` when no predicate is set)."""
        return True if self.available is None else bool(self.available())

    def check(self, computed: Mapping[str, float]) -> None:
        """Assert every ``expected`` metric appears in ``computed`` and lands within its band.

        Raises ``AssertionError`` on the first missing metric, non-finite value (guards against an
        all-NaN false green), or out-of-band deviation. Extra keys in ``computed`` are ignored.
        """
        for metric, target in self.expected.items():
            if metric not in computed:
                raise AssertionError(f"{self.name}: computed result is missing metric {metric!r}")
            got = float(computed[metric])
            if not math.isfinite(got):
                raise AssertionError(f"{self.name} [{metric}]: computed value not finite: {got}")
            band = float(self.tolerance.get(metric, 0.0))
            if abs(got - target) > band:
                raise AssertionError(
                    f"{self.name} [{metric}]: {got:.6g} is not within +/-{band:.6g} "
                    f"of the target {target:.6g} (miss {abs(got - target):.6g})"
                )


# --------------------------------------------------------------------------------- registry

_CASES: dict[str, GoldenCase] = {}


def register_golden_case(case: GoldenCase, *, overwrite: bool = False) -> GoldenCase:
    """Register ``case`` under its ``name``. Raises on a duplicate name unless ``overwrite``.

    Returns the case so a module can register-and-bind in one line
    (``FF2015 = register_golden_case(GoldenCase(...))``).
    """
    if not overwrite and case.name in _CASES:
        raise KeyError(f"golden case {case.name!r} already registered")
    _CASES[case.name] = case
    return case


def get_golden_case(name: str) -> GoldenCase:
    """Return the golden case registered under ``name``."""
    try:
        return _CASES[name]
    except KeyError:
        raise KeyError(f"no golden case registered as {name!r}") from None


def golden_cases(
    *, tier: str | None = None, available_only: bool = False
) -> tuple[GoldenCase, ...]:
    """Return registered cases, name-sorted, optionally filtered by ``tier`` / availability."""
    if tier is not None and tier not in DATA_TIERS:
        raise ValueError(f"unknown data tier {tier!r}; expected one of {DATA_TIERS}")
    cases = (c for c in _CASES.values() if tier is None or c.tier == tier)
    if available_only:
        cases = (c for c in cases if c.is_available())
    return tuple(sorted(cases, key=lambda c: c.name))


def clear_golden_cases() -> None:
    """Drop all registered cases (test-isolation helper; not for production paths)."""
    _CASES.clear()


# --------------------------------------------------------------------------------- pytest helper


def golden_params(*, tier: str | None = None) -> list[ParameterSet]:
    """Return ``pytest.param`` entries for registered cases, ready for ``@pytest.mark.parametrize``.

    Each entry carries the :class:`GoldenCase` as its single argument and the case name as its id;
    a case whose data is unavailable (:meth:`GoldenCase.is_available` is ``False``) carries a
    ``pytest.mark.skip`` so a ``WRDS-CRED`` / ``LAB-ONLY`` target self-skips on a machine that lacks
    the data instead of failing. ``pytest`` is imported lazily so this module stays import-clean at
    runtime (the helper is only ever called from a test).

    Usage::

        import pytest
        from numeraire.golden import golden_params

        @pytest.mark.parametrize("case", golden_params())
        def test_golden(case):
            case.check(compute_metrics(case))
    """
    import pytest

    params: list[ParameterSet] = []
    for case in golden_cases(tier=tier):
        marks = (
            ()
            if case.is_available()
            else (pytest.mark.skip(reason=f"{case.tier} data unavailable for {case.name!r}"),)
        )
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params
