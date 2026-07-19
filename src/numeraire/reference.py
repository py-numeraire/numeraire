"""ReferenceResult registry — reproduction targets as first-class, tiered data records.

A :class:`ReferenceResult` pins a *published* result — an exact paper, venue, table, and paper
version — to an ``expected`` metric dict plus a per-metric ``tolerance`` band on a named dataset,
tagged by a **data-access tier**.

The tier axis is a **DATA-ACCESS REQUIREMENT**, never a statement of importance or rank:

- :data:`PUBLIC` — public/redistributable (or synthetic) data; the case runs unconditionally,
  including in CI.
- :data:`CREDENTIALED` — data that is programmatically fetchable with the user's *own* subscription
  credentials (e.g. CRSP/Compustat through a connector); the case self-skips when those credentials
  are absent.
- :data:`RESTRICTED` — data that anyone may obtain but that is non-redistributable, so it needs a
  self-obtained local copy (e.g. CC-BY-NC returns that may never be committed); the case self-skips
  when that local copy is absent.

Tiers never encode importance or rank — a reproduction that needs licensed or restricted data is a
**first-class citizen**. The tier plus an optional ``available`` predicate let CI stay green on
public data while the *same* case runs verbatim wherever the private data is present — the connector
pattern, one code path, no forked assertions.

(Disambiguation: a "reference result" here is a *pinned published number*; it is unrelated to the
"reference libraries" — ``ipca`` / ``linearmodels`` — mentioned elsewhere in the project.)

This module is **core infrastructure, not a method** — it is exempt from the boundary rule's
methods/adapters import ban (it lives in ``numeraire`` proper, not ``numeraire.core``, and imports
only the standard library). The registry is process-global and open: ``numeraire`` ships a couple of
its own references and any downstream package (``numeraire-zoo``, ``numeraire-yourlab``) registers
its reproduction targets the same way, then a single parametrized test drives them all
(:func:`reference_params`).

The tolerance philosophy is the framework's: a reference asserts an **invariant plus a headline
scalar within a band**, never bit-equality — bands absorb data-vintage revisions (French/GW
live-data drift). :meth:`ReferenceResult.check` enforces the band and rejects a non-finite computed
value (an all-NaN false green).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from _pytest.mark.structures import ParameterSet

# --------------------------------------------------------------------------------- data tiers

PUBLIC = "public"
"""Public/redistributable or synthetic data — the case runs unconditionally, including in CI."""

CREDENTIALED = "credentialed"
"""Programmatically fetchable with the user's own subscription credentials — skipped when absent."""

RESTRICTED = "restricted"
"""Non-redistributable data needing a self-obtained local copy (never committed); skipped absent."""

DATA_TIERS: tuple[str, ...] = (PUBLIC, CREDENTIALED, RESTRICTED)
"""The closed set of data-access tiers, from least to most access-restricted."""

# --------------------------------------------------------------------------------- status values

VERIFIED = "verified"
"""The headline scalar was reproduced within band against the pinned fixture."""

VERIFIED_WITH_CAVEAT = "reproduced-with-caveat"
"""The economics/invariants reproduce but an exact figure is sensitive (documented in ``notes``)."""

UNVERIFIED = "UNVERIFIED"
"""A target recorded but not yet reproduced — carried so the queue is visible."""

_STATUSES: frozenset[str] = frozenset({VERIFIED, VERIFIED_WITH_CAVEAT, UNVERIFIED})


@dataclass(frozen=True)
class ReferenceResult:
    """A pinned, tiered reproduction target: a paper figure/table matched within a band.

    ``expected`` maps a metric name to the paper's value; ``tolerance`` maps a (subset of those)
    metric names to an absolute band — a metric absent from ``tolerance`` must match exactly
    (band ``0.0``, only sensible for integer counts). ``available`` is an optional zero-arg
    predicate: when it returns ``False`` the case is skipped (its data is out of reach on this
    machine). ``PUBLIC`` cases normally leave it ``None`` (always available).
    """

    name: str
    paper: str
    venue: str
    year: int
    table: str
    expected: Mapping[str, float]
    tolerance: Mapping[str, float] = field(default_factory=dict[str, float])
    tier: str = PUBLIC
    paper_version: str = "published"
    data: str = ""
    status: str = VERIFIED
    notes: str = ""
    available: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        # Snapshot the metric mappings into read-only copies before validating: the dataclass is
        # frozen, but a frozen field can still point at the caller's mutable dict — mutating that
        # dict after construction would otherwise bypass every check below.
        object.__setattr__(self, "expected", MappingProxyType(dict(self.expected)))
        object.__setattr__(self, "tolerance", MappingProxyType(dict(self.tolerance)))
        if not self.name:
            raise ValueError("ReferenceResult.name must be non-empty")
        if self.tier not in DATA_TIERS:
            raise ValueError(f"unknown data tier {self.tier!r}; expected one of {DATA_TIERS}")
        if self.status not in _STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {sorted(_STATUSES)}")
        if not self.expected:
            raise ValueError(f"{self.name}: expected must name at least one metric")
        stray = set(self.tolerance) - set(self.expected)
        if stray:
            raise ValueError(f"{self.name}: tolerance names non-expected metrics {sorted(stray)}")
        # A non-finite expected value (NaN/Inf) would auto-pass its own band check — an all-NaN
        # false green pinned at construction. A non-finite or negative tolerance is likewise
        # vacuous (an infinite band accepts anything; a negative band can never be satisfied).
        # Reject both here.
        nonfinite_expected = sorted(
            m for m, target in self.expected.items() if not math.isfinite(float(target))
        )
        if nonfinite_expected:
            raise ValueError(f"{self.name}: expected value(s) {nonfinite_expected} must be finite")
        bad_tol = sorted(
            m
            for m, band in self.tolerance.items()
            if not math.isfinite(float(band)) or float(band) < 0.0
        )
        if bad_tol:
            raise ValueError(f"{self.name}: tolerance(s) {bad_tol} must be finite and non-negative")
        # A zero band demands bit-exact equality, which is only meaningful for an integer target
        # (a count, N, …); a float scalar with no tolerance can never match across data vintages and
        # is almost always a forgotten band — reject it at construction.
        zero_band_floats = [
            m
            for m, target in self.expected.items()
            if float(self.tolerance.get(m, 0.0)) == 0.0 and not float(target).is_integer()
        ]
        if zero_band_floats:
            raise ValueError(
                f"{self.name}: metric(s) {sorted(zero_band_floats)} have a zero tolerance band but "
                f"a non-integer target; give a band, or use an integer target for exact counts"
            )

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

_CASES: dict[str, ReferenceResult] = {}


def register_reference(case: ReferenceResult, *, overwrite: bool = False) -> ReferenceResult:
    """Register ``case`` under its ``name``. Raises on a duplicate name unless ``overwrite``.

    Returns the case so a module can register-and-bind in one line
    (``FF2015 = register_reference(ReferenceResult(...))``).
    """
    if not overwrite and case.name in _CASES:
        raise KeyError(f"reference result {case.name!r} already registered")
    _CASES[case.name] = case
    return case


def get_reference(name: str) -> ReferenceResult:
    """Return the reference result registered under ``name``."""
    try:
        return _CASES[name]
    except KeyError:
        raise KeyError(f"no reference result registered as {name!r}") from None


def references(
    *, tier: str | None = None, available_only: bool = False
) -> tuple[ReferenceResult, ...]:
    """Return registered cases, name-sorted, optionally filtered by ``tier`` / availability."""
    if tier is not None and tier not in DATA_TIERS:
        raise ValueError(f"unknown data tier {tier!r}; expected one of {DATA_TIERS}")
    cases = (c for c in _CASES.values() if tier is None or c.tier == tier)
    if available_only:
        cases = (c for c in cases if c.is_available())
    return tuple(sorted(cases, key=lambda c: c.name))


def clear_references() -> None:
    """Drop all registered cases (test-isolation helper; not for production paths)."""
    _CASES.clear()


# --------------------------------------------------------------------------------- pytest helper


def reference_params(*, tier: str | None = None) -> list[ParameterSet]:
    """Return ``pytest.param`` entries for registered cases, ready for ``@pytest.mark.parametrize``.

    Each entry carries the :class:`ReferenceResult` as its single argument and the case name as its
    id; a case whose data is unavailable (:meth:`ReferenceResult.is_available` is ``False``) carries
    a ``pytest.mark.skip`` so a ``credentialed`` / ``restricted`` target self-skips on a machine
    that lacks the data instead of failing. ``pytest`` is imported lazily so this module stays
    import-clean at runtime (the helper is only ever called from a test).

    Usage::

        import pytest
        from numeraire.reference import reference_params

        @pytest.mark.parametrize("case", reference_params())
        def test_reference(case):
            case.check(compute_metrics(case))
    """
    import pytest

    params: list[ParameterSet] = []
    for case in references(tier=tier):
        marks = (
            ()
            if case.is_available()
            else (pytest.mark.skip(reason=f"{case.tier} data unavailable for {case.name!r}"),)
        )
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params
