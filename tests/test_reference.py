"""The ReferenceResult registry: tiering, band checks, and tier-gated skip.

Covers the three data-access tiers (public / credentialed / restricted): a public case always runs,
and a credentialed / restricted case self-skips through ``reference_params`` when its data is
unreachable. The restricted exemplar is modelled on the JKP 2023 replication-rate reference — its
exact counts need non-redistributable CC-BY-NC returns, so it is registered as a tier that skips in
CI yet runs verbatim wherever the local file is present (the connector pattern, one code path).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from numeraire.reference import (
    CREDENTIALED,
    PUBLIC,
    RESTRICTED,
    VERIFIED_WITH_CAVEAT,
    ReferenceResult,
    clear_references,
    get_reference,
    reference_params,
    references,
    register_reference,
)

# --------------------------------------------------------------------------------- demo registry
# Registered at import so reference_params() captures them at collection. Names are unique to this
# module; the restore_registry fixture keeps mutation tests from leaking into this set.

FF2015 = register_reference(
    ReferenceResult(
        name="demo-ff2015-grs",
        paper="Fama & French",
        venue="JFE",
        year=2015,
        table="Table 5",
        expected={"grs": 2.84, "avg_abs_alpha": 0.094},
        tolerance={"grs": 0.15, "avg_abs_alpha": 0.01},
        tier=PUBLIC,
        data="Ken French FF5 + 25 Size-B/M, 1963-07..2013-12",
    )
)

CRSP_DEMO = register_reference(
    ReferenceResult(
        name="demo-wrds-anomaly",
        paper="Placeholder",
        venue="JF",
        year=2020,
        table="Table 1",
        expected={"hl_tstat": 3.0},
        tolerance={"hl_tstat": 0.5},
        tier=CREDENTIALED,
        data="CRSP VW decile spread (needs WRDS credentials)",
        available=lambda: bool(os.environ.get("WRDS_USERNAME")),
    )
)


def _jkp_returns_present() -> bool:
    """Restricted-tier availability: the JKP returns file at an env-pointed local path."""
    return Path(os.environ.get("NUMERAIRE_JKP_RETURNS", "")).is_file()


JKP = register_reference(
    ReferenceResult(
        name="demo-jkp2023-us-capm-replication",
        paper="Jensen, Kelly & Pedersen",
        venue="JF",
        year=2023,
        table="Table I",
        expected={"replication_rate": 0.824, "n_significant": 98.0, "tau_c": 0.0035},
        tolerance={"replication_rate": 0.02, "n_significant": 3.0, "tau_c": 0.0005},
        tier=RESTRICTED,
        data="published US factor returns from jkpfactors.com (CC-BY-NC; local, uncommitted)",
        status=VERIFIED_WITH_CAVEAT,
        notes="exact 153<->119 factor mapping contested; machinery + synthetic invariants in zoo",
        available=_jkp_returns_present,
    )
)


@pytest.fixture(autouse=True)
def restore_registry():
    """Snapshot the global registry and restore it after each test (mutation isolation)."""
    from numeraire import reference

    saved = dict(reference._CASES)
    try:
        yield
    finally:
        reference._CASES.clear()
        reference._CASES.update(saved)


# --------------------------------------------------------------------------------- check() bands


def test_check_passes_within_band() -> None:
    FF2015.check({"grs": 2.90, "avg_abs_alpha": 0.090, "extra": 1.0})  # extra keys ignored


def test_check_fails_out_of_band() -> None:
    with pytest.raises(AssertionError, match="grs"):
        FF2015.check({"grs": 3.10, "avg_abs_alpha": 0.094})


def test_check_fails_on_missing_metric() -> None:
    with pytest.raises(AssertionError, match="missing metric 'avg_abs_alpha'"):
        FF2015.check({"grs": 2.84})


def test_check_rejects_non_finite() -> None:
    with pytest.raises(AssertionError, match="not finite"):
        FF2015.check({"grs": float("nan"), "avg_abs_alpha": 0.094})


def test_zero_band_demands_exact_match() -> None:
    case = ReferenceResult(
        name="demo-count",
        paper="p",
        venue="v",
        year=2020,
        table="t",
        expected={"count": 98.0},  # no tolerance entry -> band 0.0
    )
    case.check({"count": 98.0})
    with pytest.raises(AssertionError):
        case.check({"count": 99.0})


# --------------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tier": "bogus"}, "unknown data tier"),
        ({"status": "great"}, "unknown status"),
        ({"expected": {}}, "at least one metric"),
        ({"tolerance": {"nope": 0.1}}, "non-expected metrics"),
        ({"name": ""}, "must be non-empty"),
        ({"expected": {"r2": 3.51}}, "zero tolerance band but a non-integer"),
    ],
)
def test_post_init_validation(kwargs: dict[str, object], match: str) -> None:
    base = dict(name="x", paper="p", venue="v", year=2020, table="t", expected={"m": 1.0})
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ReferenceResult(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------- registry


def test_register_get_and_duplicate() -> None:
    clear_references()
    case = register_reference(
        ReferenceResult(
            name="uniq", paper="p", venue="v", year=2020, table="t", expected={"m": 1.0}
        )
    )
    assert get_reference("uniq") is case
    with pytest.raises(KeyError, match="already registered"):
        register_reference(case)
    register_reference(case, overwrite=True)  # overwrite allowed


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError, match="no reference result"):
        get_reference("does-not-exist")


def test_references_filter_by_tier_and_availability() -> None:
    names = {c.name for c in references(tier=PUBLIC)}
    assert "demo-ff2015-grs" in names
    assert "demo-jkp2023-us-capm-replication" not in names
    # restricted case is registered but unavailable in CI -> excluded by available_only
    restricted = references(tier=RESTRICTED)
    assert JKP in restricted
    assert JKP not in references(tier=RESTRICTED, available_only=True)
    with pytest.raises(ValueError, match="unknown data tier"):
        references(tier="bogus")


def test_is_available_default_and_predicate() -> None:
    assert FF2015.is_available() is True  # no predicate -> always available
    assert JKP.is_available() is False  # env var unset in CI


# --------------------------------------------------------------------------------- pytest helper


def test_reference_params_marks_unavailable_as_skip() -> None:
    params = {p.id: p for p in reference_params()}
    assert not params["demo-ff2015-grs"].marks  # public: runs
    jkp_marks = params["demo-jkp2023-us-capm-replication"].marks
    assert any(m.name == "skip" for m in jkp_marks)  # restricted: self-skips
    wrds_marks = params["demo-wrds-anomaly"].marks
    assert any(m.name == "skip" for m in wrds_marks)  # credentialed: self-skips


@pytest.mark.parametrize("case", reference_params(tier=PUBLIC))
def test_public_references_end_to_end(case: ReferenceResult) -> None:
    # A public case drives a real check() here; credentialed / restricted cases would self-skip.
    # Feed each expected value back exactly to exercise the band on registered cases.
    case.check(dict(case.expected))
