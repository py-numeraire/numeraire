"""The economic-value family: certainty-equivalent, return-loss, and performance fee.

Covers the DGU (2009) / Fleming-Kirby-Ostdiek utility metrics promoted into ``core.stats`` and the
``CEQEvaluator`` wrapper. Assertions are identities/invariants (scaling invariance of return-loss,
zero self-fee, CEQ hand value) plus the evaluator's schema conformance and registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.core.evaluators import CEQEvaluator
from numeraire.core.registry import get_evaluator
from numeraire.core.schema import validate_result
from numeraire.core.stats import certainty_equivalent, performance_fee, return_loss


def _weights_output(strategy: pd.Series):
    """A minimal WeightsOutput-like stub exposing strategy_returns() + provenance."""
    from numeraire.core.engine import WeightsOutput

    idx = strategy.index
    weights = pd.DataFrame(1.0, index=idx, columns=["a"])
    realized = pd.DataFrame(strategy.to_numpy(), index=idx, columns=["a"])
    return WeightsOutput(
        weights=weights,
        realized=realized,
        method="stub",
        run_id="stub-1",
        config_hash="deadbeef",
        data_vintage="v0",
    )


# --------------------------------------------------------------------------------- CEQ


def test_ceq_matches_hand_value() -> None:
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.04])
    expected = r.mean() - 0.5 * 1.0 * r.var(ddof=0)
    assert certainty_equivalent(r, gamma=1.0) == pytest.approx(expected)


def test_ceq_gamma_zero_is_mean_and_decreasing_in_gamma() -> None:
    r = np.array([0.02, -0.01, 0.05, 0.01])
    assert certainty_equivalent(r, gamma=0.0) == pytest.approx(r.mean())
    assert certainty_equivalent(r, gamma=1.0) > certainty_equivalent(r, gamma=5.0)


def test_ceq_drops_nan_and_guards_short() -> None:
    r = np.array([0.01, np.nan, 0.03])
    assert certainty_equivalent(r) == pytest.approx(certainty_equivalent(np.array([0.01, 0.03])))
    assert np.isnan(certainty_equivalent(np.array([0.01])))


# --------------------------------------------------------------------------------- return-loss


def test_return_loss_zero_for_scaled_benchmark() -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(0.01, 0.04, 240)
    # a positively-scaled copy has the same Sharpe -> zero return-loss (DGU sign convention)
    assert return_loss(2.5 * b, b) == pytest.approx(0.0, abs=1e-12)
    assert return_loss(b, b) == pytest.approx(0.0, abs=1e-12)


def test_return_loss_positive_when_lower_sharpe() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(0.01, 0.03, 500)
    # same mean, higher variance => worse Sharpe => positive return-loss
    noisier = (b - b.mean()) * 2.0 + b.mean()
    assert return_loss(noisier, b) > 0.0


def test_return_loss_shape_guard() -> None:
    with pytest.raises(ValueError, match="aligned 1-D"):
        return_loss(np.zeros(3), np.zeros(4))


# --------------------------------------------------------------------------------- performance fee


def test_performance_fee_zero_for_self() -> None:
    rng = np.random.default_rng(2)
    r = rng.normal(0.008, 0.04, 300)
    assert performance_fee(r, r, gamma=5.0) == pytest.approx(0.0, abs=1e-15)


def test_performance_fee_positive_for_dominating_candidate() -> None:
    rng = np.random.default_rng(3)
    b = rng.normal(0.004, 0.05, 400)
    better = b + 0.003  # uniformly higher return, same risk
    assert performance_fee(better, b, gamma=5.0) > 0.0


def test_performance_fee_shape_guard() -> None:
    with pytest.raises(ValueError, match="aligned 1-D"):
        performance_fee(np.zeros(3), np.zeros(5), gamma=1.0)


# --------------------------------------------------------------------------------- CEQ evaluator


def test_ceq_evaluator_matches_stat_and_schema() -> None:
    idx = pd.date_range("2000-01-31", periods=36, freq="ME")
    rng = np.random.default_rng(7)
    strat = pd.Series(rng.normal(0.01, 0.03, 36), index=idx)
    out = _weights_output(strat)
    df = CEQEvaluator(gamma=1.0).evaluate(out)
    validate_result(df)
    assert list(df["metric"]) == ["ceq"]
    realized_strat = out.strategy_returns().to_numpy(dtype=np.float64)
    assert df["value"].iloc[0] == pytest.approx(certainty_equivalent(realized_strat, 1.0))


def test_ceq_evaluator_registered_and_typed() -> None:
    assert get_evaluator("ceq") is not None
    with pytest.raises(TypeError, match="WeightsOutput"):
        CEQEvaluator().evaluate(object())
