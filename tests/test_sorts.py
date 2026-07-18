"""Portfolio-sort formation, label alignment, aggregation, and fail-closed guards.

The most important regression checks freeze memberships independently of realized returns and
permute every auxiliary frame. These tests bite on the old implementation, which let missing
holding-period returns alter formation cutoffs and consumed masks and weights positionally.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from numeraire.core.sorts import (
    SortAssignments,
    aggregate_assigned_portfolios,
    assign_portfolio_bins,
    sort_portfolios,
)


def _wide(
    rows: dict[str, list[float] | list[bool] | list[float | bool | None]],
    dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    return pd.DataFrame(rows, index=dates)


def test_two_bin_split_and_equal_weight_returns() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}, dates)
    returns = _wide({"a": [0.01], "b": [0.03], "c": [0.05], "d": [0.09]}, dates)

    result = sort_portfolios(signal, returns, n_bins=2)

    assert result.portfolios.loc[dates[0], 0] == np.mean([0.01, 0.03])
    assert result.portfolios.loc[dates[0], 1] == np.mean([0.05, 0.09])
    assert result.long_short.iloc[0] == np.mean([0.05, 0.09]) - np.mean([0.01, 0.03])
    assert list(result.counts.loc[dates[0]]) == [2, 2]


def test_formation_is_independent_of_missing_realized_returns() -> None:
    """A missing future return must not alter the breakpoint or formation membership."""
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}, dates)
    complete = _wide({"a": [0.01], "b": [0.03], "c": [0.05], "d": [0.09]}, dates)
    missing = complete.copy()
    missing.loc[dates[0], "b"] = np.nan

    complete_result = sort_portfolios(signal, complete, n_bins=2)
    missing_result = sort_portfolios(signal, missing, n_bins=2)

    pd.testing.assert_frame_equal(missing_result.counts, complete_result.counts)
    assert list(missing_result.counts.loc[dates[0]]) == [2, 2]
    assert missing_result.portfolios.loc[dates[0], 0] == complete.loc[dates[0], "a"]


def test_assignment_and_aggregation_are_explicit_seams() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}, dates)
    returns = _wide({"a": [0.01], "b": [np.nan], "c": [np.nan], "d": [np.nan]}, dates)

    assignments = assign_portfolio_bins(signal, n_bins=2)
    result = aggregate_assigned_portfolios(assignments, returns)

    assert isinstance(assignments, SortAssignments)
    assert assignments.breakpoints.loc[dates[0], 1] == 2.5
    assert list(assignments.bins.loc[dates[0]]) == [0.0, 0.0, 1.0, 1.0]
    assert list(result.counts.loc[dates[0]]) == [2, 2]
    assert result.portfolios.loc[dates[0], 0] == 0.01
    assert np.isnan(result.portfolios.loc[dates[0], 1])
    assert np.isnan(result.long_short.loc[dates[0]])


def test_value_weighting_uses_weights() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}, dates)
    returns = _wide({"a": [0.0], "b": [0.10], "c": [0.0], "d": [0.10]}, dates)
    weights = _wide({"a": [1.0], "b": [9.0], "c": [1.0], "d": [9.0]}, dates)

    result = sort_portfolios(signal, returns, n_bins=2, weights=weights)

    assert result.portfolios.loc[dates[0], 0] == 0.09


def test_nonpositive_or_missing_weights_do_not_fall_back_to_equal_weight() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}, dates)
    returns = _wide({"a": [0.0], "b": [0.10], "c": [0.0], "d": [0.10]}, dates)
    weights = _wide({"a": [0.0], "b": [np.nan], "c": [-1.0], "d": [0.0]}, dates)

    result = sort_portfolios(signal, returns, n_bins=2, weights=weights)

    assert list(result.counts.loc[dates[0]]) == [2, 2]
    assert result.portfolios.loc[dates[0]].isna().all()
    assert np.isnan(result.long_short.loc[dates[0]])


def test_auxiliary_frames_align_by_labels_not_position() -> None:
    dates = pd.date_range("2000-01-31", periods=2, freq="ME")
    signal = _wide(
        {"a": [1.0, 4.0], "b": [2.0, 3.0], "c": [3.0, 2.0], "d": [4.0, 1.0]},
        dates,
    )
    returns = _wide(
        {"a": [0.00, 0.04], "b": [0.10, 0.03], "c": [0.02, 0.02], "d": [0.08, 0.01]},
        dates,
    )
    weights = _wide(
        {"a": [1.0, 8.0], "b": [9.0, 3.0], "c": [2.0, 6.0], "d": [8.0, 1.0]},
        dates,
    )
    universe = _wide(
        {"a": [True, True], "b": [True, False], "c": [False, True], "d": [False, False]},
        dates,
    )

    expected = sort_portfolios(
        signal,
        returns,
        n_bins=2,
        weights=weights,
        breakpoint_universe=universe,
    )
    reversed_axes = (list(reversed(dates)), list(reversed(signal.columns)))
    actual = sort_portfolios(
        signal,
        returns,
        n_bins=2,
        weights=weights.reindex(index=reversed_axes[0], columns=reversed_axes[1]),
        breakpoint_universe=universe.reindex(index=reversed_axes[0], columns=reversed_axes[1]),
    )

    pd.testing.assert_frame_equal(actual.portfolios, expected.portfolios)
    pd.testing.assert_frame_equal(actual.counts, expected.counts)
    pd.testing.assert_series_equal(actual.long_short, expected.long_short)


def test_realized_returns_align_by_labels_after_formation() -> None:
    dates = pd.date_range("2000-01-31", periods=2, freq="ME")
    signal = _wide(
        {"a": [1.0, 4.0], "b": [2.0, 3.0], "c": [3.0, 2.0], "d": [4.0, 1.0]},
        dates,
    )
    returns = _wide(
        {"a": [0.01, 0.04], "b": [0.02, 0.03], "c": [0.03, 0.02], "d": [0.04, 0.01]},
        dates,
    )
    assignments = assign_portfolio_bins(signal, n_bins=2)

    expected = aggregate_assigned_portfolios(assignments, returns)
    actual = aggregate_assigned_portfolios(
        assignments,
        returns.reindex(index=list(reversed(dates)), columns=list(reversed(returns.columns))),
    )

    pd.testing.assert_frame_equal(actual.portfolios, expected.portfolios)
    pd.testing.assert_series_equal(actual.long_short, expected.long_short)


def test_direction_flips_long_short_sign() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}, dates)
    returns = _wide({"a": [0.01], "b": [0.02], "c": [0.03], "d": [0.10]}, dates)
    up = sort_portfolios(signal, returns, n_bins=2, direction=1).long_short.iloc[0]
    down = sort_portfolios(signal, returns, n_bins=2, direction=-1).long_short.iloc[0]
    assert up == -down


def test_breakpoint_subset_sets_cutoffs_applied_to_all_eligible_names() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    columns = ["nyse1", "nyse2", "s1", "s2", "s3", "s4"]
    signal = _wide(
        {
            column: [value]
            for column, value in zip(
                columns,
                [100.0, 110.0, 1.0, 2.0, 3.0, 4.0],
                strict=True,
            )
        },
        dates,
    )
    returns = _wide({column: [0.0] for column in columns}, dates)
    universe = _wide({column: [column.startswith("nyse")] for column in columns}, dates)

    result = sort_portfolios(signal, returns, n_bins=2, breakpoint_universe=universe)

    assert list(result.counts.loc[dates[0]]) == [5, 1]


def test_empty_or_thin_breakpoint_universe_fails_closed() -> None:
    dates = pd.date_range("2000-01-31", periods=2, freq="ME")
    signal = _wide(
        {"a": [1.0, 1.0], "b": [2.0, 2.0], "c": [3.0, 3.0], "d": [4.0, 4.0]},
        dates,
    )
    empty = pd.DataFrame(False, index=dates, columns=signal.columns)
    thin = empty.copy()
    thin.loc[dates[0], "a"] = True
    thin.loc[dates[1], ["a", "b"]] = True

    with pytest.raises(ValueError, match="breakpoint universe has 0"):
        assign_portfolio_bins(signal, n_bins=2, breakpoint_universe=empty)
    with pytest.raises(ValueError, match="breakpoint universe has 1"):
        assign_portfolio_bins(signal, n_bins=2, breakpoint_universe=thin)


def test_degenerate_breakpoint_signals_fail_closed() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [1.0], "c": [1.0], "d": [2.0]}, dates)
    universe = _wide({"a": [True], "b": [True], "c": [True], "d": [False]}, dates)

    with pytest.raises(ValueError, match="breakpoint universe has 1 distinct signal values"):
        assign_portfolio_bins(signal, n_bins=2, breakpoint_universe=universe)


def test_breakpoint_ties_that_collapse_a_requested_bin_fail_closed() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide(
        {"a": [0.0], "b": [0.0], "c": [0.0], "d": [0.0], "e": [1.0], "f": [2.0]},
        dates,
    )

    with pytest.raises(ValueError, match=r"ties .* leave at least one requested bin empty"):
        assign_portfolio_bins(signal, n_bins=3)


def test_insufficient_all_name_breakpoint_sample_fails_closed() -> None:
    dates = pd.date_range("2000-01-31", periods=2, freq="ME")
    signal = _wide({"a": [1.0, 1.0], "b": [np.nan, 2.0]}, dates)

    with pytest.raises(ValueError, match="breakpoint universe has 1"):
        assign_portfolio_bins(signal, n_bins=2)


def test_nullable_masks_use_false_for_missing_and_support_eligibility() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}, dates)
    eligibility = _wide({"a": [1.0], "b": [1.0], "c": [np.nan], "d": [0.0]}, dates)

    assignments = assign_portfolio_bins(signal, n_bins=2, eligibility=eligibility)

    assert list(assignments.bins.loc[dates[0], ["a", "b"]]) == [0.0, 1.0]
    assert assignments.bins.loc[dates[0], ["c", "d"]].isna().all()


def test_missing_breakpoint_universe_values_mean_false() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [100.0], "d": [200.0]}, dates)
    universe = _wide({"a": [1.0], "b": [1.0], "c": [np.nan], "d": [0.0]}, dates)

    assignments = assign_portfolio_bins(signal, n_bins=2, breakpoint_universe=universe)

    assert assignments.breakpoints.loc[dates[0], 1] == 1.5
    assert list(assignments.bins.loc[dates[0]]) == [0.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("argument", ["breakpoint_universe", "eligibility"])
def test_masks_reject_infinity(argument: str) -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0]}, dates)
    mask = _wide({"a": [1.0], "b": [1.0], "c": [np.inf]}, dates)

    kwargs = {argument: mask}
    with pytest.raises(ValueError, match=f"{argument} must not contain infinite values"):
        assign_portfolio_bins(signal, n_bins=2, **kwargs)


def test_masks_reject_values_other_than_boolean_or_zero_one() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0]}, dates)
    universe = _wide({"a": [1.0], "b": [1.0], "c": [2.0]}, dates)

    with pytest.raises(ValueError, match="only boolean/0-1 values"):
        assign_portfolio_bins(signal, n_bins=2, breakpoint_universe=universe)


def test_infinite_signal_return_or_weight_is_rejected() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0]}, dates)
    returns = _wide({"a": [0.01], "b": [0.02]}, dates)

    bad_signal = signal.copy()
    bad_signal.loc[dates[0], "b"] = np.inf
    with pytest.raises(ValueError, match="signal must not contain infinite values"):
        assign_portfolio_bins(bad_signal, n_bins=2)

    assignments = assign_portfolio_bins(signal, n_bins=2)
    bad_returns = returns.copy()
    bad_returns.loc[dates[0], "b"] = -np.inf
    with pytest.raises(ValueError, match="returns must not contain infinite values"):
        aggregate_assigned_portfolios(assignments, bad_returns)

    bad_weights = pd.DataFrame(1.0, index=dates, columns=signal.columns)
    bad_weights.loc[dates[0], "a"] = np.inf
    with pytest.raises(ValueError, match="weights must not contain infinite values"):
        aggregate_assigned_portfolios(assignments, returns, weights=bad_weights)


def test_unique_axes_and_equal_label_sets_are_required() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0]}, dates)
    returns = _wide({"a": [0.01], "b": [0.02]}, dates)
    duplicate_signal = pd.DataFrame([[1.0, 2.0]], index=dates, columns=["a", "a"])

    with pytest.raises(ValueError, match="signal column labels must be unique"):
        assign_portfolio_bins(duplicate_signal, n_bins=2)

    assignments = assign_portfolio_bins(signal, n_bins=2)
    with pytest.raises(ValueError, match=r"returns and assignments\.bins must have the same"):
        aggregate_assigned_portfolios(
            assignments,
            returns.rename(columns={"b": "z"}),
        )

    duplicate_weights = pd.DataFrame([[1.0, 1.0]], index=dates, columns=["a", "a"])
    with pytest.raises(ValueError, match="weights column labels must be unique"):
        aggregate_assigned_portfolios(assignments, returns, weights=duplicate_weights)

    duplicate_universe = pd.DataFrame([[True, False]], index=dates, columns=["a", "a"])
    with pytest.raises(ValueError, match="breakpoint_universe column labels must be unique"):
        assign_portfolio_bins(signal, n_bins=2, breakpoint_universe=duplicate_universe)

    duplicate_returns = pd.concat([returns, returns])
    with pytest.raises(ValueError, match="returns index labels must be unique"):
        aggregate_assigned_portfolios(assignments, duplicate_returns)


def test_assignment_validation_rejects_invalid_bin_labels() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    returns = _wide({"a": [0.01], "b": [0.02]}, dates)
    invalid = SortAssignments(
        bins=_wide({"a": [0.5], "b": [2.0]}, dates),
        breakpoints=pd.DataFrame([[1.5]], index=dates, columns=[1]),
        n_bins=2,
    )

    with pytest.raises(ValueError, match="integer labels"):
        aggregate_assigned_portfolios(invalid, returns)


def test_validation_guards() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0]}, dates)
    returns = _wide({"a": [0.01], "b": [0.02]}, dates)

    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        sort_portfolios(signal, returns, n_bins=1)
    with pytest.raises(ValueError, match="direction must be"):
        sort_portfolios(signal, returns, n_bins=2, direction=0)
