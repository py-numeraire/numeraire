"""Weight-stream portfolio simulator: drift-and-rebalance accounting, conventions explicit.

Given target weights issued at decision (signal) dates and asset returns at data frequency, the
simulator produces the realized gross/net portfolio return stream plus per-rebalance turnover and
costs. Published papers differ on the exact turnover and cost conventions, so every convention here
is an explicit named parameter, never an implicit default buried in the accounting:

- **Timing.** A weight decided at signal date ``s`` first earns the return realized over the data
  period *strictly after* ``s`` (the same no-look-ahead pairing the OOS engine uses).
- **Two accounting modes.** ``target_weight`` (default): the target is applied to every data row of
  its holding span — the constant-mix convention of academic horse races (equivalent to cost-free
  re-trading back to target each row; with one data row per rebalance, the common monthly case,
  the two modes' gross returns coincide). ``drifted_holdings``: weights drift with realized returns
  between rebalances (begin-of-period weights = prior end-of-period weights unless a rebalance
  overrides — the ``Return.portfolio`` behavioral reference).
- **Turnover** is L1 against the **pre-trade drifted** weights, ``sum_i |target_i - drifted_i|``
  (buys plus sells; halve for the long-only one-way figure). The first rebalance trades out of
  cash, so its turnover is the full initial funding.
- **Costs** are proportional (``cost_bps`` per unit of L1 turnover), charged once per rebalance to
  NAV before the holding-period return: ``net = (1 - cost) * (1 + gross) - 1`` on the trade row.
- **Cash** is the residual ``1 - sum(w)`` and earns ``rf`` (zero if not given); negative cash
  borrows at the same rate (no spread in v1).
- **Missing returns** on a held asset follow ``missing``: ``"error"`` (default — silent zeros bias
  large panels) or ``"zero"``.
- **Normalization** of incoming targets is a policy, not automatic: ``"none"`` (default — academic
  long-short books are intentionally net-0/gross-2), ``"net_budget"`` or ``"gross_budget"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from numeraire.core.data import Float

_MODES = ("target_weight", "drifted_holdings")
_MISSING = ("error", "zero")
_NORMALIZE = ("none", "net_budget", "gross_budget")


@dataclass(frozen=True)
class RebalanceSchedule:
    """Decision (signal) dates mapped to the half-open data-row spans they govern.

    ``spans[k] = (lo, hi)`` means the ``k``-th signal's target holds over data rows
    ``lo..hi-1`` — ``lo`` is the first return row strictly after the signal date, ``hi`` is the
    next signal's first row (or the end of data). Decouples the decision calendar from the data
    frequency (e.g. month-end decisions over daily returns).
    """

    data_calendar: pd.DatetimeIndex
    signal_dates: pd.DatetimeIndex
    spans: tuple[tuple[int, int], ...]

    @classmethod
    def from_signals(
        cls, data_calendar: pd.DatetimeIndex, signal_dates: pd.DatetimeIndex
    ) -> RebalanceSchedule:
        """Schedule from explicit signal dates (each trades on the next data row after it)."""
        if not data_calendar.is_monotonic_increasing or not data_calendar.is_unique:
            raise ValueError("data_calendar must be sorted and unique")
        if not signal_dates.is_monotonic_increasing or not signal_dates.is_unique:
            raise ValueError("signal_dates must be sorted and unique")
        starts = np.asarray(data_calendar.searchsorted(signal_dates, side="right"), dtype=np.int64)
        if len(starts) > 1 and (np.diff(starts) == 0).any():
            raise ValueError(
                "two signal dates map to the same first data row; "
                "at most one decision per data period"
            )
        n = len(data_calendar)
        keep = starts < n  # a signal after the last return row has nothing to earn
        kept = pd.DatetimeIndex(signal_dates[keep])
        starts = starts[keep]
        if len(starts) == 0:
            raise ValueError("no signal date is followed by any data row")
        ends = np.append(starts[1:], n)
        spans = tuple((int(lo), int(hi)) for lo, hi in zip(starts, ends, strict=True))
        return cls(data_calendar=data_calendar, signal_dates=kept, spans=spans)

    @classmethod
    def from_rule(
        cls, data_calendar: pd.DatetimeIndex, rule: str = "month_end"
    ) -> RebalanceSchedule:
        """Derive signal dates from the data calendar (``month_end``: last data date per month)."""
        if rule != "month_end":
            raise ValueError(f"unknown rule {rule!r}; supported: 'month_end'")
        by_month = pd.Series(np.arange(len(data_calendar)), index=data_calendar)
        last_rows = by_month.groupby(data_calendar.to_period("M")).max().to_numpy()
        signals = pd.DatetimeIndex(data_calendar[np.asarray(last_rows, dtype=np.int64)])
        return cls.from_signals(data_calendar, signals)


@dataclass(frozen=True)
class SimulationResult:
    """Realized simulation output plus the accounting provenance every result row needs."""

    gross: pd.Series
    net: pd.Series
    turnover: pd.Series
    costs: pd.Series
    mode: str
    meta: dict[str, Any] = field(default_factory=dict)


def _normalized(w: Float, policy: str) -> Float:
    if policy == "none":
        return w
    total = float(w.sum()) if policy == "net_budget" else float(np.abs(w).sum())
    if abs(total) < 1e-12:
        raise ValueError(f"cannot apply {policy!r}: weight sum is ~0")
    return w / total


def simulate_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    schedule: RebalanceSchedule | None = None,
    rf: pd.Series | None = None,
    cost_bps: float = 0.0,
    mode: str = "target_weight",
    normalize: str = "none",
    missing: str = "error",
) -> SimulationResult:
    """Simulate a target-weight stream over data-frequency returns (conventions in module doc).

    Parameters
    ----------
    returns:
        ``(date x asset)`` simple returns at data frequency.
    weights:
        ``(signal_date x asset)`` target weights at decision dates. Columns must be a subset of
        the returns columns (missing assets are held at 0). With no ``schedule``, the weights
        index *is* the decision calendar.
    schedule:
        Optional decoupled rebalance calendar; the weights index must then equal its
        ``signal_dates``.
    rf:
        Per-period risk-free return on the data calendar (cash leg); default 0.
    cost_bps:
        Proportional cost in basis points per unit of L1 turnover, charged at each rebalance.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}; got {mode!r}")
    if missing not in _MISSING:
        raise ValueError(f"missing must be one of {_MISSING}; got {missing!r}")
    if normalize not in _NORMALIZE:
        raise ValueError(f"normalize must be one of {_NORMALIZE}; got {normalize!r}")
    index = returns.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex")
    sched = (
        schedule
        if schedule is not None
        else RebalanceSchedule.from_signals(index, pd.DatetimeIndex(weights.index))
    )
    if not pd.DatetimeIndex(weights.index).equals(sched.signal_dates):
        weights = weights.reindex(sched.signal_dates)
        if weights.isna().to_numpy().any():
            raise ValueError("weights are missing rows for some schedule signal dates")
    extra = [c for c in weights.columns if c not in returns.columns]
    if extra:
        raise ValueError(f"weights carry assets absent from returns: {extra}")
    w_all = weights.reindex(columns=returns.columns).fillna(0.0).to_numpy(dtype=np.float64)
    r_all = returns.to_numpy(dtype=np.float64)
    if rf is None:
        rf_all = np.zeros(len(index), dtype=np.float64)
    else:
        rf_re = rf.reindex(index)
        if rf_re.isna().to_numpy().any():
            raise ValueError("rf is missing values for some return dates")
        rf_all = rf_re.to_numpy(dtype=np.float64)

    cost_rate = cost_bps / 1e4
    lo0 = sched.spans[0][0]
    n_rows = len(index) - lo0
    gross = np.full(n_rows, np.nan, dtype=np.float64)
    net = np.full(n_rows, np.nan, dtype=np.float64)
    turnover = np.zeros(len(sched.spans), dtype=np.float64)
    costs = np.zeros(len(sched.spans), dtype=np.float64)

    state = np.zeros(r_all.shape[1], dtype=np.float64)  # drifted risky weights (starts in cash)
    for k, (lo, hi) in enumerate(sched.spans):
        target = _normalized(w_all[k], normalize)
        turnover[k] = float(np.abs(target - state).sum())
        costs[k] = cost_rate * turnover[k]
        state = target.copy()
        for i in range(lo, hi):
            held = target if mode == "target_weight" else state
            cash_held = 1.0 - float(held.sum())
            r_i = r_all[i]
            bad = np.isnan(r_i) & (held != 0.0)
            if bad.any():
                if missing == "error":
                    names = [str(c) for c in returns.columns[np.flatnonzero(bad)]]
                    raise ValueError(
                        f"missing return for held asset(s) {names} at {index[i]}; "
                        "set missing='zero' to impute (biased) or handle delistings upstream"
                    )
                r_i = np.where(np.isnan(r_i), 0.0, r_i)
            r_i = np.where(np.isnan(r_i), 0.0, r_i)  # unheld nan contributes nothing
            port_r = float(held @ r_i) + cash_held * float(rf_all[i])
            if 1.0 + port_r <= 0.0:
                raise ValueError(f"portfolio wealth wiped out at {index[i]} (return {port_r:.4f})")
            gross[i - lo0] = port_r
            net[i - lo0] = (1.0 - costs[k]) * (1.0 + port_r) - 1.0 if i == lo else port_r
            # drift: end-of-period weights (constant-mix drifts one row from target each row);
            # the cash leg needs no explicit state — it is always 1 - sum(weights)
            state = held * (1.0 + r_i) / (1.0 + port_r)

    out_index = index[lo0:]
    sig = sched.signal_dates
    return SimulationResult(
        gross=pd.Series(gross, index=out_index, name="gross_return"),
        net=pd.Series(net, index=out_index, name="net_return"),
        turnover=pd.Series(turnover, index=sig, name="turnover_l1"),
        costs=pd.Series(costs, index=sig, name="cost"),
        mode=mode,
        meta={
            "cost_bps": cost_bps,
            "normalize": normalize,
            "missing": missing,
            "turnover_convention": "l1_pretrade_drift",
            "cost_timing": "at_rebalance_before_period_return",
            "cash": "rf" if rf is not None else "zero",
            "initial": "from_cash",
        },
    )
