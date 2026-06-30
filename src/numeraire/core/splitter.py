"""Walk-forward splitters. Yields ``(train, test)`` views, PIT-aware.

The splitter only uses the :class:`~numeraire.core.protocols.DataView` calendar plus
``between``, so it is shape-agnostic. The horizon purge (a train fold's targets never reach
into its test fold) is enforced by the view's :meth:`~numeraire.core.data.TimeSeriesView.aligned`,
which drops any feature whose ``(t, t+h]`` target is not realized by the train cutoff;
``embargo`` adds an optional extra gap on top for serial-correlation safety.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from numeraire.core.data import TimeSeriesView


@dataclass(frozen=True)
class WalkForwardSplitter:
    """Expanding- or rolling-window walk-forward splitter.

    Parameters
    ----------
    min_train:
        Minimum number of calendar observations in the first train fold.
    test_size:
        Number of calendar observations per test fold (also the step between folds).
    expanding:
        ``True`` → train grows from the start each fold; ``False`` → rolling window of
        ``min_train`` observations.
    embargo:
        Extra calendar observations to drop between the train cutoff and the first test date,
        on top of the automatic horizon purge. Default ``0``.
    """

    min_train: int = 60
    test_size: int = 12
    expanding: bool = True
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.min_train < 1:
            raise ValueError("min_train must be >= 1")
        if self.test_size < 1:
            raise ValueError("test_size must be >= 1")
        if self.embargo < 0:
            raise ValueError("embargo must be >= 0")

    def split(self, view: TimeSeriesView) -> Iterator[tuple[TimeSeriesView, TimeSeriesView]]:
        """Yield ``(train, test)`` view pairs; each test calendar is strictly future of train."""
        cal = view.calendar
        n = len(cal)
        i = self.min_train
        while i + self.embargo + self.test_size <= n:
            cutoff = cal[i - 1]  # last training observation
            train_lo = 0 if self.expanding else i - self.min_train
            train_lb = _lower_bound(cal, train_lo)
            train = view.between(train_lb, cutoff)

            test_start = cal[i - 1 + self.embargo]
            test_end = cal[i - 1 + self.embargo + self.test_size]
            test = view.between(test_start, test_end)
            yield train, test
            i += self.test_size


def _lower_bound(cal: pd.DatetimeIndex, lo: int) -> pd.Timestamp:
    """Timestamp strictly below ``cal[lo]`` so ``between(lb, ...)`` includes ``cal[lo]``."""
    first: pd.Timestamp = cal[0]
    if lo == 0:
        return first - timedelta(days=1)
    return cal[lo - 1]
