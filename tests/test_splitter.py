"""Unit tests for WalkForwardSplitter."""

from __future__ import annotations

import pytest

from conftest import make_monthly_view
from numeraire.core.protocols import Splitter
from numeraire.core.splitter import WalkForwardSplitter


def test_is_splitter() -> None:
    assert isinstance(WalkForwardSplitter(), Splitter)


def test_expanding_folds_grow() -> None:
    v = make_monthly_view(n=120)
    sp = WalkForwardSplitter(min_train=60, test_size=12, expanding=True)
    folds = list(sp.split(v))
    assert len(folds) == 5  # (120 - 60) / 12
    train_sizes = [len(tr.calendar) for tr, _ in folds]
    assert train_sizes == sorted(train_sizes)  # monotonically growing
    assert train_sizes[0] == 60


def test_rolling_folds_constant_width() -> None:
    v = make_monthly_view(n=120)
    sp = WalkForwardSplitter(min_train=60, test_size=12, expanding=False)
    train_sizes = [len(tr.calendar) for tr, _ in sp.split(v)]
    assert all(s == 60 for s in train_sizes)


def test_test_fold_strictly_future_of_train() -> None:
    v = make_monthly_view(n=120)
    sp = WalkForwardSplitter(min_train=48, test_size=24)
    for train, test in sp.split(v):
        assert test.calendar.min() > train.calendar.max()
        assert len(test.calendar) == 24


def test_embargo_drops_dates_after_cutoff() -> None:
    v = make_monthly_view(n=120)
    base = list(WalkForwardSplitter(min_train=60, test_size=12, embargo=0).split(v))
    emb = list(WalkForwardSplitter(min_train=60, test_size=12, embargo=3).split(v))
    # embargo pushes the first test date later for the first fold
    assert emb[0][1].calendar.min() > base[0][1].calendar.min()


def test_bad_params_rejected() -> None:
    with pytest.raises(ValueError, match="min_train"):
        WalkForwardSplitter(min_train=0)
    with pytest.raises(ValueError, match="test_size"):
        WalkForwardSplitter(test_size=0)
    with pytest.raises(ValueError, match="embargo"):
        WalkForwardSplitter(embargo=-1)
