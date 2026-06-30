"""Spine smoke + conformance tests (public/synthetic data only)."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
import pytest

from numeraire import (
    RESULT_COLUMNS,
    available_evaluators,
    get_evaluator,
    register_evaluator,
    validate_result,
)
from numeraire.core import capabilities
from numeraire.core.protocols import Estimator, Evaluator, Model


class _DummyModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}


class _DummyEstimator:
    def fit(self, view: object) -> _DummyModel:
        return _DummyModel()


class _DummyEvaluator:
    requires: ClassVar[set[str]] = {capabilities.TO_WEIGHTS}

    def evaluate(self, oos_output: object) -> pd.DataFrame:
        return pd.DataFrame({c: [0] for c in RESULT_COLUMNS})


def test_protocols_are_runtime_checkable() -> None:
    assert isinstance(_DummyModel(), Model)
    assert isinstance(_DummyEstimator(), Estimator)
    assert isinstance(_DummyEvaluator(), Evaluator)


def test_result_schema_validation() -> None:
    good = pd.DataFrame({c: [1] for c in RESULT_COLUMNS})
    validate_result(good)  # should not raise
    bad = good.drop(columns=["metric"])
    with pytest.raises(ValueError, match="metric"):
        validate_result(bad)


def test_evaluator_registry_roundtrip() -> None:
    ev = _DummyEvaluator()
    register_evaluator("dummy", ev, overwrite=True)
    assert "dummy" in available_evaluators()
    assert get_evaluator("dummy") is ev
    with pytest.raises(KeyError):
        get_evaluator("does-not-exist")


def test_capabilities_are_strings() -> None:
    assert capabilities.TO_WEIGHTS in capabilities.BUNDLED
    assert all(isinstance(c, str) for c in capabilities.BUNDLED)
