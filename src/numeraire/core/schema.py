"""The standard tidy, long-format result schema.

Every evaluator emits rows in this schema; downstream plotting (plotnine / R) consumes it,
so the plotting choice stays decoupled. Stability is promised on this schema (semver).
"""

from __future__ import annotations

import pandas as pd

RESULT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "method",
    "date",
    "metric",
    "value",
    "universe",
    "capability",
    "protocol",
    "config_hash",
    "data_vintage",
)
"""Minimum columns every result table must carry (in any order).

``protocol`` labels the evaluation discipline the row was produced under: ``"walk_forward"`` (the
framework's out-of-sample walk-forward path, which every weights/forecast evaluator emits) or
``"in_sample"`` (a single full-sample fit, the paper cross-sectional-pricing tradition). It makes an
explanatory in-sample number unconfusable with an out-of-sample one.
"""


def validate_result(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` is missing any required result-schema column.

    Extra columns are allowed; only the presence of :data:`RESULT_COLUMNS` is enforced.
    """
    present = {str(c) for c in df.columns}
    missing = [c for c in RESULT_COLUMNS if c not in present]
    if missing:
        raise ValueError(f"result table missing required columns: {missing}")
