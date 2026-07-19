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

ATTRITION_COLUMNS: tuple[str, ...] = ("n_obs", "n_dropped")
"""Optional, schema-additive attrition columns.

Evaluators that compare a model against a benchmark or a realized target (out-of-sample R^2,
squared-error difference, Clark-West, the cross-sectional pricing metrics) attach ``n_obs`` — the
size of the joint finite sample the metric was computed on — and ``n_dropped`` — the count of
candidate observations excluded by that joint mask. They make selective missingness auditable on the
row itself. They are *optional*: rows from evaluators without a benchmark comparison omit them, and
:func:`validate_result` never requires them (only that, when present, they are non-negative counts).
"""


def validate_result(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` violates the result schema.

    Enforces that every column in :data:`RESULT_COLUMNS` is present; extra columns are allowed. When
    the optional :data:`ATTRITION_COLUMNS` are present they must hold non-negative counts (no
    negative or non-finite attrition), but they are never required.
    """
    present = {str(c) for c in df.columns}
    missing = [c for c in RESULT_COLUMNS if c not in present]
    if missing:
        raise ValueError(f"result table missing required columns: {missing}")
    for col in ATTRITION_COLUMNS:
        if col in present:
            # NaN is a legitimate "not applicable" for a row from an evaluator that emits no
            # attrition (e.g. concatenated with benchmark-comparison rows); only reject a present
            # value that is negative or infinite.
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if bool((values < 0).any()) or bool((values == float("inf")).any()):
                raise ValueError(f"result column {col!r} must hold non-negative counts")
