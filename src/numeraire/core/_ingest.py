"""Frame-ingestion seam — normalize any supported frame to pandas at the view constructors' door.

The public protocol signatures are pandas-typed (a deliberate decision through 1.0), and the hot
loop is numpy — so a non-pandas frame is converted **once, at the boundary**, and never seen again
by the engine. This keeps ``core`` pandas-primary while letting callers hand in a polars (or any
narwhals-compatible) frame without a hard dependency: narwhals is optional, and a plain
``.to_pandas()`` duck-type is the fallback.

``to_pandas`` accepts, in order:

1. a pandas ``DataFrame`` / ``Series`` — returned unchanged (the overwhelmingly common path; the
   pandas index is preserved, which the ``DatetimeIndex``-based views rely on);
2. any **narwhals-native** frame (polars, pyarrow, modin, …) when narwhals is installed, via
   ``narwhals.stable.v2.from_native(...).to_pandas()``;
3. anything exposing a ``.to_pandas()`` returning a pandas object (e.g. a bare polars frame with no
   narwhals present).

Anything else raises ``TypeError``. Note that a polars frame carries no row index, so its pandas
form has a default ``RangeIndex`` — fine for the tidy-panel :class:`CrossSectionView` (which keys
off a date *column*), but a :class:`TimeSeriesView` still needs the caller to supply a
``DatetimeIndex``.
"""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import pandas as pd

# ``pd.Series`` without a type argument is "partially unknown" under basedpyright-strict; pin the
# element type to ``Any`` at the seam so the normalized-frame type is fully known downstream.
_Frame: TypeAlias = "pd.DataFrame | pd.Series[Any]"


def _narwhals() -> Any:
    try:
        import narwhals.stable.v2 as nw
    except ImportError:  # pragma: no cover - narwhals is an optional dependency
        return None
    return nw


def to_pandas(obj: Any, *, what: str = "frame") -> pd.DataFrame | pd.Series[Any]:
    """Normalize ``obj`` to a pandas ``DataFrame``/``Series`` (order per the module docstring)."""
    if isinstance(obj, pd.DataFrame | pd.Series):
        return cast("_Frame", obj)
    nw = _narwhals()
    if nw is not None:
        try:
            native = nw.from_native(obj)
        except TypeError:
            native = None  # not a narwhals-native frame; try the duck-type fallback
        if native is not None:  # pragma: no cover - exercised only in polars/pyarrow envs
            return cast("_Frame", native.to_pandas())
    to_pd = getattr(obj, "to_pandas", None)
    if callable(to_pd):
        out = to_pd()
        if isinstance(out, pd.DataFrame | pd.Series):
            return cast("_Frame", out)
    raise TypeError(
        f"cannot ingest {what}: expected a pandas frame, a narwhals-native frame, or an object "
        f"with a .to_pandas() method; got {type(obj).__name__}"
    )
