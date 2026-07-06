# Installation

`numeraire` requires **Python 3.11 or newer**. Its runtime dependencies are the scientific-Python
core only — numpy, scipy, pandas, and scikit-learn.

## From PyPI

```bash
pip install numeraire
```

The base install is the **spine** — the point-in-time views, the walk-forward engine, the native
evaluators, the statistical tests, and the result schema — plus the universal baselines. It carries
no method-specific or plotting dependencies.

## Extras

Opt into the companion packages and optional integrations through extras:

| Extra | Pulls in | For |
| --- | --- | --- |
| `numeraire[all]` | `numeraire-graphics` + `numeraire-dataset` | the full companion ecosystem |
| `numeraire[graphics]` | `numeraire-graphics` | grammar-of-graphics figures over results and Output objects |
| `numeraire[data]` | `numeraire-dataset` | open, reproducible data loaders and point-in-time builders |
| `numeraire[skfolio]` | `skfolio` | the constrained-portfolio-optimizer adapter |

```bash
pip install "numeraire[all]"
```

Each companion package also installs on its own (`pip install numeraire-graphics`) and depends back
on a compatible `numeraire`; see {doc}`ecosystem`.

## With uv

In a [uv](https://docs.astral.sh/uv/)-managed project:

```bash
uv add numeraire            # or: uv add "numeraire[all]"
```

## From source (development)

```bash
git clone https://github.com/py-numeraire/numeraire
cd numeraire
uv sync --extra dev         # spine + the full development toolchain
```

The development environment adds `ruff`, `basedpyright`, `import-linter`, `pytest`, and the docs
toolchain. The standard checks:

```bash
uv run ruff check . && uv run ruff format --check .
uv run basedpyright src/numeraire/core     # strict types on the spine
uv run lint-imports                        # the architecture boundary
uv run pytest                              # tests (public / synthetic data only)
```

## Verify

```python
from importlib.metadata import version

import numeraire  # noqa: F401

print(version("numeraire"))
```
