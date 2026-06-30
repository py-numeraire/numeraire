# numeraire

A research framework providing a **stable bedrock** for **backtesting, comparing, and
replicating** empirical asset-pricing / financial-econometrics methods (IPCA, VoC, KNS, 1/A,
factor-model tests, …), **extensible by design** so new methods plug in as first-class extensions.

The *numéraire* is the reference unit against which all prices are measured. Core stays
**representation-agnostic**: it defines *capabilities* (what a model can produce — weights, pricing,
…), never a specific method's internal form, so linear-factor, nonlinear/RFF, neural, and
distributional methods are all first-class.

## Architecture (the boundary rule)

`numeraire.core` is exactly the modules that depend on no specific method and that every method
depends on. Dependency arrows point toward `core`; **`core` never imports a method, an adapter, or
a reference library.** This is enforced in CI by `import-linter` — the lint rule *is* the
architecture (see `pyproject.toml [tool.importlinter]`).

```
src/numeraire/
  core/        # spine: DataView/Estimator/Splitter/Evaluator protocols, capabilities,
               # result schema, evaluator registry  (stable, strict-typed, high-coverage)
  adapters/    # optional extras: thin wrappers (ipca, linearmodels) — glue, not spine
  methods/     # published methods bundled as extensions (VoC, 1/A, classical tests, …)
```

Methods register via the `numeraire.methods` entry-point group, so external packages
(`numeraire-yourlab`, `numeraire-<method>`) are first-class peers without editing core.

## Install

```bash
uv sync --extra dev            # dev environment
uv sync --extra all            # + ipca / linearmodels adapters
```

Base install is the spine + native general evaluators only; method/adapter deps are extras.

## Develop

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run basedpyright src/numeraire/core                # strict types on core
uv run lint-imports                                   # architecture boundary
uv run pytest                                         # tests (public/synthetic data only)
```

## Status

Pre-1.0, development. Usable via GitHub install; **not on PyPI yet** (the capability layer is
expected to crystallize once three real adapters land). The spine (`DataView`, walk-forward OOS
engine, `Splitter`, native evaluators) is in place; the first method (1/A conservative slope) is
wired end-to-end.

License: BSD-3-Clause. Never commit CRSP/WRDS/proprietary data or credentials (`data/`, `ref/`,
`.env` are git-ignored).
