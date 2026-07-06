# The ecosystem

`numeraire` is the spine of a small, deliberately decoupled family of packages. The spine ships only
tiny public example slices and **no bundled methods**: everything else — plotting, data, and the
methods themselves — is an optional, first-class peer that depends *back* on `numeraire` and is
discovered at install time. This is the {doc}`boundary rule <architecture>` expressed as packaging:
the arrows point toward the spine, never away from it.

## The packages

| Package | Role | Install |
| --- | --- | --- |
| **[numeraire](https://numeraire.py-numeraire.org/)** | The spine: point-in-time views, the walk-forward engine, evaluators, statistical tests, the result schema, and the universal baselines. | `pip install numeraire` |
| **[numeraire-graphics](https://graphics.py-numeraire.org/)** | Grammar-of-graphics figures (plotnine) over the tidy result schema and the engine's Output objects. Every plot returns a `ggplot`; it never draws or saves for you. | `pip install numeraire-graphics` |
| **[numeraire-dataset](https://dataset.py-numeraire.org/)** | Open, reproducible data loaders and point-in-time builders — code, not data — cleaning public (and, with your own credentials, licensed) sources into tidy tables the framework consumes. | `pip install numeraire-dataset` |

Each package name links to its documentation site
(`numeraire.py-numeraire.org`, `graphics.py-numeraire.org`, `dataset.py-numeraire.org`); source and releases
are on [PyPI](https://pypi.org/project/numeraire/) and GitHub (org
[py-numeraire](https://github.com/py-numeraire)).

The convenience extras on the spine pull the companions in for you:

```bash
pip install "numeraire[all]"        # graphics + dataset
pip install "numeraire[graphics]"   # graphics only
pip install "numeraire[data]"       # dataset only
```

## Methods are packages too

Reproductions of published methods, and a lab's own unpublished methods, are **not** part of the
spine. A method is any object with `fit(view) -> model` whose fitted model advertises a capability;
it registers through the `numeraire.methods` entry-point group and pins a compatible `numeraire`, so
installing it makes it discoverable without any edit to core. See {doc}`extending` for the full
recipe — writing a method, self-certifying it with the conformance suite, and pinning a reproduction
target. A public reproduction collection is in preparation.

## Compatibility

The spine's result schema and public API follow semantic versioning; the companion packages consume
only that stable surface. Each companion pins a compatible `numeraire` (for example `numeraire>=0.2`
within the `0.x` line), so `pip install "numeraire[all]"` always resolves a matching set. While the
project is `0.x` the capability protocols may still evolve between minor versions; changes are
recorded in the {doc}`changelog`.

## Design boundary

Keeping the layers separate is a design choice, not an accident:

- **The spine stays dependency-light and visualization-free.** A backtest never imports a plotting
  library; a figure is produced downstream from the tidy result schema, so the choice of plotting
  library stays decoupled.
- **No licensed data is ever redistributed inside a wheel.** The data package ships transparent ETL;
  build outputs land in a local cache, never the repository.
- **Methods evolve independently of the spine.** A new estimator, or a fix to a reproduction, ships
  on its own package's release cadence without waiting on a core release.
