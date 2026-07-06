"""Sphinx configuration for the numeraire documentation site.

Prose pages are authored in MyST markdown; the API reference is generated from the
in-source docstrings by autodoc + autosummary. The build is run with ``-W`` in CI so any
cross-reference rot fails fast.
"""

from __future__ import annotations

from importlib import metadata

project = "numeraire"
author = "Yuheng Wu"
copyright = "2026, Yuheng Wu"

try:
    release = metadata.version("numeraire")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout without install
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "smartquotes",
]
myst_heading_anchors = 3

# -- autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_typehints = "signature"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"numeraire {version}"

# Canonical base URL for the built site (served at the domain root by Cloudflare Pages).
html_baseurl = "https://py-numeraire.org/"

html_theme_options = {
    "github_url": "https://github.com/py-numeraire/numeraire",
    "external_links": [
        {"name": "graphics", "url": "https://graphics.py-numeraire.org/"},
        {"name": "dataset", "url": "https://dataset.py-numeraire.org/"},
    ],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/numeraire/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navigation_with_keys": False,
    "show_prev_next": True,
    "show_toc_level": 2,
    "navbar_align": "left",
}
html_static_path = ["_static"]
