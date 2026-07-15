"""Sphinx configuration for the ALE-MTL documentation."""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../src"))

project = "ALE-MTL"
author = "ALE-MTL contributors"
copyright = f"{date.today().year}, {author}"
release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
html_title = "ALE-MTL documentation"
html_theme_options = {
    "description": "Accumulated Local Effects for Multi-task Learning",
    "fixed_sidebar": True,
    "github_user": "",
    "github_repo": "",
}
