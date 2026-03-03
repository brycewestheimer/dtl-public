# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "Distributed Template Library (DTL)"
author = "Bryce M. Westheimer"
copyright = "2026, Bryce M. Westheimer"
release = "0.1.0-alpha.1"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "breathe",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

myst_heading_anchors = 3

exclude_patterns = [
    "_build",
    "_generated",
    "Thumbs.db",
    ".DS_Store",
    "archive/**",
]

todo_include_todos = False

suppress_warnings = [
    "toc.not_included",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = "DTL Documentation"
html_static_path: list[str] = []

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# -- Path context ------------------------------------------------------------

# Keep docs self-contained while allowing relative links to repository files.
DOCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOCS_ROOT.parent

# -- Breathe/Doxygen integration ----------------------------------------------

DOXYGEN_XML_DIR = Path(
    os.environ.get(
        "DTL_DOXYGEN_XML_DIR",
        str(REPO_ROOT / "docs" / "_generated" / "doxygen" / "xml"),
    )
).resolve()

breathe_projects = {
    "DTL": str(DOXYGEN_XML_DIR),
}
breathe_default_project = "DTL"
breathe_domain_by_extension = {
    "h": "c",
    "hpp": "cpp",
}
