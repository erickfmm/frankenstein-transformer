from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Frankestein Transformer"
copyright = "2026, Erick F. Merino M."
author = "Erick F. Merino M."
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_mdinclude",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "transformers",
    "sentence_transformers",
    "datasets",
    "streamlit",
    "wandb",
    "tensorboard",
    "pynvml",
    "sentencepiece",
    "sacremoses",
    "psutil",
    "tqdm",
    "numpy",
    "pandas",
    "accelerate",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

source_suffix = {
    ".rst": "restructuredtext",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

suppress_warnings = [
    "autodoc.import_object",
    "docutils",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}
