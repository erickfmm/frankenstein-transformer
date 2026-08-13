"""Load bundled Frankenstein presets to populate the plugin's preset dropdown.

Resolves the Frankenstein package's ``configs/`` directory (shipped as package
data) and enumerates the top-level ``*.yaml`` preset files. Each preset name
maps to its full YAML text so the DashAI form can populate the
``frankenstein_yaml`` field when a user picks one.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


def _frankenstein_root() -> str:
    """Return the Frankenstein package root directory.

    The ``frankenstein-transformer`` distribution ships the ``src`` package and
    a sibling ``configs/`` directory. We resolve it relative to the imported
    ``src`` module so it works both in editable (``-e``) and installed mode.

    Returns
    -------
    str
        Absolute path to the Frankenstein repository/package root.

    Raises
    ------
    RuntimeError
        If the Frankenstein ``src`` package cannot be imported.
    """
    try:
        import src  # type: ignore  # Frankenstein's top-level package is ``src``
    except ImportError as exc:  # pragma: no cover - depends on install env
        raise RuntimeError(
            "frankenstein-transformer is not installed (cannot import 'src'). "
            "Install it before using the dashai-frankenstein plugin."
        ) from exc
    return os.path.dirname(os.path.dirname(os.path.abspath(src.__file__)))


@lru_cache(maxsize=1)
def list_presets() -> List[Tuple[str, str]]:
    """Enumerate the bundled Frankenstein presets.

    Returns
    -------
    list of (name, path)
        One tuple per ``configs/*.yaml`` preset, sorted by name. The name is
        the file stem (e.g. ``"mini"``, ``"tinybert"``).
    """
    configs_dir = os.path.join(_frankenstein_root(), "configs")
    presets: List[Tuple[str, str]] = []
    if os.path.isdir(configs_dir):
        for entry in sorted(os.listdir(configs_dir)):
            if entry.lower().endswith((".yaml", ".yml")):
                presets.append((os.path.splitext(entry)[0], os.path.join(configs_dir, entry)))
    return presets


def preset_names() -> List[str]:
    """Return just the preset name list (for the pydantic enum field)."""
    return [name for name, _ in list_presets()]


def load_preset_yaml(name: str) -> Optional[str]:
    """Return the raw YAML text for a preset name, or ``None`` if unknown.

    Parameters
    ----------
    name : str
        Preset name (file stem, e.g. ``"mini"``).

    Returns
    -------
    str or None
        The preset's YAML text, or ``None``.
    """
    for candidate, path in list_presets():
        if candidate == name:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()
    return None


def preset_choices() -> Dict[str, str]:
    """Mapping of preset name -> YAML text (for programmatic consumption)."""
    return {name: (open(path, "r", encoding="utf-8").read()) for name, path in list_presets()}
