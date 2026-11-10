"""Layered plugin discovery.

Three layers, all registering through :func:`src.registry.register_spec`:

1. Installed distributions advertising the ``frankenstein.plugins``
   entry-point group. Load failures are logged as warnings and skipped —
   third-party code must never break core startup.
2. Config-declared modules through the top-level ``plugins:`` YAML key
   (module paths or ``module:object`` references, plus local ``.py`` files).
   A failure here is a hard error: the config explicitly asked for the plugin.
3. In-repo built-ins (:func:`src.registry.ensure_builtins`).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pathlib
from typing import Iterable, Optional

try:
    from src.registry import ensure_builtins
except ImportError:  # legacy flat sys.path layout (src/ on sys.path)
    from registry import ensure_builtins
try:
    from src.registry.errors import PluginLoadError
except ImportError:  # legacy flat sys.path layout
    from registry.errors import PluginLoadError

__all__ = ["load_plugins", "load_entry_point_plugins", "load_plugin_paths"]

_LOGGER = logging.getLogger("frankenstein.plugins")
_ENTRY_POINTS_LOADED = False


def load_entry_point_plugins() -> None:
    """Import installed ``frankenstein.plugins`` entry points (once).

    Each entry point must resolve to a zero-argument callable that registers a
    plugin component. Failures are logged and skipped.
    """
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return
    try:
        selected = entry_points(group="frankenstein.plugins")
    except TypeError:  # pragma: no cover - legacy importlib API
        selected = entry_points().get("frankenstein.plugins", [])
    for entry_point in selected:
        try:
            target = entry_point.load()
            target()
        except Exception as exc:  # noqa: BLE001 - plugin isolation
            _LOGGER.warning(
                "Plugin entry point %r failed to load and was skipped: %s",
                getattr(entry_point, "name", entry_point),
                exc,
            )


def _import_from_path(path: pathlib.Path) -> None:
    """Import a local ``.py`` plugin file.

    Args:
        path: Filesystem path to the plugin module.

    Raises:
        PluginLoadError: If the file cannot be imported.
    """
    module_name = f"frankenstein_plugin_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"Cannot import plugin file {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def load_plugin_paths(paths: Optional[Iterable[str]]) -> None:
    """Import every config-declared plugin.

    Args:
        paths: Import strings (``"pkg.module"``, ``"pkg.module:register"``) or
            local file paths.

    Raises:
        PluginLoadError: If an entry cannot be imported or its registration
            callable fails.
    """
    for entry in paths or ():
        target = str(entry).strip()
        if not target:
            continue
        try:
            candidate = pathlib.Path(target)
            if candidate.suffix == ".py" or "/" in target or "\\" in target:
                if not candidate.is_absolute():
                    candidate = pathlib.Path.cwd() / candidate
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                _import_from_path(candidate)
                continue
            module_name, _, attribute = target.partition(":")
            module = importlib.import_module(module_name)
            if attribute:
                getattr(module, attribute)()
        except Exception as exc:  # noqa: BLE001 - explicit config intent
            raise PluginLoadError(
                f"Could not load plugin {target!r}: {exc}"
            ) from exc


def load_plugins(paths: Optional[Iterable[str]] = None) -> None:
    """Populate the registry from all three plugin layers (idempotent).

    Args:
        paths: Config-declared plugin entries (the top-level ``plugins:`` key).
    """
    ensure_builtins()
    load_entry_point_plugins()
    load_plugin_paths(paths)
