"""Layered plugin loading: config paths, entry points, failure semantics."""

from __future__ import annotations

import pathlib
import unittest
from unittest import mock

from src.registry import ComponentFamily, registry
from src.registry.errors import PluginLoadError


_PLUGIN_SOURCE = """
from src.plugins.api import register_activation

register_activation(
    name="demo_activation",
    builder="torch.nn:Identity",
    description="Demo activation.",
    description_es="Activación demo.",
)
"""


class PluginPathTests(unittest.TestCase):
    """Config-declared plugin paths load and register."""

    def test_local_file_plugin_registers(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "demo_plugin.py"
            path.write_text(_PLUGIN_SOURCE, encoding="utf-8")
            from src.plugins.loader import load_plugin_paths

            load_plugin_paths([str(path)])
        try:
            self.assertIn("demo_activation", registry.names(ComponentFamily.ACTIVATION))
        finally:
            registry.unregister(ComponentFamily.ACTIVATION, "demo_activation")

    def test_missing_plugin_is_hard_error(self):
        from src.plugins.loader import load_plugin_paths

        with self.assertRaises(PluginLoadError):
            load_plugin_paths(["./does/not/exist.py"])

    def test_module_path_plugin_registers(self):
        import sys
        import tempfile
        import types

        module = types.ModuleType("demo_frankenstein_plugin")
        exec(compile(_PLUGIN_SOURCE, "<plugin>", "exec"), module.__dict__)
        with mock.patch.dict(sys.modules, {"demo_frankenstein_plugin": module}):
            from src.plugins.loader import load_plugin_paths

            load_plugin_paths(["demo_frankenstein_plugin"])
        try:
            self.assertIn("demo_activation", registry.names(ComponentFamily.ACTIVATION))
        finally:
            registry.unregister(ComponentFamily.ACTIVATION, "demo_activation")


class EntryPointTests(unittest.TestCase):
    """Entry-point plugins load once; failures are isolated."""

    def test_entry_point_failure_is_skipped(self):
        from src.plugins import loader

        loader._ENTRY_POINTS_LOADED = False
        bad = mock.Mock()
        bad.load.side_effect = RuntimeError("boom")
        bad.name = "bad"
        with mock.patch("importlib.metadata.entry_points", return_value=[bad]):
            loader.load_entry_point_plugins()  # must not raise
        loader._ENTRY_POINTS_LOADED = False

    def test_entry_point_success_registers(self):
        import sys
        import types

        module = types.ModuleType("demo_entry_plugin")
        exec(compile(_PLUGIN_SOURCE, "<plugin>", "exec"), module.__dict__)
        callback = mock.Mock()
        callback.load.return_value = lambda: exec(
            compile(_PLUGIN_SOURCE, "<plugin>", "exec"), module.__dict__
        )
        callback.name = "demo"
        with mock.patch.dict(sys.modules, {"demo_entry_plugin": module}):
            from src.plugins import loader

            loader._ENTRY_POINTS_LOADED = False
            with mock.patch("importlib.metadata.entry_points", return_value=[callback]):
                loader.load_entry_point_plugins()
            loader._ENTRY_POINTS_LOADED = False
        try:
            self.assertIn("demo_activation", registry.names(ComponentFamily.ACTIVATION))
        finally:
            registry.unregister(ComponentFamily.ACTIVATION, "demo_activation")
