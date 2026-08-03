"""Unit tests for StorageManager."""
import os
import tempfile
import unittest

try:
    from src.utils.storage_manager import StorageManager
    _IMPORTS_OK = True
except ImportError:
    _IMPORTS_OK = False


@unittest.skipUnless(_IMPORTS_OK, "src.utils.storage_manager unavailable")
class StorageManagerRegisterFileTests(unittest.TestCase):
    def test_register_existing_file_returns_true(self):
        mgr = StorageManager(limit_gb=500.0)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello")
            path = f.name
        try:
            result = mgr.register_file(path)
            self.assertTrue(result)
        finally:
            os.unlink(path)

    def test_register_nonexistent_file_returns_true(self):
        mgr = StorageManager(limit_gb=500.0)
        result = mgr.register_file("/nonexistent/path/file.bin")
        self.assertTrue(result)

    def test_register_file_accumulates_used_bytes(self):
        mgr = StorageManager(limit_gb=500.0)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 1000)
            path = f.name
        try:
            mgr.register_file(path)
            self.assertGreater(mgr.used_bytes, 0)
        finally:
            os.unlink(path)

    def test_register_file_exceeds_limit_returns_false(self):
        mgr = StorageManager(limit_gb=0.0)  # effectively 0 bytes
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            result = mgr.register_file(path)
            self.assertFalse(result)
        finally:
            os.unlink(path)


@unittest.skipUnless(_IMPORTS_OK, "src.utils.storage_manager unavailable")
class StorageManagerTempFileTests(unittest.TestCase):
    def setUp(self):
        self._mgr = StorageManager()

    def tearDown(self):
        self._mgr.cleanup()
        # Remove temp_data dir if empty (best-effort)
        try:
            os.rmdir("./temp_data")
        except OSError:
            pass

    def test_create_temp_file_returns_path(self):
        path = self._mgr.create_temp_file()
        self.assertIsInstance(path, str)
        self.assertTrue(os.path.exists(path))

    def test_create_temp_file_custom_suffix(self):
        path = self._mgr.create_temp_file(suffix=".json")
        self.assertTrue(path.endswith(".json"))

    def test_create_temp_file_tracked(self):
        path = self._mgr.create_temp_file()
        self.assertIn(path, self._mgr.temp_files)

    def test_multiple_temp_files_all_tracked(self):
        paths = [self._mgr.create_temp_file() for _ in range(3)]
        for p in paths:
            self.assertIn(p, self._mgr.temp_files)


@unittest.skipUnless(_IMPORTS_OK, "src.utils.storage_manager unavailable")
class StorageManagerCleanupTests(unittest.TestCase):
    def _mgr(self):
        return StorageManager()

    def test_cleanup_removes_temp_files(self):
        mgr = self._mgr()
        path = mgr.create_temp_file()
        mgr.cleanup()
        self.assertFalse(os.path.exists(path))
        # Clean up
        try:
            os.rmdir("./temp_data")
        except OSError:
            pass

    def test_cleanup_clears_temp_files_list(self):
        mgr = self._mgr()
        mgr.create_temp_file()
        mgr.cleanup()
        self.assertEqual(mgr.temp_files, [])
        try:
            os.rmdir("./temp_data")
        except OSError:
            pass

    def test_cleanup_idempotent(self):
        mgr = self._mgr()
        mgr.cleanup()
        mgr.cleanup()  # should not raise
        try:
            os.rmdir("./temp_data")
        except OSError:
            pass

    def test_cleanup_handles_missing_file_gracefully(self):
        mgr = self._mgr()
        mgr.temp_files.append("/nonexistent/phantom.tmp")
        mgr.cleanup()  # should not raise


if __name__ == "__main__":
    unittest.main()
