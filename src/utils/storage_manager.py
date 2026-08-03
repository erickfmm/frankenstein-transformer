"""Disk usage manager for dataset caching and temporary file tracking.

Tracks cumulative file sizes against a configurable limit and provides
best-effort cleanup of registered temporary files.
"""

import os
import logging
import tempfile
from pathlib import Path


class StorageManager:
    """Manages disk usage to stay under a configurable storage limit.

    Tracks cumulative bytes consumed by registered files and provides
    temporary file creation with automatic registration for later cleanup.

    Attributes:
        limit_bytes: Maximum allowed cumulative storage in bytes.
        used_bytes: Current cumulative bytes consumed by registered files.
        temp_files: List of paths to temporary files created via
            :meth:`create_temp_file`.
    """

    def __init__(self, limit_gb: float = 500.0):
        """Initialize the storage manager.

        Args:
            limit_gb: Storage limit in gigabytes. Defaults to 500 GB.
        """
        self.limit_bytes = limit_gb * 1024**3
        self.used_bytes = 0
        self.temp_files = []

    def register_file(self, path: str) -> bool:
        """Register a file's size and check whether the limit has been exceeded.

        Args:
            path: Filesystem path to the file to register.

        Returns:
            ``True`` if the cumulative usage is still within the limit,
            ``False`` if the limit has been exceeded.
        """
        try:
            size = os.path.getsize(path)
            self.used_bytes += size

            if self.used_bytes > self.limit_bytes:
                logging.warning(f"Storage limit exceeded: {self.used_bytes/1024**3:.2f}GB")
                return False
            return True
        except:
            return True

    def create_temp_file(self, suffix: str = ".tmp") -> str:
        """Create a temporary file under ``./temp_data`` and register it.

        Args:
            suffix: File suffix for the temporary file (default ``".tmp"``).

        Returns:
            Absolute path to the created temporary file.
        """
        temp_dir = Path("./temp_data")
        temp_dir.mkdir(exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(
            dir=temp_dir,
            suffix=suffix,
            delete=False
        )
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name

    def cleanup(self):
        """Remove all registered temporary files and clear the tracking list."""
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except:
                pass
        self.temp_files.clear()
