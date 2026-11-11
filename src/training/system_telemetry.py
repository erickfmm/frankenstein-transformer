"""Process- and machine-level CPU/RAM telemetry for the training loop.

The GPU telemetry in :mod:`src.training.trainer` answers "is the accelerator
busy?"; this module answers the questions a CPU-first sweep actually needs:
how much resident memory each training process holds, how many cores it is
really using, and how close the host is to swapping or OOM-killing a worker.

Everything here is device independent, so it is sampled on every step on CPU
*and* on CUDA runs — the host is the shared resource when several workers run
in parallel, and the GPU-only columns cannot see it.

The column names and the zero defaults are exported so the trainer can append
them to the metrics CSV without duplicating the list:

* :data:`SYSTEM_TELEMETRY_COLUMNS` — the CSV header suffix, in order.
* :data:`SYSTEM_TELEMETRY_DEFAULTS` — the same keys mapped to ``0.0``.

``psutil`` is a hard dependency of the project, but every sampling path is
guarded so a stripped-down environment (no ``psutil``, no ``/proc``, Windows)
degrades to zeros instead of failing a training run.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

#: CSV columns appended to the training metrics log, in order.
SYSTEM_TELEMETRY_COLUMNS: Tuple[str, ...] = (
    # --- current process -------------------------------------------------
    "cpu_percent",          # % CPU of this process; 100 = one saturated core
    "cpu_cores",            # cpu_percent / 100, i.e. cores in use
    "cpu_threads",          # OS threads in this process (torch pool + loaders)
    "cpu_rss_mb",           # resident set size, MB
    "cpu_rss_peak_mb",      # high-water mark of cpu_rss_mb, MB
    "cpu_thread_hot_pct",   # busiest single thread, %
    # --- whole machine ---------------------------------------------------
    "sys_cpu_percent",      # % CPU of the host (0..100 * cores)
    "sys_ram_used_pct",     # % of physical RAM in use
    "sys_ram_used_gb",      # GiB of physical RAM in use
    "sys_ram_avail_gb",     # GiB available without swapping
    "sys_swap_used_pct",    # % of swap in use (0 when there is no swap)
    "sys_load_1m",          # 1-minute load average
)

#: Zero-filled telemetry payload, used when sampling is disabled or fails.
SYSTEM_TELEMETRY_DEFAULTS: Dict[str, float] = {
    name: 0.0 for name in SYSTEM_TELEMETRY_COLUMNS
}


def default_system_telemetry() -> Dict[str, float]:
    """Return a fresh all-zero telemetry payload.

    Returns:
        A new dict with every :data:`SYSTEM_TELEMETRY_COLUMNS` key mapped to
        ``0.0``. A fresh copy is returned per call so callers can mutate it.
    """
    return dict(SYSTEM_TELEMETRY_DEFAULTS)


def _read_thread_cpu_seconds() -> Dict[int, float]:
    """Read per-thread CPU time from ``/proc/self/task/<tid>/stat``.

    Linux exposes the cumulative user+system time of every thread there. The
    values are deltas in ``sysconf("SC_CLK_TCK")`` ticks, converted to seconds
    so :meth:`SystemTelemetrySampler.sample` can turn them into per-thread
    utilisation percentages without another syscall.

    Returns:
        ``{thread_id: cpu_seconds}``; empty when ``/proc`` is unavailable.
    """
    try:
        ticks = float(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, ValueError, OSError):
        return {}
    if ticks <= 0:
        return {}
    usage: Dict[int, float] = {}
    try:
        task_dir = f"/proc/{os.getpid()}/task"
        for entry in os.listdir(task_dir):
            try:
                with open(f"{task_dir}/{entry}/stat", "r", encoding="utf-8") as handle:
                    fields = handle.read().rsplit(") ", 1)[-1].split()
                # After the "(comm) " suffix, `state` is field 3: utime/stime
                # are fields 14/15 overall, i.e. indexes 11 and 12 here.
                usage[int(entry)] = (float(fields[11]) + float(fields[12])) / ticks
            except (OSError, IndexError, ValueError):
                continue
    except OSError:
        return {}
    return usage


class SystemTelemetrySampler:
    """Sample CPU/RAM usage of the current process and of the whole machine.

    The sampler keeps the state that rate-based metrics need (the previous
    per-thread CPU counters, the RSS high-water mark) so successive calls
    report usage *since the previous sample*, which is exactly the semantics
    of ``psutil.Process.cpu_percent(interval=None)``. The first
    :meth:`sample` call after construction therefore reports near-zero
    process CPU; call :meth:`prime` once when a run starts if the very first
    step should be meaningful.

    Attributes:
        enabled: When false, :meth:`sample` returns zeros (schema toggle
            ``training.system_metrics_enabled``).
        thread_breakdown: When true, the busiest-thread percentage is
            collected from ``/proc`` (schema toggle
            ``training.system_metrics_thread_breakdown``).
    """

    def __init__(
        self,
        enabled: bool = True,
        thread_breakdown: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.thread_breakdown = bool(thread_breakdown)
        self._psutil: Optional[Any] = None
        self._process: Optional[Any] = None
        self._peak_rss_mb = 0.0
        self._thread_prev: Dict[int, float] = {}
        self._thread_prev_at: Optional[float] = None
        if self.enabled:
            self._install_backend()
            self.prime()

    # -- setup ---------------------------------------------------------
    def _install_backend(self) -> None:
        """Bind ``psutil`` and the current process, or leave the backend unset."""
        try:
            import psutil  # imported lazily so importing this module stays cheap

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
        except Exception:  # noqa: BLE001 - telemetry must never break training
            self._psutil = None
            self._process = None

    def prime(self) -> None:
        """Reset the rate baselines so the next sample covers a real interval."""
        self.prime_rates()

    def prime_rates(self) -> None:
        """Take the zero-th CPU reading that ``psutil`` needs for percentages."""
        if not self.enabled or self._psutil is None:
            return
        try:
            if self._process is not None:
                self._process.cpu_percent(interval=None)
            self._psutil.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            pass
        if self.thread_breakdown:
            self._thread_prev = _read_thread_cpu_seconds()
            self._thread_prev_at = time.monotonic()

    # -- sampling ------------------------------------------------------
    def _sample_thread_hot_pct(self) -> float:
        """Return the CPU percentage of the busiest single thread."""
        if not self.thread_breakdown:
            return 0.0
        current = _read_thread_cpu_seconds()
        now = time.monotonic()
        previous, previous_at = self._thread_prev, self._thread_prev_at
        self._thread_prev, self._thread_prev_at = current, now
        if not previous or previous_at is None:
            return 0.0
        elapsed = now - previous_at
        if elapsed <= 0:
            return 0.0
        hottest = 0.0
        for thread_id, cpu_seconds in current.items():
            delta = cpu_seconds - previous.get(thread_id, cpu_seconds)
            if delta <= 0:
                continue
            hottest = max(hottest, 100.0 * delta / elapsed)
        # A thread can at most saturate one core per wall-clock second.
        return min(hottest, 100.0)

    def sample(self) -> Dict[str, float]:
        """Return one CPU/RAM telemetry row.

        Returns:
            A dict with one entry per :data:`SYSTEM_TELEMETRY_COLUMNS` key.
            Missing backends yield the zero defaults, never an exception.
        """
        telemetry = default_system_telemetry()
        if not self.enabled:
            return telemetry

        try:
            if self._process is not None:
                memory = self._process.memory_info()
                rss_mb = float(memory.rss) / (1024.0 ** 2)
                self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)
                cpu_percent = float(self._process.cpu_percent(interval=None) or 0.0)
                telemetry["cpu_percent"] = cpu_percent
                telemetry["cpu_cores"] = cpu_percent / 100.0
                telemetry["cpu_threads"] = float(self._process.num_threads())
                telemetry["cpu_rss_mb"] = rss_mb
                telemetry["cpu_rss_peak_mb"] = self._peak_rss_mb
        except Exception:  # noqa: BLE001
            pass

        try:
            if self._psutil is not None:
                telemetry["sys_cpu_percent"] = float(
                    self._psutil.cpu_percent(interval=None) or 0.0
                )
                virtual = self._psutil.virtual_memory()
                telemetry["sys_ram_used_pct"] = float(virtual.percent)
                telemetry["sys_ram_used_gb"] = float(virtual.used) / (1024.0 ** 3)
                telemetry["sys_ram_avail_gb"] = float(virtual.available) / (1024.0 ** 3)
                try:
                    swap = self._psutil.swap_memory()
                    telemetry["sys_swap_used_pct"] = float(swap.percent)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    telemetry["sys_load_1m"] = float(os.getloadavg()[0])
                except (AttributeError, OSError):
                    pass
        except Exception:  # noqa: BLE001
            pass

        try:
            telemetry["cpu_thread_hot_pct"] = self._sample_thread_hot_pct()
        except Exception:  # noqa: BLE001
            pass

        return telemetry

    def thread_usage(self) -> List[Tuple[int, float]]:
        """Return ``(thread_id, percent)`` pairs sorted by descending usage.

        This is the detailed view behind ``cpu_thread_hot_pct``: it reads the
        per-thread counters once and reports how the process' CPU is spread
        across its threads, which is what tells a fixed
        ``OMP_NUM_THREADS`` apart from a thread pool that is actually being
        used.

        Returns:
            One pair per thread, busiest first; empty when the breakdown is
            disabled or ``/proc`` is unavailable.
        """
        if not self.enabled or not self.thread_breakdown:
            return []
        current = _read_thread_cpu_seconds()
        now = time.monotonic()
        previous, previous_at = self._thread_prev, self._thread_prev_at
        self._thread_prev, self._thread_prev_at = current, now
        if not previous or previous_at is None:
            return []
        elapsed = now - previous_at
        if elapsed <= 0:
            return []
        usage = [
            (thread_id, min(100.0 * (seconds - previous.get(thread_id, seconds)) / elapsed, 100.0))
            for thread_id, seconds in current.items()
        ]
        usage = [item for item in usage if item[1] > 0.0]
        usage.sort(key=lambda item: item[1], reverse=True)
        return usage


def format_thread_usage(usage: List[Tuple[int, float]], limit: int = 4) -> str:
    """Render :meth:`SystemTelemetrySampler.thread_usage` as a compact string.

    Args:
        usage: Pairs returned by :meth:`SystemTelemetrySampler.thread_usage`.
        limit: Maximum number of threads to render.

    Returns:
        ``"tid:pct|tid:pct"`` for the busiest threads, or ``""`` when empty.
    """
    return "|".join(
        f"{thread_id}:{percent:.0f}" for thread_id, percent in usage[: max(int(limit), 1)]
    )
