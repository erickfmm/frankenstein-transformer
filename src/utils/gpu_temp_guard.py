"""GPU temperature supervisor for CUDA training safety.

Provides a **CPU-side supervisor** that owns the lifecycle of a GPU training
subprocess. The supervisor itself holds no GPU memory and runs purely on the
CPU, polling GPU temperature via NVML (primary) or ``nvidia-smi`` (fallback).

State machine
--------------
The supervisor re-launches the training child (resuming from the latest
rolling checkpoint) and watches the GPU temperature every
``poll_interval_seconds``:

* ``temp >= critical_threshold_c``:
    - ``switch_on_thermal == False`` (default): **permanent abort** — kill the
      child and return a non-zero exit code (hardware unsafe).
    - ``switch_on_thermal == True``: **CPU continuation** — checkpoint the
      child via SIGUSR1, kill it, then re-launch with ``--device cpu`` so
      training continues on CPU. The supervisor then stops polling temperature
      (no GPU in use) and waits for the CPU child to finish.
* ``pause_threshold_c <= temp < critical_threshold_c``: **thermal pause** —
  send SIGUSR1 (request a rolling checkpoint flush), wait up to
  ``checkpoint_grace_seconds`` for the child to acknowledge via a sentinel
  file, then SIGKILL the process group. Wait for the GPU to cool below
  ``resume_threshold_c`` and re-launch on GPU with
  ``--resume-from-checkpoint auto``.
* child exits cleanly (rc==0): training done, return 0.
* child crashes (rc!=0, not killed by us): return the child's exit code.

Backward compatibility
----------------------
:class:`GPUTemperatureGuard` is kept as a thin shim that exposes the
temperature-reading helpers (``read_temperature_c``, threshold properties) so
that existing tests and lightweight in-process callers continue to work. The
supervisor (:class:`GPUTempSupervisor`) is the new primary class.
"""

from __future__ import annotations

import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

# ``torch`` is optional here: the supervisor is CPU-only and reads temperature
# via nvidia-smi/NVML directly, so importing this module must not hard-require
# torch. Callers that need CUDA-aware behaviour guard torch themselves.
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - CPU-only CI / environments without torch
    torch = None  # type: ignore


class GPUTelemetryError(RuntimeError):
    """Fatal telemetry error indicating training must stop to avoid unsafe execution."""


@dataclass
class GPUTempCheckResult:
    """Result of a single GPU temperature check cycle.

    Attributes:
        temp_c: Current GPU temperature in degrees Celsius.
        paused: Whether training was paused during this check.
        pause_duration_s: Total seconds spent paused (0 if not paused).
        checks_during_pause: Number of temperature polls performed while paused.
        critical_seen: Whether the critical threshold was reached at any point.
    """

    temp_c: float
    paused: bool
    pause_duration_s: float
    checks_during_pause: int
    critical_seen: bool

    @property
    def repair_action(self) -> str:
        """Human-readable label describing the thermal action taken."""
        if not self.paused:
            return "none"
        prefix = "thermal_pause_critical" if self.critical_seen else "thermal_pause"
        return f"{prefix}_{int(round(self.pause_duration_s))}s"


# ---------------------------------------------------------------------------
# Temperature reading (shared by the supervisor and the compat shim)
# ---------------------------------------------------------------------------
def _parse_temperature(raw: str) -> float:
    value = (raw or "").strip()
    if not value or value.lower() in {"n/a", "[not supported]"}:
        raise GPUTelemetryError(f"Invalid GPU temperature payload: {raw!r}")
    match = re.search(r"[-+]?\d*\.?\d+", value)
    if match is None:
        raise GPUTelemetryError(f"Unable to parse GPU temperature payload: {raw!r}")
    temp_c = float(match.group(0))
    if not math.isfinite(temp_c) or temp_c <= 0.0:
        raise GPUTelemetryError(f"Non-finite or invalid GPU temperature: {temp_c!r}")
    return temp_c


def read_temperature_from_nvidia_smi(nvml_device_index: int = 0) -> float:
    """Read GPU temperature via ``nvidia-smi`` subprocess.

    Args:
        nvml_device_index: GPU index to query.

    Returns:
        Temperature in Celsius.

    Raises:
        GPUTelemetryError: If nvidia-smi is unavailable or returns an invalid
            payload.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu",
        "--format=csv,noheader,nounits",
        "-i",
        str(int(nvml_device_index)),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except FileNotFoundError as exc:
        raise GPUTelemetryError("nvidia-smi command not found") from exc
    except subprocess.CalledProcessError as exc:
        raise GPUTelemetryError(
            f"nvidia-smi failed (exit={exc.returncode}): {exc.stderr or exc.stdout}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise GPUTelemetryError("nvidia-smi timed out") from exc
    except Exception as exc:
        raise GPUTelemetryError(f"Unexpected nvidia-smi failure: {exc}") from exc

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise GPUTelemetryError("nvidia-smi returned empty temperature output")
    return _parse_temperature(lines[0])


class GPUTemperatureGuard:
    """Backward-compatible temperature reader / threshold holder.

    This class no longer implements the in-process pause/resume loop (that
    responsibility moved to :class:`GPUTempSupervisor`). It still exposes
    ``read_temperature_c`` and the threshold properties so legacy callers and
    tests that only need a temperature probe can keep working unchanged.

    Attributes:
        is_active: Whether temperature probing is available (CUDA available
            and guard enabled).
        pause_threshold_c: Temperature above which the supervisor pauses.
        resume_threshold_c: Temperature below which the supervisor resumes.
        critical_threshold_c: Optional critical / permanent-abort threshold.
        poll_interval_seconds: Seconds between temperature polls.
        last_temperature_c: Most recently read temperature, or ``None``.
        total_pause_events: Cumulative pause events (kept for API parity; the
            supervisor owns the real counter now).
        total_paused_seconds: Cumulative paused seconds (API parity only).
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device: str,
        nvml_device_index: int = 0,
        pause_threshold_c: float = 90.0,
        resume_threshold_c: float = 80.0,
        critical_threshold_c: Optional[float] = None,
        poll_interval_seconds: float = 30.0,
    ):
        """Initialize the temperature guard shim.

        Args:
            enabled: Whether the guard is enabled.
            device: Target device string (probing only activates for CUDA).
            nvml_device_index: NVML device index for multi-GPU systems.
            pause_threshold_c: Pause threshold in Celsius. Must be > 0.
            resume_threshold_c: Resume threshold in Celsius. Must be > 0 and
                < ``pause_threshold_c``.
            critical_threshold_c: Optional critical threshold. Must be > 0
                when provided.
            poll_interval_seconds: Seconds between temperature polls. > 0.

        Raises:
            ValueError: If threshold or poll interval constraints are
                violated.
        """
        self._logger = logging.getLogger(__name__)
        self._device = str(device or "cpu")
        self._enabled = bool(enabled)
        self._nvml_device_index = int(nvml_device_index)
        self._pause_threshold_c = float(pause_threshold_c)
        self._resume_threshold_c = float(resume_threshold_c)
        self._critical_threshold_c = (
            None if critical_threshold_c is None else float(critical_threshold_c)
        )
        self._poll_interval_seconds = float(poll_interval_seconds)

        if self._pause_threshold_c <= 0:
            raise ValueError("pause_threshold_c must be > 0")
        if self._resume_threshold_c <= 0:
            raise ValueError("resume_threshold_c must be > 0")
        if self._resume_threshold_c >= self._pause_threshold_c:
            raise ValueError("resume_threshold_c must be < pause_threshold_c")
        if self._poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        if self._critical_threshold_c is not None and self._critical_threshold_c <= 0:
            raise ValueError("critical_threshold_c must be > 0 when provided")

        self._active = (
            self._enabled
            and self._device.startswith("cuda")
            and torch is not None
            and torch.cuda.is_available()
        )

        self._nvml_module = None
        self._nvml_handle = None
        self._nvml_disabled = False
        self._nvml_fallback_logged = False

        self.last_temperature_c: Optional[float] = None
        self.total_pause_events = 0
        self.total_paused_seconds = 0.0

        if self._active:
            self._init_nvml()

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def pause_threshold_c(self) -> float:
        return self._pause_threshold_c

    @property
    def resume_threshold_c(self) -> float:
        return self._resume_threshold_c

    @property
    def critical_threshold_c(self) -> Optional[float]:
        return self._critical_threshold_c

    @property
    def poll_interval_seconds(self) -> float:
        return self._poll_interval_seconds

    def _init_nvml(self):
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._nvml_device_index)
            self._nvml_module = pynvml
        except Exception as exc:
            self._nvml_disabled = True
            if not self._nvml_fallback_logged:
                self._logger.warning(
                    "NVML unavailable for thermal guard, falling back to nvidia-smi: %s",
                    exc,
                )
                self._nvml_fallback_logged = True

    def _read_temperature_from_nvml(self) -> float:
        if self._nvml_disabled or self._nvml_module is None or self._nvml_handle is None:
            raise GPUTelemetryError("NVML backend unavailable")
        try:
            pynvml = self._nvml_module
            return float(
                pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            )
        except Exception as exc:
            raise GPUTelemetryError(f"NVML temperature read failed: {exc}") from exc

    def _read_temperature_from_nvidia_smi(self) -> float:
        """Backward-compat wrapper around the module-level nvidia-smi reader."""
        return read_temperature_from_nvidia_smi(self._nvml_device_index)

    def _parse_temperature(self, raw: str) -> float:
        """Backward-compat wrapper around the module-level parser."""
        return _parse_temperature(raw)

    def read_temperature_c(self) -> float:
        """Read the current GPU temperature in Celsius.

        Attempts NVML first, then ``nvidia-smi``. Returns ``0.0`` when the
        guard is not active (CPU device or disabled).

        Raises:
            GPUTelemetryError: If both backends fail to provide a valid
                temperature reading.
        """
        if not self._active:
            self.last_temperature_c = None
            return 0.0

        nvml_error: Optional[Exception] = None
        if not self._nvml_disabled:
            try:
                temp_c = self._read_temperature_from_nvml()
                self.last_temperature_c = temp_c
                return temp_c
            except Exception as exc:
                nvml_error = exc
                self._nvml_disabled = True
                if not self._nvml_fallback_logged:
                    self._logger.warning(
                        "Thermal guard NVML read failed, falling back to nvidia-smi: %s",
                        exc,
                    )
                    self._nvml_fallback_logged = True

        try:
            temp_c = self._read_temperature_from_nvidia_smi()
            self.last_temperature_c = temp_c
            return temp_c
        except Exception as smi_exc:
            if nvml_error is None:
                raise GPUTelemetryError(f"GPU telemetry failed: {smi_exc}") from smi_exc
            raise GPUTelemetryError(
                f"GPU telemetry failed (NVML and nvidia-smi). "
                f"NVML: {nvml_error}; nvidia-smi: {smi_exc}"
            ) from smi_exc

    def wait_until_safe(self, *, context: str = "") -> GPUTempCheckResult:
        """Block until GPU temperature drops below the resume threshold.

        Kept for backward compatibility with callers/tests that drive an
        in-process pause loop. The supervisor does not use this method; it
        kills and restarts the child instead.

        Args:
            context: Optional human-readable label for log messages.

        Returns:
            A :class:`GPUTempCheckResult` describing the check outcome.
        """
        if not self._active:
            return GPUTempCheckResult(
                temp_c=0.0,
                paused=False,
                pause_duration_s=0.0,
                checks_during_pause=0,
                critical_seen=False,
            )

        temp_c = self.read_temperature_c()
        if temp_c <= self._pause_threshold_c:
            return GPUTempCheckResult(
                temp_c=temp_c,
                paused=False,
                pause_duration_s=0.0,
                checks_during_pause=0,
                critical_seen=(
                    self._critical_threshold_c is not None
                    and temp_c >= self._critical_threshold_c
                ),
            )

        critical_seen = (
            self._critical_threshold_c is not None
            and temp_c >= self._critical_threshold_c
        )
        checks = 0
        started = time.perf_counter()
        context_label = context or "training"

        self._logger.warning(
            "[ThermalGuard] %s paused: GPU %.1fC > %.1fC (resume <= %.1fC)",
            context_label,
            temp_c,
            self._pause_threshold_c,
            self._resume_threshold_c,
        )

        while temp_c > self._resume_threshold_c:
            time.sleep(self._poll_interval_seconds)
            checks += 1
            temp_c = self.read_temperature_c()
            if (
                self._critical_threshold_c is not None
                and temp_c >= self._critical_threshold_c
            ):
                critical_seen = True
            self._logger.warning(
                "[ThermalGuard] %s waiting: GPU %.1fC (resume <= %.1fC)",
                context_label,
                temp_c,
                self._resume_threshold_c,
            )

        duration_s = time.perf_counter() - started
        self.total_pause_events += 1
        self.total_paused_seconds += duration_s
        self._logger.warning(
            "[ThermalGuard] %s resumed after %.1fs at %.1fC",
            context_label,
            duration_s,
            temp_c,
        )

        return GPUTempCheckResult(
            temp_c=temp_c,
            paused=True,
            pause_duration_s=duration_s,
            checks_during_pause=checks,
            critical_seen=critical_seen,
        )


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------
SUPERVISOR_CHILD_ENV_VAR = "FRANKESTEIN_SUPERVISOR_CHILD"
CHECKPOINT_DONE_SENTINEL = ".supervisor_checkpoint_done"
RESUME_AUTO = "auto"


def _kill_process_group(proc: "subprocess.Popen") -> None:
    """SIGKILL the whole process group of *proc* (the worker + its children)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


class GPUTempSupervisor:
    """CPU-side supervisor that kills/restarts a GPU training subprocess on thermal events.

    The supervisor spawns the training command as a **direct child** process
    (``start_new_session=True`` so it leads its own process group). It polls
    GPU temperature every ``poll_interval_seconds`` and, on a thermal event:

    * Sends ``SIGUSR1`` to the child to request an immediate rolling
      checkpoint flush (the child acknowledges by writing a sentinel file in
      ``checkpoint_dir``).
    * Waits up to ``checkpoint_grace_seconds`` for the sentinel, then
      ``SIGKILL`` the child's process group.
    * Waits for the GPU to cool below ``resume_threshold_c`` and re-launches
      the child with ``--resume-from-checkpoint auto`` (the child loads the
      latest rolling checkpoint itself).

    At the critical threshold the behaviour depends on ``switch_on_thermal``:

    * ``switch_on_thermal == False``: kill and **abort permanently** (return 2).
    * ``switch_on_thermal == True``: checkpoint, kill, and re-launch the child
      with ``--device cpu`` (resume on CPU). The supervisor then stops polling
      temperature and just waits for the CPU child to finish.

    Attributes:
        is_active: Whether the supervisor is active (enabled + CUDA available).
        pause_threshold_c: Temperature that triggers a kill+cooldown+restart.
        resume_threshold_c: Temperature below which the GPU child is re-launched.
        critical_threshold_c: Permanent-abort (or CPU-switch) threshold.
        poll_interval_seconds: Seconds between temperature polls while the
            GPU child is running.
        checkpoint_grace_seconds: Max seconds to wait for the child's
            checkpoint acknowledgement before SIGKILL.
        switch_on_thermal: Whether critical temp triggers CPU continuation.
        total_pause_events: Cumulative kill+restart events.
        total_paused_seconds: Cumulative seconds spent cooling down.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        device: str,
        nvml_device_index: int = 0,
        pause_threshold_c: float = 90.0,
        resume_threshold_c: float = 80.0,
        critical_threshold_c: Optional[float] = None,
        poll_interval_seconds: float = 30.0,
        checkpoint_dir: str = "checkpoints",
        checkpoint_grace_seconds: float = 30.0,
        switch_on_thermal: bool = False,
        child_argv: Optional[List[str]] = None,
        child_env: Optional[dict] = None,
        log_fn: Callable[[str], None] = logging.warning,
    ):
        """Initialize the supervisor.

        Args:
            enabled: Whether the supervisor is enabled.
            device: Target device string (supervisor only activates for CUDA).
            nvml_device_index: GPU index to monitor.
            pause_threshold_c: Kill threshold in Celsius. Must be > 0.
            resume_threshold_c: Re-launch threshold in Celsius. Must be > 0
                and < ``pause_threshold_c``.
            critical_threshold_c: Optional permanent-abort / CPU-switch
                threshold. Must be > 0 when provided.
            poll_interval_seconds: Seconds between temperature polls. > 0.
            checkpoint_dir: Directory where the child writes rolling
                checkpoints and the checkpoint-done sentinel file.
            checkpoint_grace_seconds: Grace window after SIGUSR1 before
                SIGKILL. Must be > 0.
            switch_on_thermal: If True, critical temp triggers CPU
                continuation instead of permanent abort.
            child_argv: Argument vector used to (re)launch the training child.
                The supervisor prepends ``sys.executable -u`` if the first
                element is not already an interpreter path.
            child_env: Optional environment override for the child. If
                ``None``, ``os.environ`` is used (with the supervisor-child
                marker added).
            log_fn: Callable used for log messages (defaults to
                ``logging.warning``).

        Raises:
            ValueError: If threshold / interval / grace constraints are
                violated.
        """
        self._logger = logging.getLogger(__name__)
        self._log = log_fn
        self._device = str(device or "cpu")
        self._enabled = bool(enabled)
        self._nvml_device_index = int(nvml_device_index)
        self._pause_threshold_c = float(pause_threshold_c)
        self._resume_threshold_c = float(resume_threshold_c)
        self._critical_threshold_c = (
            None if critical_threshold_c is None else float(critical_threshold_c)
        )
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._checkpoint_dir = str(checkpoint_dir or "checkpoints")
        self._checkpoint_grace_seconds = float(checkpoint_grace_seconds)
        self._switch_on_thermal = bool(switch_on_thermal)
        self._child_argv = list(child_argv or [])
        self._child_env = dict(child_env) if child_env is not None else None

        if self._pause_threshold_c <= 0:
            raise ValueError("pause_threshold_c must be > 0")
        if self._resume_threshold_c <= 0:
            raise ValueError("resume_threshold_c must be > 0")
        if self._resume_threshold_c >= self._pause_threshold_c:
            raise ValueError("resume_threshold_c must be < pause_threshold_c")
        if self._poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        if self._checkpoint_grace_seconds <= 0:
            raise ValueError("checkpoint_grace_seconds must be > 0")
        if self._critical_threshold_c is not None and self._critical_threshold_c <= 0:
            raise ValueError("critical_threshold_c must be > 0 when provided")

        self._active = (
            self._enabled
            and self._device.startswith("cuda")
            and torch is not None
            and torch.cuda.is_available()
        )

        # Temperature probe reused for both the running-child monitor and the
        # cooldown wait. It only activates on CUDA.
        self._probe = GPUTemperatureGuard(
            enabled=self._enabled,
            device=self._device,
            nvml_device_index=self._nvml_device_index,
            pause_threshold_c=self._pause_threshold_c,
            resume_threshold_c=self._resume_threshold_c,
            critical_threshold_c=self._critical_threshold_c,
            poll_interval_seconds=self._poll_interval_seconds,
        )

        self.total_pause_events = 0
        self.total_paused_seconds = 0.0

    # -- public properties --------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def pause_threshold_c(self) -> float:
        return self._pause_threshold_c

    @property
    def resume_threshold_c(self) -> float:
        return self._resume_threshold_c

    @property
    def critical_threshold_c(self) -> Optional[float]:
        return self._critical_threshold_c

    @property
    def poll_interval_seconds(self) -> float:
        return self._poll_interval_seconds

    @property
    def checkpoint_grace_seconds(self) -> float:
        return self._checkpoint_grace_seconds

    @property
    def switch_on_thermal(self) -> bool:
        return self._switch_on_thermal

    # -- temperature helpers ------------------------------------------------
    def read_temperature_c(self) -> float:
        """Read the current GPU temperature via the probe.

        Returns:
            Temperature in Celsius, or ``0.0`` if the supervisor is inactive.
        """
        return self._probe.read_temperature_c()

    # -- supervisor loop ----------------------------------------------------
    def run(self) -> int:
        """Run the supervisor loop until the child finishes or a permanent abort.

        Returns:
            ``0`` on successful completion, ``2`` on permanent thermal abort,
            or the child's non-zero exit code on crash.
        """
        if not self._active:
            # Not a CUDA device or disabled: run the child directly in-process
            # of the supervisor (no temperature monitoring).
            if not self._child_argv:
                self._log("[Supervisor] inactive and no child argv configured; nothing to do")
                return 0
            self._log("[Supervisor] inactive (non-CUDA); launching child without thermal monitoring")
            return self._launch_and_wait(device_override=None)

        cpu_mode = False  # Once True, we stop polling temp and just wait for the CPU child.

        while True:
            # Don't spawn into an already-hot GPU: cool first.
            temp_c = self._read_temp_or_none()
            if temp_c is not None and temp_c >= self._pause_threshold_c:
                self._log(
                    f"[Supervisor] GPU already at {temp_c:.1f}C >= {self._pause_threshold_c:.1f}C "
                    f"before launch; cooling to {self._resume_threshold_c:.1f}C"
                )
                if not self._wait_below(self._resume_threshold_c, allow_abort=True):
                    return 2

            device_override = "cpu" if cpu_mode else None
            cur = self._read_temp_or_none()
            self._log(
                f"[Supervisor] launching {'CPU' if cpu_mode else 'GPU'} child "
                f"(temp {cur if cur is not None else '?'}C)"
            )
            rc = self._launch_and_wait(device_override=device_override, monitor_temp=not cpu_mode)
            if rc == 0:
                self._log("[Supervisor] child completed successfully")
                return 0
            if rc < 0:
                # Negative return codes are supervisor-internal signals.
                if rc == -1:  # thermal pause -> cooldown + relaunch
                    if not self._wait_below(self._resume_threshold_c, allow_abort=True):
                        return 2
                    self._log(
                        f"[Supervisor] RESUME: re-launching GPU child (temp <= {self._resume_threshold_c:.1f}C)"
                    )
                    continue
                if rc == -2:  # critical -> CPU switch
                    self._log(
                        "[Supervisor] critical threshold reached with switch_on_thermal=True; "
                        "switching to CPU continuation"
                    )
                    cpu_mode = True
                    continue
                if rc == -3:  # permanent abort
                    return 2
                # Unknown internal signal: treat as crash.
                self._log(f"[Supervisor] unexpected internal signal {rc}; aborting")
                return 2
            # Non-zero child exit (crash).
            self._log(f"[Supervisor] child exited unexpectedly (code {rc}); aborting")
            return rc

    # -- internals ----------------------------------------------------------
    def _read_temp_or_none(self) -> Optional[float]:
        if not self._active:
            return None
        try:
            return float(self._probe.read_temperature_c())
        except GPUTelemetryError as exc:
            self._log(f"[Supervisor] temperature read failed: {exc}")
            return None

    def _build_child_argv(self, device_override: Optional[str]) -> List[str]:
        """Build the argv for (re)launching the child.

        Ensures the child runs under the Python interpreter, disables its own
        guard (``--no-gpu-temp-guard``) to avoid nesting, and asks it to resume
        from the latest rolling checkpoint (``--resume-from-checkpoint auto``)
        unless the caller already supplied a resume flag.
        """
        argv = list(self._child_argv)
        # Ensure interpreter prefix.
        if not argv or not _looks_like_interpreter(argv[0]):
            argv = [sys.executable, "-u"] + argv

        # Make sure the child knows it is a supervisor child and disables its
        # own in-process guard. We append overriding flags at the end so they
        # win over any earlier duplicates (argparse takes the last value for
        # non-append arguments).
        if "--no-gpu-temp-guard" not in argv and "--gpu-temp-guard" not in argv:
            argv.append("--no-gpu-temp-guard")

        if "--resume-from-checkpoint" not in argv:
            argv.extend(["--resume-from-checkpoint", RESUME_AUTO])

        if device_override is not None:
            argv.extend(["--device", device_override])

        return argv

    def _build_child_env(self) -> dict:
        env = dict(self._child_env) if self._child_env is not None else os.environ.copy()
        env[SUPERVISOR_CHILD_ENV_VAR] = "1"
        return env

    def _launch_and_wait(
        self,
        *,
        device_override: Optional[str] = None,
        monitor_temp: bool = True,
    ) -> int:
        """Launch the child and monitor it.

        Returns:
            ``0`` if the child exits cleanly, the child's non-zero exit code
            on crash, or a negative supervisor-internal signal:
            ``-1`` (thermal pause -> cooldown + relaunch), ``-2`` (critical
            with switch_on_thermal -> CPU continuation), ``-3`` (permanent
            abort).
        """
        argv = self._build_child_argv(device_override=device_override)
        env = self._build_child_env()

        try:
            proc = subprocess.Popen(
                argv,
                env=env,
                start_new_session=True,
            )
        except Exception as exc:
            self._log(f"[Supervisor] failed to launch child: {exc}")
            return 1

        killed_by_us = False
        completed = False

        while True:
            if monitor_temp and self._active:
                temp_c = self._read_temp_or_none()
                if temp_c is not None:
                    if (
                        self._critical_threshold_c is not None
                        and temp_c >= self._critical_threshold_c
                    ):
                        self._log(
                            f"[Supervisor] CRITICAL: {temp_c:.1f}C >= {self._critical_threshold_c:.1f}C"
                        )
                        self._request_checkpoint(proc)
                        _kill_process_group(proc)
                        killed_by_us = True
                        if self._switch_on_thermal:
                            return -2
                        return -3
                    if temp_c >= self._pause_threshold_c:
                        self._log(
                            f"[Supervisor] PAUSE: {temp_c:.1f}C >= {self._pause_threshold_c:.1f}C "
                            f"- checkpointing then killing GPU child"
                        )
                        self._request_checkpoint(proc)
                        _kill_process_group(proc)
                        killed_by_us = True
                        return -1

            rc = proc.poll()
            if rc is not None:
                if killed_by_us:
                    break
                if rc == 0:
                    completed = True
                else:
                    return rc
                break

            time.sleep(self._poll_interval_seconds if monitor_temp else 1.0)

        if completed:
            return 0
        # We killed the child for a thermal pause; the caller handles cooldown.
        return -1

    def _request_checkpoint(self, proc: "subprocess.Popen") -> None:
        """Send SIGUSR1 to the child and wait (briefly) for the sentinel file."""
        sentinel = os.path.join(self._checkpoint_dir, CHECKPOINT_DONE_SENTINEL)
        # Clear any stale sentinel.
        try:
            if os.path.exists(sentinel):
                os.remove(sentinel)
        except OSError:
            pass

        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGUSR1)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            self._log(f"[Supervisor] could not send SIGUSR1 to child: {exc}")
            return

        deadline = time.perf_counter() + self._checkpoint_grace_seconds
        while time.perf_counter() < deadline:
            if proc.poll() is not None:
                return
            if os.path.exists(sentinel):
                self._log("[Supervisor] child acknowledged checkpoint flush")
                # Clean up the sentinel so the next cycle starts fresh.
                try:
                    os.remove(sentinel)
                except OSError:
                    pass
                return
            time.sleep(0.5)
        self._log(
            f"[Supervisor] checkpoint grace window ({self._checkpoint_grace_seconds:.0f}s) "
            "expired; killing without explicit acknowledgement"
        )

    def _wait_below(self, threshold_c: float, *, allow_abort: bool) -> bool:
        """Block until temperature <= ``threshold_c``.

        Args:
            threshold_c: Target temperature.
            allow_abort: If True, return False (permanent abort) when the
                critical threshold is reached during cooldown.

        Returns:
            True when cooled below ``threshold_c``; False on permanent abort.
        """
        started = time.perf_counter()
        while True:
            temp_c = self._read_temp_or_none()
            if temp_c is not None:
                if (
                    allow_abort
                    and self._critical_threshold_c is not None
                    and temp_c >= self._critical_threshold_c
                ):
                    self._log(
                        f"[Supervisor] STOP during cooldown: {temp_c:.1f}C >= "
                        f"{self._critical_threshold_c:.1f}C - aborting"
                    )
                    return False
                if temp_c <= threshold_c:
                    duration = time.perf_counter() - started
                    self.total_pause_events += 1
                    self.total_paused_seconds += duration
                    self._log(
                        f"[Supervisor] cooled to {temp_c:.1f}C after {duration:.1f}s"
                    )
                    return True
                self._log(
                    f"[Supervisor] cooling: {temp_c:.1f}C (target <= {threshold_c:.1f}C)"
                )
            time.sleep(self._poll_interval_seconds)


def _looks_like_interpreter(path: str) -> bool:
    """Heuristic: does *path* look like a Python interpreter invocation?"""
    if not path:
        return False
    base = os.path.basename(str(path)).lower()
    return base in {"python", "python3", "python3.9", "python3.10", "python3.11", "python3.12"} or "python" in base