import os
import signal
import subprocess
import unittest
from importlib.util import find_spec
from unittest.mock import patch

TORCH_AVAILABLE = find_spec("torch") is not None
if TORCH_AVAILABLE:
    from src.utils.gpu_temp_guard import (
        GPUTelemetryError,
        GPUTemperatureGuard,
        GPUTempSupervisor,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for gpu temp guard tests")
class GpuTempGuardCompatShimTests(unittest.TestCase):
    """The legacy GPUTemperatureGuard is kept as a thin temperature-probe shim."""

    def test_public_threshold_properties(self):
        guard = GPUTemperatureGuard(
            enabled=False,
            device="cpu",
            pause_threshold_c=91.0,
            resume_threshold_c=79.0,
            critical_threshold_c=95.0,
            poll_interval_seconds=12.5,
        )
        self.assertAlmostEqual(guard.pause_threshold_c, 91.0)
        self.assertAlmostEqual(guard.resume_threshold_c, 79.0)
        self.assertAlmostEqual(float(guard.critical_threshold_c), 95.0)
        self.assertAlmostEqual(guard.poll_interval_seconds, 12.5)

    def test_inactive_read_temperature_returns_zero(self):
        guard = GPUTemperatureGuard(enabled=False, device="cpu")
        self.assertFalse(guard.is_active)
        self.assertAlmostEqual(guard.read_temperature_c(), 0.0)

    def test_wait_until_safe_inactive_returns_not_paused(self):
        guard = GPUTemperatureGuard(enabled=False, device="cpu")
        result = guard.wait_until_safe(context="unit-test")
        self.assertFalse(result.paused)

    def test_nvidia_smi_failure_is_fatal(self):
        from src.utils.gpu_temp_guard import read_temperature_from_nvidia_smi
        with patch(
            "src.utils.gpu_temp_guard.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="gpu lost"),
        ):
            with self.assertRaises(GPUTelemetryError):
                read_temperature_from_nvidia_smi(0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for gpu temp guard tests")
class GPUTempSupervisorInitTests(unittest.TestCase):
    def _sup(self, **kw):
        defaults = dict(
            enabled=False,
            device="cpu",
            pause_threshold_c=90.0,
            resume_threshold_c=80.0,
            poll_interval_seconds=30.0,
            checkpoint_grace_seconds=15.0,
            child_argv=["-m", "src.cli", "train", "--config", "x"],
        )
        defaults.update(kw)
        return GPUTempSupervisor(**defaults)

    def test_threshold_properties(self):
        sup = self._sup(
            pause_threshold_c=88.0,
            resume_threshold_c=75.0,
            poll_interval_seconds=10.0,
            checkpoint_grace_seconds=20.0,
            critical_threshold_c=95.0,
        )
        self.assertAlmostEqual(sup.pause_threshold_c, 88.0)
        self.assertAlmostEqual(sup.resume_threshold_c, 75.0)
        self.assertAlmostEqual(sup.poll_interval_seconds, 10.0)
        self.assertAlmostEqual(sup.checkpoint_grace_seconds, 20.0)
        self.assertAlmostEqual(float(sup.critical_threshold_c), 95.0)
        self.assertFalse(sup.switch_on_thermal)

    def test_invalid_thresholds_raise(self):
        with self.assertRaises(ValueError):
            self._sup(pause_threshold_c=0)
        with self.assertRaises(ValueError):
            self._sup(resume_threshold_c=0)
        with self.assertRaises(ValueError):
            self._sup(pause_threshold_c=80.0, resume_threshold_c=80.0)
        with self.assertRaises(ValueError):
            self._sup(poll_interval_seconds=0)
        with self.assertRaises(ValueError):
            self._sup(checkpoint_grace_seconds=0)
        with self.assertRaises(ValueError):
            self._sup(critical_threshold_c=0)

    def test_inactive_on_cpu_device(self):
        sup = self._sup(device="cpu")
        self.assertFalse(sup.is_active)

    def test_inactive_read_temperature_returns_zero(self):
        sup = self._sup(device="cpu")
        self.assertAlmostEqual(sup.read_temperature_c(), 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for gpu temp guard tests")
class GPUTempSupervisorRunTests(unittest.TestCase):
    """Supervisor run() behaviour with mocked Popen and temperature."""

    def _make_supervisor(self, temps, switch_on_thermal=False, critical=None, pause=90.0, resume=80.0):
        sup = GPUTempSupervisor(
            enabled=True,
            device="cuda",
            pause_threshold_c=pause,
            resume_threshold_c=resume,
            critical_threshold_c=critical,
            poll_interval_seconds=0.01,
            checkpoint_grace_seconds=0.05,
            checkpoint_dir="/tmp/opencode/_sup_test_ckpt",
            switch_on_thermal=switch_on_thermal,
            child_argv=["-m", "dummy", "train"],
        )
        # Force active even without a real GPU.
        sup._active = True
        sup._probe._active = True
        os.makedirs(sup._checkpoint_dir, exist_ok=True)
        temp_iter = iter(temps)

        def fake_read():
            try:
                return float(next(temp_iter))
            except StopIteration:
                return float(temps[-1])

        sup._probe.read_temperature_c = fake_read
        sup._read_temp_or_none = lambda: fake_read()
        return sup

    def test_child_completes_cleanly(self):
        # run() reads: precheck, cur-log, then per monitor cycle: 1 read + 1 poll.
        # precheck=70, cur-log=70, monitor reads 70, poll returns 0 → done.
        sup = self._make_supervisor(temps=[70.0, 70.0, 70.0])

        class FakePopen:
            def __init__(self, *a, **kw):
                self.pid = 1

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        with patch("src.utils.gpu_temp_guard.subprocess.Popen", side_effect=FakePopen), \
             patch("src.utils.gpu_temp_guard._kill_process_group") as kill, \
             patch("src.utils.gpu_temp_guard.time.sleep"):
            rc = sup.run()
        self.assertEqual(rc, 0)
        kill.assert_not_called()

    def test_thermal_pause_then_resume(self):
        # Cycle 1: precheck=70, cur-log=70, monitor=95 → kill → return -1.
        # Cooldown: reads 75 (≤80) → True.
        # Cycle 2: precheck=70, cur-log=70, monitor=70, poll=0 → done.
        sup = self._make_supervisor(
            temps=[70.0, 70.0, 95.0, 75.0, 70.0, 70.0, 70.0],
            pause=90.0,
            resume=80.0,
        )
        relaunches = {"n": 0}

        class FakePopen:
            def __init__(self, *a, **kw):
                relaunches["n"] += 1
                self.pid = 1

            def poll(self):
                # Launch 1: never exits on its own (supervisor kills it).
                # Launch 2: exits cleanly.
                return None if relaunches["n"] == 1 else 0

            def wait(self, timeout=None):
                return 0

        with patch("src.utils.gpu_temp_guard.subprocess.Popen", side_effect=FakePopen), \
             patch("src.utils.gpu_temp_guard._kill_process_group") as kill, \
             patch("src.utils.gpu_temp_guard.time.sleep"), \
             patch("src.utils.gpu_temp_guard.os.killpg") as killpg, \
             patch("src.utils.gpu_temp_guard.os.getpgid", return_value=1), \
             patch("src.utils.gpu_temp_guard.os.path.exists", return_value=False):
            rc = sup.run()
        self.assertEqual(rc, 0)
        self.assertEqual(relaunches["n"], 2)
        self.assertGreaterEqual(sup.total_pause_events, 1)

    def test_critical_permanent_abort(self):
        # precheck=70, cur-log=70, monitor=100 ≥ critical 95 → kill → return -3 → abort.
        sup = self._make_supervisor(
            temps=[70.0, 70.0, 100.0, 100.0],
            critical=95.0,
            switch_on_thermal=False,
        )

        class FakePopen:
            def __init__(self, *a, **kw):
                self.pid = 1

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

        with patch("src.utils.gpu_temp_guard.subprocess.Popen", side_effect=FakePopen), \
             patch("src.utils.gpu_temp_guard._kill_process_group") as kill, \
             patch("src.utils.gpu_temp_guard.time.sleep"), \
             patch("src.utils.gpu_temp_guard.os.killpg"), \
             patch("src.utils.gpu_temp_guard.os.getpgid", return_value=1), \
             patch("src.utils.gpu_temp_guard.os.path.exists", return_value=False):
            rc = sup.run()
        self.assertEqual(rc, 2)
        kill.assert_called()

    def test_critical_switch_on_thermal_continues_on_cpu(self):
        # Cycle 1: precheck=70, cur-log=70, monitor=100 ≥ critical 95 → kill → -2 (CPU).
        # Cooldown: reads 70 → True.
        # Cycle 2 (CPU): precheck=70 (GPU still cool, no-op), cur-log=70, monitor=70,
        #   poll=0 → done.
        sup = self._make_supervisor(
            temps=[70.0, 70.0, 100.0, 70.0, 70.0, 70.0, 70.0],
            critical=95.0,
            switch_on_thermal=True,
        )
        launches = {"n": 0, "devices": []}

        class FakePopen:
            def __init__(self, *a, **kw):
                launches["n"] += 1
                self.pid = 1
                argv = a[0] if a else kw.get("args", [])
                if "--device" in argv:
                    idx = argv.index("--device")
                    launches["devices"].append(argv[idx + 1])
                else:
                    launches["devices"].append(None)

            def poll(self):
                return None if launches["n"] == 1 else 0

            def wait(self, timeout=None):
                return 0

        with patch("src.utils.gpu_temp_guard.subprocess.Popen", side_effect=FakePopen), \
             patch("src.utils.gpu_temp_guard._kill_process_group") as kill, \
             patch("src.utils.gpu_temp_guard.time.sleep"), \
             patch("src.utils.gpu_temp_guard.os.killpg"), \
             patch("src.utils.gpu_temp_guard.os.getpgid", return_value=1), \
             patch("src.utils.gpu_temp_guard.os.path.exists", return_value=False):
            rc = sup.run()
        self.assertEqual(rc, 0)
        self.assertEqual(launches["n"], 2)
        # Second launch (CPU continuation) should include --device cpu.
        self.assertIn("cpu", launches["devices"])


if __name__ == "__main__":
    unittest.main()