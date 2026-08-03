"""Extended unit tests for GPUTemperatureGuard."""
import subprocess
import unittest
from importlib.util import find_spec
from unittest.mock import patch

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    from src.utils.gpu_temp_guard import GPUTelemetryError, GPUTemperatureGuard, GPUTempCheckResult


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GPUTempGuardInitTests(unittest.TestCase):
    def _guard(self, **kw):
        defaults = dict(
            enabled=False,
            device="cpu",
            pause_threshold_c=90.0,
            resume_threshold_c=80.0,
            poll_interval_seconds=30.0,
        )
        defaults.update(kw)
        return GPUTemperatureGuard(**defaults)

    def test_properties_match_constructor_args(self):
        g = self._guard(pause_threshold_c=88.0, resume_threshold_c=75.0, poll_interval_seconds=10.0)
        self.assertAlmostEqual(g.pause_threshold_c, 88.0)
        self.assertAlmostEqual(g.resume_threshold_c, 75.0)
        self.assertAlmostEqual(g.poll_interval_seconds, 10.0)

    def test_critical_threshold_none_by_default(self):
        g = self._guard()
        self.assertIsNone(g.critical_threshold_c)

    def test_critical_threshold_stored_when_provided(self):
        g = self._guard(critical_threshold_c=95.0)
        self.assertAlmostEqual(float(g.critical_threshold_c), 95.0)

    def test_pause_threshold_zero_raises(self):
        with self.assertRaises(ValueError):
            self._guard(pause_threshold_c=0.0, resume_threshold_c=-5.0)

    def test_resume_threshold_zero_raises(self):
        with self.assertRaises(ValueError):
            self._guard(resume_threshold_c=0.0)

    def test_resume_gte_pause_raises(self):
        with self.assertRaises(ValueError):
            self._guard(pause_threshold_c=80.0, resume_threshold_c=90.0)

    def test_resume_equal_pause_raises(self):
        with self.assertRaises(ValueError):
            self._guard(pause_threshold_c=80.0, resume_threshold_c=80.0)

    def test_poll_interval_zero_raises(self):
        with self.assertRaises(ValueError):
            self._guard(poll_interval_seconds=0.0)

    def test_negative_critical_threshold_raises(self):
        with self.assertRaises(ValueError):
            self._guard(critical_threshold_c=-1.0)

    def test_inactive_when_disabled(self):
        g = self._guard(enabled=False)
        self.assertFalse(g.is_active)

    def test_inactive_on_cpu_even_if_enabled(self):
        g = self._guard(enabled=True, device="cpu")
        self.assertFalse(g.is_active)

    def test_initial_counters_zero(self):
        g = self._guard()
        self.assertEqual(g.total_pause_events, 0)
        self.assertAlmostEqual(g.total_paused_seconds, 0.0)
        self.assertIsNone(g.last_temperature_c)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GPUTempGuardReadTempTests(unittest.TestCase):
    def _inactive_guard(self):
        return GPUTemperatureGuard(enabled=False, device="cpu")

    def test_inactive_guard_returns_zero_and_no_update(self):
        g = self._inactive_guard()
        temp = g.read_temperature_c()
        self.assertAlmostEqual(temp, 0.0)
        self.assertIsNone(g.last_temperature_c)

    def test_nvml_backend_unavailable_falls_back_to_nvidia_smi(self):
        g = self._inactive_guard()
        g._active = True
        g._nvml_disabled = True  # Force NVML path to skip
        with patch.object(g, "_read_temperature_from_nvidia_smi", return_value=72.0):
            temp = g.read_temperature_c()
        self.assertAlmostEqual(temp, 72.0)
        self.assertAlmostEqual(g.last_temperature_c, 72.0)

    def test_nvidia_smi_timeout_raises_telemetry_error(self):
        g = self._inactive_guard()
        with patch(
            "src.utils.gpu_temp_guard.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["nvidia-smi"], 5),
        ):
            with self.assertRaises(GPUTelemetryError) as ctx:
                g._read_temperature_from_nvidia_smi()
        self.assertIn("timed out", str(ctx.exception))

    def test_nvidia_smi_file_not_found_raises_telemetry_error(self):
        g = self._inactive_guard()
        with patch(
            "src.utils.gpu_temp_guard.subprocess.run",
            side_effect=FileNotFoundError("nvidia-smi not found"),
        ):
            with self.assertRaises(GPUTelemetryError) as ctx:
                g._read_temperature_from_nvidia_smi()
        self.assertIn("nvidia-smi", str(ctx.exception))

    def test_nvidia_smi_empty_output_raises_telemetry_error(self):
        import subprocess as sp

        class _FakeResult:
            stdout = ""
            returncode = 0

        g = self._inactive_guard()
        with patch("src.utils.gpu_temp_guard.subprocess.run", return_value=_FakeResult()):
            with self.assertRaises(GPUTelemetryError):
                g._read_temperature_from_nvidia_smi()


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GPUTempGuardParseTests(unittest.TestCase):
    def _g(self):
        return GPUTemperatureGuard(enabled=False, device="cpu")

    def test_parse_valid_integer(self):
        self.assertAlmostEqual(self._g()._parse_temperature("72"), 72.0)

    def test_parse_valid_float(self):
        self.assertAlmostEqual(self._g()._parse_temperature("72.5"), 72.5)

    def test_parse_value_with_units(self):
        # "72 C" is a valid pattern
        self.assertAlmostEqual(self._g()._parse_temperature("72 C"), 72.0)

    def test_parse_na_raises(self):
        with self.assertRaises(GPUTelemetryError):
            self._g()._parse_temperature("N/A")

    def test_parse_not_supported_raises(self):
        with self.assertRaises(GPUTelemetryError):
            self._g()._parse_temperature("[Not Supported]")

    def test_parse_empty_string_raises(self):
        with self.assertRaises(GPUTelemetryError):
            self._g()._parse_temperature("")

    def test_parse_non_numeric_raises(self):
        with self.assertRaises(GPUTelemetryError):
            self._g()._parse_temperature("UNKNOWN_TEMP")

    def test_parse_zero_or_negative_raises(self):
        with self.assertRaises(GPUTelemetryError):
            self._g()._parse_temperature("0")


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GPUTempCheckResultTests(unittest.TestCase):
    def test_repair_action_none_when_not_paused(self):
        r = GPUTempCheckResult(temp_c=70.0, paused=False, pause_duration_s=0.0, checks_during_pause=0, critical_seen=False)
        self.assertEqual(r.repair_action, "none")

    def test_repair_action_thermal_pause_when_paused(self):
        r = GPUTempCheckResult(temp_c=79.0, paused=True, pause_duration_s=30.0, checks_during_pause=1, critical_seen=False)
        self.assertIn("thermal_pause", r.repair_action)
        self.assertIn("30", r.repair_action)

    def test_repair_action_critical_when_critical_seen(self):
        r = GPUTempCheckResult(temp_c=79.0, paused=True, pause_duration_s=60.0, checks_during_pause=2, critical_seen=True)
        self.assertIn("critical", r.repair_action)

    def test_repair_action_rounds_duration(self):
        r = GPUTempCheckResult(temp_c=79.0, paused=True, pause_duration_s=45.7, checks_during_pause=1, critical_seen=False)
        self.assertIn("46", r.repair_action)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GPUTempGuardWaitTests(unittest.TestCase):
    def test_below_threshold_returns_immediately(self):
        g = GPUTemperatureGuard(enabled=False, device="cpu", pause_threshold_c=90.0, resume_threshold_c=80.0)
        g._active = True
        with patch.object(g, "read_temperature_c", return_value=85.0):
            result = g.wait_until_safe()
        self.assertFalse(result.paused)
        self.assertAlmostEqual(result.temp_c, 85.0)

    def test_above_threshold_pauses_then_resumes(self):
        g = GPUTemperatureGuard(enabled=False, device="cpu", pause_threshold_c=90.0, resume_threshold_c=80.0, poll_interval_seconds=30.0)
        g._active = True
        with (
            patch.object(g, "read_temperature_c", side_effect=[92.0, 88.0, 78.0]),
            patch("src.utils.gpu_temp_guard.time.sleep", return_value=None),
        ):
            result = g.wait_until_safe()
        self.assertTrue(result.paused)
        self.assertEqual(result.checks_during_pause, 2)

    def test_inactive_guard_returns_zero_result(self):
        g = GPUTemperatureGuard(enabled=False, device="cpu")
        result = g.wait_until_safe()
        self.assertFalse(result.paused)
        self.assertAlmostEqual(result.temp_c, 0.0)
        self.assertEqual(result.checks_during_pause, 0)

    def test_critical_seen_set_when_temp_exceeds_critical(self):
        g = GPUTemperatureGuard(
            enabled=False,
            device="cpu",
            pause_threshold_c=90.0,
            resume_threshold_c=80.0,
            critical_threshold_c=95.0,
        )
        g._active = True
        with (
            patch.object(g, "read_temperature_c", side_effect=[96.0, 79.0]),
            patch("src.utils.gpu_temp_guard.time.sleep", return_value=None),
        ):
            result = g.wait_until_safe()
        self.assertTrue(result.critical_seen)

    def test_pause_events_counter_incremented(self):
        g = GPUTemperatureGuard(enabled=False, device="cpu", pause_threshold_c=90.0, resume_threshold_c=80.0)
        g._active = True
        with (
            patch.object(g, "read_temperature_c", side_effect=[91.0, 79.0]),
            patch("src.utils.gpu_temp_guard.time.sleep", return_value=None),
        ):
            g.wait_until_safe()
        self.assertEqual(g.total_pause_events, 1)


if __name__ == "__main__":
    unittest.main()
