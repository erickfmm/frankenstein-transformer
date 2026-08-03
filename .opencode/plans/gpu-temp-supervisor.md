# GPU Temperature Supervisor — CPU-side kill+restart with checkpoint resume

## Goal
Replace the in-process `GPUTemperatureGuard` (which only pauses the training loop
via `wait_until_safe`) with a **CPU-side supervisor** that owns the training
subprocess lifecycle. The supervisor monitors GPU temp via `nvidia-smi`/NVML
every N seconds, kills the GPU subprocess on thermal events (after first
flushing a rolling checkpoint via SIGUSR1), waits for cooldown, and re-launches
the child which resumes from the latest rolling checkpoint.

## Supervisor state machine

```
                  ┌─────────────────────────────────────────────────┐
                  │  spawn child (GPU) with --resume-from-checkpoint │
                  └─────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌──────────────────────────────────────────────────┐
        │  monitor loop: every poll_interval_seconds        │
        │    read temp; check child exit status             │
        └──────────────────────────────────────────────────┘
            │              │                │              │
   temp≥crit   temp≥pause    child exited    child crashed
            │              │                │              │
            ▼              ▼                ▼              ▼
   [see crit   SIGUSR1      rc==0 →          rc!=0 →
    branch]    +grace       return 0         return rc
              +SIGKILL
              +cooldown
              +re-launch
```

### Critical threshold branch (`temp >= gpu_temp_critical_threshold_c`)
- **`switch_on_thermal == false`** (default): kill child, **permanent abort**
  (return code 2). Training stops entirely — hardware is unsafe.
- **`switch_on_thermal == true`**: SIGUSR1 (checkpoint) → grace window →
  SIGKILL → re-launch child with `--device cpu` (resume from checkpoint on
  CPU). The CPU child runs to completion; supervisor stops monitoring temp
  (no GPU in use) and just waits for the child to exit.

### Pause threshold branch (`pause ≤ temp < critical`)
Always: SIGUSR1 → grace window (`gpu_temp_checkpoint_grace_seconds`, default 30s)
→ SIGKILL → wait until `temp ≤ resume_threshold_c` → re-launch on GPU with
`--resume-from-checkpoint auto`.

## Checkpoint-resume mechanism
- Supervisor sends **SIGUSR1** to the direct child process to request a
  rolling checkpoint flush.
- Trainer registers a SIGUSR1 handler that sets `_checkpoint_request_pending`
  flag; the training loop checks this flag each step and, when set, calls
  `_save_rolling_checkpoint` immediately then clears the flag and writes a
  sentinel file `checkpoints/.supervisor_checkpoint_done` to signal readiness
  (so the supervisor knows the flush completed before SIGKILL).
- Child is always launched as a **direct child** (`start_new_session=True`,
  `os.setsid`), so `os.killpg` works and SIGUSR1 reaches it.
- New CLI arg `--resume-from-checkpoint <path|auto>`: when `auto`, the trainer
  globs `checkpoints/titan_rolling_step_*.pt`, picks the highest `global_step`,
  and calls `load_checkpoint` before training starts, then skips to the saved
  epoch/step.

## Files to modify

### `src/utils/gpu_temp_guard.py` (rewrite)
- New primary class: `GPUTempSupervisor` (CPU-only — does NOT import torch at
  module top; uses `nvidia-smi`/NVML via the existing parsing helpers).
- Keep `GPUTelemetryError`, `read_temperature_c` (NVML + nvidia-smi), and
  `_parse_temperature` helpers.
- `GPUTempSupervisor.__init__(*, enabled, nvml_device_index,
  pause_threshold_c, resume_threshold_c, critical_threshold_c=None,
  poll_interval_seconds, checkpoint_dir, checkpoint_grace_seconds,
  switch_on_thermal, child_argv, child_env=None, log_fn=print)`.
- `run() -> int`: the supervisor loop described above.
- `_kill_process_group(proc)`, `_request_checkpoint(proc)` (SIGUSR1 +
  sentinel-file wait), `_wait_below(threshold)`, `_launch_child(device_override)`.
- Keep `GPUTemperatureGuard` as a **thin compat shim** that exposes
  `read_temperature_c` + threshold properties (for tests that still construct
  it directly). Mark deprecated.

### `src/training/trainer.py`
- Add `resume_from_checkpoint: Optional[str] = None` to `TrainingConfig`.
- Add `gpu_temp_checkpoint_grace_seconds: float = 30.0` to `TrainingConfig`.
- Register SIGUSR1 handler in `TitanTrainer.__init__` (sets
  `_checkpoint_request_pending=True`).
- In the training step loop, check `_checkpoint_request_pending`; if set,
  flush `_save_rolling_checkpoint`, write sentinel file, clear flag.
- Wire `--resume-from-checkpoint auto|<path>`: on startup, if set, find
  latest rolling checkpoint and call `load_checkpoint`, then resume from the
  saved epoch/step (skip already-trained batches).
- **Remove** the in-process thermal guard code that the supervisor replaces:
  `_enforce_thermal_guard`, `_handle_critical_thermal_offload`,
  `_monitor_thermal_offload_and_maybe_reload`, `_wait_for_gpu_resume_temperature`,
  `_switch_to_cpu_only_after_gpu_error`, `_offload_training_state_to_cpu`
  (the supervisor handles all of this by killing+respawning).
- Keep `switch_on_thermal` in `TrainingConfig` — it's passed to the supervisor
  to decide critical-temp behavior (CPU continue vs permanent abort).
- The `gpu_temp_guard` instance on the trainer becomes a no-op passthrough
  when running under the supervisor (env `FRANKESTEIN_SUPERVISOR_CHILD=1`).

### `src/training/main.py`
- In `main()`, if `training_config.gpu_temp_guard_enabled` and CUDA device:
  - Build `child_argv` = `[sys.executable, "-u", "-m", "src.cli", "train",
    "--config", <path>, "--resume-from-checkpoint", "auto",
    "--no-gpu-temp-guard", ...]` with env `FRANKESTEIN_SUPERVISOR_CHILD=1`.
  - Construct `GPUTempSupervisor` and `return supervisor.run()`.
- Add `--resume-from-checkpoint` CLI arg.
- `_run_sbert_task`: when guard enabled, spawn SBERT as a subprocess under
  the supervisor too (same pattern — build argv, wrap in supervisor). The
  SBERT child gets `--no-gpu-temp-guard` to avoid nesting.
- Update `_validate_gpu_temp_guard_config` (keep; add grace-seconds > 0 check).
- Remove the now-dead `switch_on_thermal` auto-enable logic (the supervisor
  reads `switch_on_thermal` directly).

### `src/sbert/train_sbert.py`
- Register SIGUSR1 handler for rolling-checkpoint flush (same pattern as MLM
  trainer).
- Add `--resume-from-checkpoint` arg; on startup, load latest rolling
  SBERT checkpoint and resume.
- Remove the in-process `_handle_critical_thermal_offload`,
  `_monitor_thermal_offload`, `_save_thermal_resume_artifact` code (supervisor
  replaces it). The `gpu_temp_guard` instance becomes a no-op when running
  under the supervisor.

### `src/cli.py`
- Add `--resume-from-checkpoint` arg to the `train` subcommand.
- Keep `--gpu-temp-guard` / `--no-gpu-temp-guard` (now controls the supervisor).
- Keep `--switch-on-thermal` / `--no-switch-on-thermal` (passed to supervisor).
- Keep threshold override flags.

### `src/schema/_training.yaml`
- Update `gpu_temp_guard_enabled` description → supervisor semantics.
- Keep `switch_on_thermal` — description updated: when true, critical temp
  triggers CPU continuation (resume on CPU); when false, critical temp =
  permanent abort.
- Update `gpu_temp_critical_threshold_c` description → permanent-abort OR
  CPU-switch threshold depending on `switch_on_thermal`.
- Add `gpu_temp_checkpoint_grace_seconds` (number, default 30.0).
- Add `resume_from_checkpoint` (string, optional — path or `"auto"`).

### `src/schema/_examples.yaml`
- Add `gpu_temp_checkpoint_grace_seconds: 30.0`.
- Add `resume_from_checkpoint:` (null).

### `src/streamlit_gui/app.py`
- Add `gpu_temp_checkpoint_grace_seconds` + `resume_from_checkpoint` inputs.
- Update `switch_on_thermal` help text (CPU-continuation semantics).

### Tests
- `tests/test_gpu_temp_guard.py` + `tests/test_gpu_temp_guard_extended.py`:
  rewrite to test `GPUTempSupervisor` (mock `subprocess.Popen`, mock
  `read_temperature_c` sequences, assert kill+restart+resume + critical-abort
  + critical-CPU-continue branches).
- `tests/test_cli_gpu_temp_flags.py`: add `--resume-from-checkpoint`,
  keep `--switch-on-thermal`.
- New: `tests/test_supervisor_checkpoint_resume.py` — assert `auto` picks
  latest rolling checkpoint.
- `tests/test_yaml_examples.py`: ensure updated examples validate.

## Constraints honored
- Schema is source of truth: every new key added to `_training.yaml`.
- `additionalProperties: false` preserved at every nested object.
- No torch import at module top of `gpu_temp_guard.py` (CI runs CPU-only).
- `FRANKESTEIN_SUPERVISOR_CHILD=1` env var prevents supervisor nesting.
- SIGUSR1 requires direct child (no shell wrapper) — use
  `start_new_session=True` + `os.setsid`.