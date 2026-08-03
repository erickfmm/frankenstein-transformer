# Training Safety and Stability Specification

> Cross-references: [Schema Reference](schema-reference.md) · [Optimizers](optimizers.md) · [CLI Reference](cli-reference.md) · [Architecture](architecture.md)

## Gradient Accumulation

Gradients are accumulated over `K = gradient_accumulation_steps` micro-batches before each optimizer step:

```
g_acc = (1/K) Σ_{i=1}^{K} g_i
```

Effective batch size = `batch_size × gradient_accumulation_steps`. Learning rate should scale linearly with effective batch size.

## Global Norm Clipping

After accumulation, gradients are clipped by global norm:

```
g_clip = g_acc · min(1, τ / (‖g_acc‖₂ + ε))
```

where `τ = grad_clip_max_norm`. This prevents individual gradient spikes from destabilizing training.

## Post-Clip Explosion Guard

After clipping, an additional safety check uses `inf_post_clip_threshold`:

```
if max(|g_clip|) > inf_post_clip_threshold:
    → trigger NaN/Inf retry logic
```

This catches cases where clipping reduces the norm but individual elements remain pathologically large.

## NaN/Inf Retry Logic

```
Initialize retry counter r = 0
For each optimizer step:
    Accumulate gradients for K micro-batches
    Apply global norm clipping with τ
    If post-clip gradient exceeds threshold OR NaN/Inf detected:
        If r < max_nan_retries:
            Restore safe state / skip step
            r = r + 1
            continue
        Else:
            Stop training with failure state
    Run optimizer step
    Update scheduler
    Checkpoint if step mod checkpoint_every_n_steps == 0
    Update best checkpoints
    Emit CSV + telemetry
```

NaN/Inf checks occur every `nan_check_interval` steps.

## Checkpointing Policy

### Rolling Checkpoints

| Parameter | Description |
|---|---|
| `checkpoint_every_n_steps` | Save a checkpoint every N optimizer steps |
| `max_rolling_checkpoints` | Keep at most this many recent checkpoints (oldest pruned) |

### Best Checkpoints

| Parameter | Description |
|---|---|
| `num_best_checkpoints` | Track and retain the N checkpoints with lowest validation loss |

## Telemetry

### CSV Logging

| Parameter | Description |
|---|---|
| `csv_log_path` | File path for step-level CSV output |
| `csv_rotate_on_schema_change` | If true, create a new CSV file when logging schema changes |

CSV columns include: step, loss, learning_rate, gradient_norm, GPU temperature (if NVML enabled), and per-block gradient norms (if enabled).

### GPU Metrics

| Parameter | Description |
|---|---|
| `gpu_metrics_backend` | `nvml` (NVIDIA Management Library) or `none` |
| `nvml_device_index` | GPU device index for NVML queries |

### Block Gradient Norms

| Parameter | Description |
|---|---|
| `enable_block_grad_norms` | Include per-block gradient norm in telemetry |
| `gradient_log_interval` | Log gradient statistics every N steps |
| `telemetry_log_interval` | Log heavy telemetry (GPU metrics, block norms) every N optimizer steps |

## GPU Thermal Guard

Protects hardware by monitoring GPU temperature and taking action at configurable thresholds:

| Parameter | Description |
|---|---|
| `gpu_temp_guard` (CLI flag) | Enable/disable thermal monitoring |
| `gpu_temp_pause_threshold_c` | Temperature (°C) at which training **pauses** |
| `gpu_temp_resume_threshold_c` | Temperature (°C) at which training **resumes** |
| `gpu_temp_critical_threshold_c` | Temperature (°C) at which training **aborts** |
| `gpu_temp_poll_interval_seconds` | Seconds between temperature checks |
| `switch_on_thermal` (CLI flag) | Enable automatic device switching on thermal events |

### Thermal State Machine

```
Normal → [temp > pause_threshold] → Paused
Paused → [temp < resume_threshold] → Normal
Any state → [temp > critical_threshold] → Aborted
```

When `switch_on_thermal` is enabled, the system may automatically switch to a cooler device (e.g., CUDA → CPU) on thermal events.

## GaLore Controls

Gradient Low-Rank Projection (GaLore) reduces optimizer memory by projecting 2D gradients into a low-rank subspace:

| Parameter | Description |
|---|---|
| `use_galore` | Enable GaLore strategy |
| `galore_rank` | Low-rank projection dimension `r` |
| `galore_update_interval` | How often to refresh the SVD projection |
| `galore_scale` | Gradient scaling factor in projected space |
| `galore_max_dim` | Maximum tensor dimension eligible for GaLore projection |

GaLore works synergistically with memory-efficient optimizers (Adafactor, APOLLO) and is most beneficial when VRAM is dominated by optimizer state.

## Scheduler Types

| Type | Behavior | Key Parameters |
|---|---|---|
| `cosine` | Cosine decay from initial LR to ~0 | `scheduler_total_steps`, `scheduler_warmup_ratio` |
| `constant` | Fixed LR throughout training | None |
| `linear_warmup_then_constant` | Linear warmup then flat | `scheduler_total_steps`, `scheduler_warmup_ratio` |

The scheduler is updated after each optimizer step. Warmup steps = `scheduler_warmup_ratio × scheduler_total_steps`.

## Training Step Algorithm (Full)

```
Require: Batch stream, config C
Initialize retry counter r = 0
For each optimizer step:
    Accumulate gradients for K = C.gradient_accumulation_steps micro-batches
    Apply global norm clipping with τ = C.grad_clip_max_norm
    If post-clip gradient exceeds C.inf_post_clip_threshold OR NaN/Inf detected:
        If r < C.max_nan_retries:
            Restore safe state / skip step; r = r + 1
            continue
        Else:
            Stop training with failure state
    Run optimizer step selected by optimizer_class
    Update scheduler (cosine, constant, or linear_warmup_then_constant)
    If step mod checkpoint_every_n_steps == 0:
        Save rolling checkpoint and prune to max_rolling_checkpoints
    Update best checkpoints up to num_best_checkpoints
    Emit CSV + telemetry following gradient_log_interval and telemetry_log_interval
```

## Stability Best Practices

1. **Start with `grad_clip_max_norm = 1.0`** — the standard value for transformer training.
2. **Set `max_nan_retries = 3`** — allows transient instability without aborting.
3. **Use `inf_post_clip_threshold`** as a secondary safety net (e.g., 10.0).
4. **Enable `use_amp`** on modern GPUs for ~50% memory savings with automatic gradient scaling.
5. **Monitor `csv_log_path`** output to detect training degradation early.
6. **Set `checkpoint_every_n_steps`** to a value that balances disk I/O with recovery granularity (e.g., 1000).
7. **Use `gpu_temp_guard`** in production environments to prevent hardware damage.
