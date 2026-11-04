#!/usr/bin/env python3
"""Convert an experiment output folder into consolidated CSV files.

Usage::

    python experimentos/to_csv.py [N]

``N`` is the experiment folder name under ``experimentos/`` (default ``2``),
or an absolute path to the experiment folder.

The script reads (without modifying) the files in ``experimentos/<N>/`` and
writes to ``experimentos/<N>/csvs/``:

* ``inference.csv`` -- original inference rows plus one column per
  ``key=value`` pair found in ``notes`` and ``notes_unparsed`` for leftovers.
* ``param_metrics.csv`` -- one row per combo id from ``param_metrics.json``.
* ``results.json.csv`` -- one row per entry of ``results.json`` (the huge
  ``stdout``/``stderr`` command output fields are omitted).
* ``results.md.csv`` -- the markdown table of ``results.md``.
* ``training_metrics.csv`` -- every row of every CSV in ``training_metrics/``
  with a leading ``run`` column (source file name without ``.csv``).
* ``training_metrics_avg.csv`` -- one row per run with the mean of every
  numeric column.
* ``training_metrics_std.csv`` -- one row per run with the sample standard
  deviation (``n - 1``) of every numeric column.
* ``training_metrics_final.csv`` -- one row per run with its last row.

Only the Python standard library is used.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

#: Columns kept from ``results.json``; ``stdout`` and ``stderr`` are omitted.
RESULTS_JSON_COLUMNS = (
    "category",
    "combo_id",
    "status",
    "duration_sec",
    "returncode",
    "notes",
    "checkpoint_path",
    "deploy_dir",
    "metrics_path",
    "test_accuracy",
    "test_loss",
)

CSV_KWARGS = {"newline": "", "encoding": "utf-8"}


def resolve_experiment_dir(argument: str | None) -> Path:
    """Resolve the experiment folder from the CLI argument.

    Args:
        argument: Folder name under ``experimentos/`` or an absolute path.
            ``None`` selects ``experimentos/2``.

    Returns:
        Absolute path to the experiment folder.
    """
    if argument is None:
        return SCRIPT_DIR / "2"
    candidate = Path(argument).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.is_dir():
        return candidate.resolve()
    return (SCRIPT_DIR / candidate).resolve()


def parse_notes(notes: str) -> tuple[list[tuple[str, str]], str]:
    """Split a ``notes`` cell into ``key=value`` pairs and leftovers.

    Args:
        notes: Raw notes text, e.g. ``"mask_position=20 top5_hit=True"``.

    Returns:
        Tuple ``(pairs, unparsed)`` where ``pairs`` preserves order and
        ``unparsed`` joins the tokens that are not ``key=value``.
    """
    pairs: list[tuple[str, str]] = []
    leftover: list[str] = []
    for token in notes.split():
        key, separator, value = token.partition("=")
        if separator and key and value:
            pairs.append((key, value))
        else:
            leftover.append(token)
    return pairs, " ".join(leftover)


def convert_inference(exp_dir: Path, out_dir: Path) -> tuple[Path, int, list[str]]:
    """Convert ``inference.csv``, expanding the ``notes`` column.

    Args:
        exp_dir: Experiment folder containing the source file.
        out_dir: Destination folder for the converted file.

    Returns:
        Tuple ``(path, row_count, extra_columns)`` for the written file.
    """
    src = exp_dir / "inference.csv"
    dst = out_dir / "inference.csv"

    with src.open(**CSV_KWARGS) as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    notes_index = header.index("notes")

    extra_columns: list[str] = []
    seen: set[str] = set()
    parsed: list[tuple[list[str], dict[str, str], str]] = []
    for row in rows[1:]:
        if len(row) <= notes_index:
            row = row + [""] * (notes_index + 1 - len(row))
        pairs, unparsed = parse_notes(row[notes_index])
        values = dict(pairs)
        for key in values:
            if key not in seen:
                seen.add(key)
                extra_columns.append(key)
        parsed.append((row, values, unparsed))

    with dst.open("w", **CSV_KWARGS) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header + extra_columns + ["notes_unparsed"])
        for row, values, unparsed in parsed:
            writer.writerow(
                row + [values.get(key, "") for key in extra_columns] + [unparsed]
            )

    return dst, len(parsed), extra_columns


def convert_param_metrics(exp_dir: Path, out_dir: Path) -> tuple[Path, int, list[str]]:
    """Flatten ``param_metrics.json`` into one row per combo.

    Args:
        exp_dir: Experiment folder containing the source file.
        out_dir: Destination folder for the converted file.

    Returns:
        Tuple ``(path, row_count, metric_columns)`` for the written file.
    """
    src = exp_dir / "param_metrics.json"
    dst = out_dir / "param_metrics.csv"

    data: dict[str, dict[str, object]] = json.loads(src.read_text(encoding="utf-8"))
    metric_columns: list[str] = []
    seen: set[str] = set()
    for metrics in data.values():
        for key in metrics:
            if key not in seen:
                seen.add(key)
                metric_columns.append(key)

    with dst.open("w", **CSV_KWARGS) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["combo_id"] + metric_columns)
        for combo_id, metrics in data.items():
            writer.writerow(
                [combo_id] + [metrics.get(key, "") for key in metric_columns]
            )

    return dst, len(data), metric_columns


def convert_results_json(exp_dir: Path, out_dir: Path) -> tuple[Path, int]:
    """Convert the ``results.json`` list into a flat CSV.

    Args:
        exp_dir: Experiment folder containing the source file.
        out_dir: Destination folder for the converted file.

    Returns:
        Tuple ``(path, row_count)`` for the written file.
    """
    src = exp_dir / "results.json"
    dst = out_dir / "results.json.csv"

    entries: list[dict[str, object]] = json.loads(src.read_text(encoding="utf-8"))
    with dst.open("w", **CSV_KWARGS) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(RESULTS_JSON_COLUMNS)
        for entry in entries:
            writer.writerow(
                ["" if entry.get(column) is None else entry.get(column)
                 for column in RESULTS_JSON_COLUMNS]
            )

    return dst, len(entries)


def convert_results_md(exp_dir: Path, out_dir: Path) -> tuple[Path, int]:
    """Convert the markdown table of ``results.md`` into a CSV.

    Args:
        exp_dir: Experiment folder containing the source file.
        out_dir: Destination folder for the converted file.

    Returns:
        Tuple ``(path, row_count)`` for the written file.
    """
    src = exp_dir / "results.md"
    dst = out_dir / "results.md.csv"

    lines = [line.strip() for line in src.read_text(encoding="utf-8").splitlines()]
    table = [line for line in lines if line.startswith("|")]
    if len(table) < 2:
        raise ValueError(f"no markdown table found in {src}")

    header = [cell.strip() for cell in table[0].strip("|").split("|")]
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in table[2:]
        if not set(line) <= {"|", "-", ":", " "}
    ]

    with dst.open("w", **CSV_KWARGS) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            writer.writerow(row[: len(header)])

    return dst, len(rows)


def _format_number(value: float) -> str:
    """Format a statistic with full round-trip precision."""
    return repr(float(value))


def convert_training_metrics(exp_dir: Path, out_dir: Path) -> dict[str, tuple[Path, int]]:
    """Consolidate every CSV in ``training_metrics/``.

    Writes the concatenated file plus per-run aggregate files (mean, sample
    standard deviation and last row).

    Args:
        exp_dir: Experiment folder containing ``training_metrics/``.
        out_dir: Destination folder for the converted files.

    Returns:
        Mapping of output name to ``(path, row_count)``.
    """
    metrics_dir = exp_dir / "training_metrics"
    sources = sorted(metrics_dir.glob("*.csv"))
    if not sources:
        raise ValueError(f"no CSV files found in {metrics_dir}")

    combined = out_dir / "training_metrics.csv"
    average = out_dir / "training_metrics_avg.csv"
    deviation = out_dir / "training_metrics_std.csv"
    final = out_dir / "training_metrics_final.csv"

    header: list[str] | None = None
    column_count = 0
    # Per-run, per-column Welford statistics: count, mean, M2.
    stats_by_run: dict[str, list[list[float]]] = {}
    is_numeric: list[bool] = []
    last_rows: dict[str, list[str]] = {}

    with combined.open("w", **CSV_KWARGS) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for source in sources:
            run = source.stem
            with source.open(**CSV_KWARGS) as source_handle:
                reader = csv.reader(source_handle)
                source_header = next(reader)
                if header is None:
                    header = source_header
                    column_count = len(header)
                    writer.writerow(["run"] + header)
                    is_numeric = [True] * column_count
                elif source_header != header:
                    raise ValueError(
                        f"header mismatch in {source}: expected {header}, "
                        f"got {source_header}"
                    )

                run_stats = [[0.0, 0.0, 0.0] for _ in range(column_count)]
                last_row: list[str] = [""] * column_count
                for row in reader:
                    if not row:
                        continue
                    if len(row) < column_count:
                        row = row + [""] * (column_count - len(row))
                    elif len(row) > column_count:
                        row = row[:column_count]
                    writer.writerow([run] + row)
                    last_row = row
                    for index in range(column_count):
                        value = row[index]
                        if value == "":
                            continue
                        try:
                            number = float(value)
                        except ValueError:
                            is_numeric[index] = False
                            continue
                        column = run_stats[index]
                        column[0] += 1.0
                        delta = number - column[1]
                        column[1] += delta / column[0]
                        column[2] += delta * (number - column[1])
                last_rows[run] = last_row
                stats_by_run[run] = run_stats

    assert header is not None
    numeric_indices = [index for index, flag in enumerate(is_numeric) if flag]
    numeric_header = ["run"] + [header[index] for index in numeric_indices]

    def write_statistics(path: Path, kind: str) -> None:
        with path.open("w", **CSV_KWARGS) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(numeric_header)
            for run, _ in last_rows.items():
                output_row = [run]
                run_stats = stats_by_run[run]
                for index in numeric_indices:
                    count, mean, m2 = run_stats[index]
                    if count == 0:
                        value = ""
                    elif kind == "avg":
                        value = _format_number(mean)
                    else:
                        variance = m2 / (count - 1) if count > 1 else 0.0
                        value = _format_number(math.sqrt(max(variance, 0.0)))
                    output_row.append(value)
                writer.writerow(output_row)

    write_statistics(average, "avg")
    write_statistics(deviation, "std")

    with final.open("w", **CSV_KWARGS) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["run"] + header)
        for run, row in last_rows.items():
            writer.writerow([run] + row)

    return {
        "training_metrics.csv": (combined, len(last_rows)),
        "training_metrics_avg.csv": (average, len(last_rows)),
        "training_metrics_std.csv": (deviation, len(last_rows)),
        "training_metrics_final.csv": (final, len(last_rows)),
    }


def main(argv: list[str]) -> int:
    """Run every conversion and print a summary.

    Args:
        argv: Command line arguments (``[script, [experiment]]``).

    Returns:
        Process exit code.
    """
    exp_dir = resolve_experiment_dir(argv[1] if len(argv) > 1 else None)
    if not exp_dir.is_dir():
        print(f"error: experiment folder not found: {exp_dir}", file=sys.stderr)
        return 1

    out_dir = exp_dir / "csvs"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"experiment: {exp_dir}")
    print(f"output:     {out_dir}")

    conversions = [
        ("inference.csv", lambda: convert_inference(exp_dir, out_dir)),
        ("param_metrics.csv", lambda: convert_param_metrics(exp_dir, out_dir)),
        ("results.json.csv", lambda: convert_results_json(exp_dir, out_dir)),
        ("results.md.csv", lambda: convert_results_md(exp_dir, out_dir)),
    ]

    for name, convert in conversions:
        try:
            result = convert()
        except FileNotFoundError as error:
            print(f"skip {name}: {error}")
            continue
        path = result[0]
        details = f" ({result[1]} rows"
        if len(result) > 2:
            details += f", extra columns: {', '.join(result[2]) or 'none'}"
        details += ")"
        print(f"wrote {path.name}{details}")

    try:
        training = convert_training_metrics(exp_dir, out_dir)
    except (FileNotFoundError, ValueError) as error:
        print(f"skip training metrics: {error}")
    else:
        for name, (path, _rows) in training.items():
            line_count = sum(1 for _ in path.open(encoding="utf-8")) - 1
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"wrote {name} ({line_count} data rows, {size_mb:.1f} MiB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
