"""Utilities for Kneedle-based curve fitting of exclusive latency baselines."""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "baseline_evaluations" / "mudi" / "curvefit"
FIG_ROOT = OUTPUT_ROOT / "fig"


CLASS_CSV = Path(__file__).resolve().parents[1] / "workload_class.csv"

def _load_workload_classes() -> Dict[str, str]:
    if not CLASS_CSV.exists():
        return {}
    df = pd.read_csv(CLASS_CSV)
    if "workload" not in df.columns or "on_offline" not in df.columns:
        return {}
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        workload = str(row.workload).strip()
        if not workload:
            continue
        mapping[workload] = str(row.on_offline).lower()
    return mapping


WORKLOAD_CLASS = _load_workload_classes()

#colocated files key columns
#single workload files key columns
SINGLE_KEY_COLUMNS = ["Type", "freq"]


EXCLUSIVE_STEPS: Tuple[int, ...] = tuple(range(10, 100, 10))


def _is_inference(name: str) -> bool:
    lowered = name.lower()
    return "-inf" in lowered or "inference" in lowered


def _is_train(name: str) -> bool:
    lowered = name.lower()
    return "-train" in lowered or lowered.endswith("train")


def _is_cuda_sample(name: str) -> bool:
    return "cuda_samples" in name.lower()


def _determine_order(workloads: Sequence[str]) -> List[int]:
    n_workloads = len(workloads)
    if n_workloads <= 1:
        return list(range(n_workloads))
    if all(_is_inference(name) for name in workloads) or all(_is_train(name) for name in workloads):
        return list(range(n_workloads))
    cuda_indices = [idx for idx, name in enumerate(workloads) if _is_cuda_sample(name)]
    if cuda_indices:
        lead = cuda_indices[0]
        return [lead] + [idx for idx in range(n_workloads) if idx != lead]
    online_indices = [idx for idx, name in enumerate(workloads)
                      if WORKLOAD_CLASS.get(name, "offline").lower() == "online"]
    if online_indices:
        lead = online_indices[0]
        return [lead] + [idx for idx in range(n_workloads) if idx != lead]
    return list(range(n_workloads))


def compute_kneedle(delta: np.ndarray, latency: np.ndarray) -> Dict[str, float]:
    """Return Kneedle knee point and slopes for the provided series."""
    if delta.size == 0 or latency.size == 0:
        raise ValueError("delta and latency must contain at least one element")

    x_min = float(delta.min())
    x_max = float(delta.max())
    if math.isclose(x_max, x_min):
        x_norm = np.zeros_like(delta, dtype=float)
    else:
        x_norm = (delta - x_min) / (x_max - x_min)

    y_min = float(latency.min())
    y_max = float(latency.max())
    if math.isclose(y_max, y_min):
        y_norm = np.zeros_like(latency, dtype=float)
    else:
        y_norm = (latency - y_min) / (y_max - y_min)

    decreasing = latency[0] > latency[-1]
    if decreasing:
        y_curve = 1.0 - y_norm
    else:
        y_curve = y_norm

    difference = y_curve - x_norm
    knee_index = int(np.argmax(difference))
    delta0 = float(delta[knee_index])
    l0 = float(latency[knee_index])

    left_mask = delta <= delta0
    right_mask = delta > delta0

    k1 = _fit_slope(delta[left_mask], latency[left_mask], delta0, l0)
    k2 = _fit_slope(delta[right_mask], latency[right_mask], delta0, l0)

    return {
        "delta0": delta0,
        "latency0": l0,
        "slope_left": k1,
        "slope_right": k2,
        "knee_index": knee_index,
    }


def _fit_slope(delta: np.ndarray, latency: np.ndarray, delta0: float, l0: float) -> float:
    """Least-squares slope for points anchored at the knee."""
    if delta.size == 0:
        return float("nan")

    x = (delta - delta0).reshape(-1, 1)
    y = (latency - l0).reshape(-1, 1)

    if np.allclose(x, 0.0):
        return 0.0

    slope, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(np.squeeze(slope))


def piecewise_predict(delta: Iterable[float], params: Dict[str, float]) -> np.ndarray:
    """Return predicted latency for each delta using fitted parameters."""
    delta_arr = np.asarray(list(delta), dtype=float)
    out = np.empty_like(delta_arr)
    for idx, value in enumerate(delta_arr):
        if value <= params["delta0"]:
            slope = params["slope_left"]
        else:
            slope = params["slope_right"]
        if math.isnan(slope):
            slope = 0.0
        out[idx] = slope * (value - params["delta0"]) + params["latency0"]
    return out


def sanitize_filename(name: str) -> str:
    """Return a filesystem-friendly variant of the workload name."""
    safe_chars = [c if c.isalnum() or c in ("-", "_") else "_" for c in name]
    return "".join(safe_chars)


def resolve_output_dir(output_dir: Path) -> Path:
    """Route outputs to the fig directory unless an absolute path is provided."""
    if output_dir.is_absolute():
        target = output_dir
    else:
        target = FIG_ROOT / output_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def _colocated_key_columns(n_comb: int) -> List[str]:
    """Return the required key columns for a colocated workload CSV."""
    if n_comb < 1:
        raise ValueError("n_comb must be at least 1")
    if n_comb == 1:
        return list(SINGLE_KEY_COLUMNS)

    workloads = [f"workload{i+1}" for i in range(n_comb)]
    freqs = [f"freq{i+1}" for i in range(n_comb)]
    return workloads + freqs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_csv",
        type=Path,
        nargs="+",
        default=[Path("data/baseline_metrics/0206_FREQ1530_baseline_metrics.csv")],
        help="One or more baseline metrics CSVs with Exclusive columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("kneedle"),
        help="Sub-directory (relative to artifacts/baseline_evaluations/mudi/curvefit/fig) used when --plot is enabled.",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="0206_FREQ1530_kneedle_params.csv",
        help="Filename for the summary CSV (placed under the summary directory).",
    )
    parser.add_argument(
        "--n_comb",
        type=int,
        default=1,
        help="Number of colocated workloads (1 for single baseline input).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Emit figure outputs in addition to the summary CSV.",
    )
    return parser


def load_exclusive_series(
    row: pd.Series, steps: Sequence[int] = EXCLUSIVE_STEPS
) -> Tuple[np.ndarray, np.ndarray]:
    values: List[float] = []
    deltas: List[float] = []
    for step in steps:
        col = f"Exclusive{step}"
        if col not in row.index:
            continue
        value = row[col]
        if pd.notna(value):
            values.append(float(value))
            deltas.append(float(step))
    return np.asarray(deltas, dtype=float), np.asarray(values, dtype=float)


def process(
    input_csvs: Sequence[Path],
    output_dir: Path,
    summary_name: str,
    n_comb: int,
    plot: bool,
) -> Path:
    resolved_output_dir = resolve_output_dir(output_dir) if plot else None
    df = _load_inputs(input_csvs, n_comb)

    if n_comb == 1:
        return _process_single(df, resolved_output_dir, summary_name, plot)

    return _process_colocated(df, resolved_output_dir, summary_name, n_comb, plot)


def _load_inputs(paths: Sequence[Path], n_comb: int) -> pd.DataFrame:
    """Read and merge one or more CSV inputs, dropping duplicate workload pairs."""
    frames = []
    expected_columns = None
    key_columns = _colocated_key_columns(n_comb)
    for csv_path in paths:
        df = pd.read_csv(csv_path)
        missing = [col for col in key_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")
        if expected_columns is None:
            expected_columns = list(df.columns)
        elif list(df.columns) != expected_columns:
            raise ValueError(
                f"Column mismatch for {csv_path}. Expected {expected_columns}, got {list(df.columns)}"
            )
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=key_columns, keep="first")
    merged = merged.sort_values(key_columns).reset_index(drop=True)
    return merged


def _process_single(
    df: pd.DataFrame, output_dir: Optional[Path], summary_name: str, plot: bool
) -> Path:
    summary_rows: List[Dict[str, float]] = []
    exclusive_steps = EXCLUSIVE_STEPS
    if exclusive_steps[-1] != 100:
        exclusive_steps = exclusive_steps + (100,)
    dense_delta = np.linspace(exclusive_steps[0], exclusive_steps[-1], 161)

    for _, row in df.iterrows():
        workload = str(row.get("Type", f"row_{_}"))
        deltas, latency = load_exclusive_series(row, exclusive_steps)
        if deltas.size == 0:
            continue

        params = compute_kneedle(deltas, latency)
        predictions = piecewise_predict(dense_delta, params)

        if plot and output_dir is not None:
            plot_path = output_dir / f"{sanitize_filename(workload)}.png"
            _plot_workload(workload, deltas, latency, dense_delta, predictions, params, plot_path)

        summary_rows.append(
            {
                "Type": workload,
                "freq": row.get("freq", float("nan")),
                "delta0": params["delta0"],
                "latency0": params["latency0"],
                "slope_left": params["slope_left"],
                "slope_right": params["slope_right"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_root = FIG_ROOT.parent / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / summary_name
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def _process_colocated(
    df: pd.DataFrame,
    output_dir: Optional[Path],
    summary_name: str,
    n_comb: int,
    plot: bool,
) -> Path:
    summary_rows: List[Dict[str, float]] = []

    share_columns = [col for col in df.columns if col.startswith("(") and "w1_" in col]
    share_pattern = re.compile(r"w(\d+)_([0-9]+)")

    for _, row in df.iterrows():
        workloads = [str(row.get(f"workload{i+1}", f"workload_{i+1}")).strip() for i in range(n_comb)]
        freqs = [row.get(f"freq{i+1}") for i in range(n_comb)]
        if any(pd.isna(freq) for freq in freqs):
            continue

        per_workload_shares: List[List[float]] = [[] for _ in range(n_comb)]
        per_workload_values: List[List[float]] = [[] for _ in range(n_comb)]

        for col in share_columns:
            matches = share_pattern.findall(col)
            if len(matches) < n_comb:
                continue

            try:
                values = row[col]
            except KeyError:
                continue

            if pd.isna(values):
                continue

            if isinstance(values, str):
                try:
                    parsed = ast.literal_eval(values)
                except (SyntaxError, ValueError):
                    continue
            else:
                parsed = values

            if not isinstance(parsed, (list, tuple)) or len(parsed) < n_comb:
                continue

            for idx, (workload_idx_str, share_str) in enumerate(matches[:n_comb]):
                workload_idx = int(workload_idx_str) - 1
                if workload_idx < 0 or workload_idx >= n_comb:
                    continue

                share_value = float(share_str)
                if share_value >= 100:
                    continue
                metric_value = parsed[idx]
                if metric_value is None or (isinstance(metric_value, float) and math.isnan(metric_value)):
                    continue

                per_workload_shares[workload_idx].append(share_value)
                per_workload_values[workload_idx].append(float(metric_value))

        order = _determine_order(workloads)
        if order != list(range(n_comb)):
            workloads = [workloads[idx] for idx in order]
            freqs = [freqs[idx] for idx in order]
            per_workload_shares = [per_workload_shares[idx] for idx in order]
            per_workload_values = [per_workload_values[idx] for idx in order]

        combo_slug = "__".join(sanitize_filename(name) for name in workloads)
        combo_label = " | ".join(workloads)

        summary_row: Dict[str, float] = {}
        for idx, workload in enumerate(workloads):
            shares = np.asarray(per_workload_shares[idx], dtype=float)
            values = np.asarray(per_workload_values[idx], dtype=float)
            if shares.size == 0 or values.size == 0:
                continue

            order = np.argsort(shares)
            shares = shares[order]
            values = values[order]

            params = compute_kneedle(shares, values)
            dense_delta = np.linspace(shares.min(), shares.max(), 161)
            predictions = piecewise_predict(dense_delta, params)

            suffix = f"w{idx+1}"
            summary_row[f"workload{idx+1}"] = workload
            summary_row[f"{suffix}_freq"] = freqs[idx]
            summary_row[f"{suffix}_cutoff"] = params["delta0"]
            summary_row[f"{suffix}_value0"] = params["latency0"]
            summary_row[f"{suffix}_slope_left"] = params["slope_left"]
            summary_row[f"{suffix}_slope_right"] = params["slope_right"]

            if plot and output_dir is not None:
                plot_title = f"{workload} | combo {combo_label}"
                plot_name = f"{combo_slug}__{suffix}.png"
                plot_path = output_dir / plot_name
                _plot_workload(
                    plot_title,
                    shares,
                    values,
                    dense_delta,
                    predictions,
                    params,
                    plot_path,
                )

        complete = all(summary_row.get(f"w{i+1}_cutoff") is not None for i in range(n_comb))
        if summary_row and complete:
            summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_root = FIG_ROOT.parent / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / summary_name
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def _plot_workload(
    workload: str,
    delta: np.ndarray,
    latency: np.ndarray,
    dense_delta: np.ndarray,
    predictions: np.ndarray,
    params: Dict[str, float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(delta, latency, label="Measured latency", zorder=3)
    plt.plot(dense_delta, predictions, label="Piecewise fit", zorder=2)
    plt.axvline(params["delta0"], linestyle="--", label=f"Knee Δ0={params['delta0']:.0f}%")
    plt.axhline(params["latency0"], linestyle=":")
    plt.xlabel("GPU allocation Δ (%)")
    plt.ylabel("Metric value")
    plt.title(workload)
    plt.legend()
    plt.grid(True, which="major", linestyle="--", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary_path = process(args.input_csv, args.output_dir, args.summary_name, args.n_comb, args.plot)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
