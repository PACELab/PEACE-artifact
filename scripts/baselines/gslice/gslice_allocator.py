#!/usr/bin/env python3
"""GSlice allocator sweep over throughput grids.

This script evaluates the single-tick GSlice-style allocator for every
workload group present in a stage-2 throughput grid CSV. It mirrors the
workflow used by the Kneedle utilities: caps are sourced from the
``delta0`` column of a kneepoint summary CSV, arrivals come from the
``Exclusive100`` column of a baseline metrics CSV, results are written to
the ``summary/`` sub-directory, and diagnostic figures are emitted under
``fig/``.

The allocator starts from a configurable split (default 50/50 for comb2, 30/30/40 for comb3), computes
throughput residuals relative to the arrivals, proposes new GPU
percentages subject to kneepoint caps, applies a fairness step to enforce
the 100% budget, and finally selects the closest available split from the
grid (no interpolation is performed). The chosen split, associated
throughputs, and residuals are recorded in the summary CSV and the
associated figure highlights before/after throughput, residuals, change
factors, and the final percentage decision.
Grid columns are bucketised (10-point for ``n_comb == 2``, 5-point otherwise).
The allocator rounds percentages while
searching for matching columns; if a fair split is absent from the grid
the loop terminates early, the requested percentages are preserved in the
summary, and throughput-oriented fields remain blank.
Final splits therefore always respect the configured percentage granularity.
"""

from __future__ import annotations

import argparse
import ast
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, cast

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "baseline_evaluations" / "gslice"
SUMMARY_ROOT = OUTPUT_ROOT / "summary"
FIG_ROOT = OUTPUT_ROOT / "fig"

PERCENTAGE_PATTERN = re.compile(r"w(\d+)_(\d+(?:\.\d+)?)")
ROUNDING_INCREMENT_COMB2 = 10.0
ROUNDING_INCREMENT_OTHER = 5.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SplitMetrics:
    """Container for split metadata extracted from the throughput grid."""

    column: str
    percentages: Tuple[float, ...]
    throughputs: Tuple[float, ...]


@dataclass
class AllocationResult:
    """Summary of a single allocator run for a workload colocated group."""

    workloads: Tuple[str, ...]
    freqs: Tuple[float, ...]
    start_split: Tuple[float, ...]
    proposal_split: Tuple[float, ...]
    fair_split: Tuple[float, ...]
    selected_split: Tuple[float, ...]
    start_throughput: Tuple[float, ...]
    selected_throughput: Tuple[float, ...] | None
    arrival: Tuple[float, ...]
    residual_start: Tuple[float, ...]
    residual_selected: Tuple[float, ...] | None
    change_factors: Tuple[float, ...]
    caps: Tuple[float, ...]
    metrics_source: str | None
    iterations: int


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """Return a filesystem-friendly variant of ``name``."""

    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def parse_tuple(value: str, expected_len: int) -> Tuple[float, ...]:
    """Parse a tuple-like string into a length-``expected_len`` tuple of floats."""

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unable to parse throughput tuple: {value!r}") from exc
    if not isinstance(parsed, (list, tuple)) or len(parsed) != expected_len:
        raise ValueError(
            f"Throughput entry must be length-{expected_len} tuple, got: {value!r}"
        )
    return tuple(float(part) for part in parsed)


def load_caps(caps_csv: Path) -> Dict[str, float]:
    """Return ``workload -> cap`` mapping from a kneepoint summary CSV."""

    df = pd.read_csv(caps_csv)
    if "Type" not in df.columns or "delta0" not in df.columns:
        raise ValueError(
            f"Caps CSV {caps_csv} must contain 'Type' and 'delta0' columns"
        )
    caps: Dict[str, float] = {}
    for _, row in df.iterrows():
        workload = str(row["Type"]).strip()
        if not workload:
            continue
        try:
            caps[workload] = float(row["delta0"])
        except (TypeError, ValueError):
            logging.warning("Skipping cap for %s due to non-numeric delta0", workload)
    return caps


def load_arrivals(arrival_csv: Path) -> Dict[str, float]:
    """Return ``workload -> arrival`` mapping from Exclusive100 columns."""
    #use exclusive100 as arrival rate
    arrival_percentage = 100
    df = pd.read_csv(arrival_csv)
    if "Type" not in df.columns or f"Exclusive{arrival_percentage}" not in df.columns:
        raise ValueError(
            f"Arrival CSV {arrival_csv} must contain 'Type' and 'Exclusive{arrival_percentage}' columns"
        )
    arrivals: Dict[str, float] = {}
    for _, row in df.iterrows():
        workload = str(row["Type"]).strip()
        if not workload:
            continue
        try:
            arrivals[workload] = float(row[f"Exclusive{arrival_percentage}"])
        except (TypeError, ValueError):
            logging.warning(
                "Skipping arrival for %s due to non-numeric Exclusive%s", workload, arrival_percentage
            )
    return arrivals


def extract_splits(row: pd.Series, num_workloads: int) -> List[SplitMetrics]:
    """Extract split metrics for an ``n``-workload throughput row."""

    splits: List[SplitMetrics] = []
    for column in row.index:
        if not isinstance(column, str):
            continue
        matches = PERCENTAGE_PATTERN.findall(column)
        if len(matches) != num_workloads:
            continue
        percentages: List[float | None] = [None] * num_workloads
        valid = True
        for index_str, pct_str in matches:
            workload_idx = int(index_str) - 1
            if workload_idx < 0 or workload_idx >= num_workloads:
                valid = False
                break
            if percentages[workload_idx] is not None:
                valid = False
                break
            percentages[workload_idx] = float(pct_str)
        if not valid or any(value is None for value in percentages):
            continue
        if not math.isclose(sum(cast(float, pct) for pct in percentages), 100.0, rel_tol=1e-6, abs_tol=1e-6):
            continue
        if all(math.isclose(cast(float, pct), 100.0, abs_tol=1e-6) for pct in percentages):
            continue
        try:
            throughputs = parse_tuple(str(row[column]), num_workloads)
        except ValueError as exc:
            raise ValueError(f"Failed to parse throughput tuple for column {column}") from exc
        splits.append(
            SplitMetrics(
                column=column,
                percentages=tuple(cast(float, pct) for pct in percentages),
                throughputs=throughputs,
            )
        )
    if not splits:
        raise ValueError("No valid GPU percentage columns found in throughput row")
    return splits


def make_column_name(percentages: Iterable[float]) -> str:
    """Return the canonical column name for a split."""

    rounded = [int(round(value)) for value in percentages]
    pieces = [f"w{idx + 1}_{rounded[idx]}" for idx in range(len(rounded))]
    return f"({', '.join(pieces)})"


def rounded_split(percentages: Iterable[float]) -> Tuple[int, ...]:
    """Return the rounded integer split tuple used to index grid columns."""

    return tuple(int(round(value)) for value in percentages)


def round_split_to_increment(
    percentages: Iterable[float],
    caps: Iterable[float],
    increment: float = ROUNDING_INCREMENT_OTHER,
) -> Tuple[float, ...]:
    """Return percentages snapped to ``increment`` while staying within caps."""

    values = list(percentages)
    caps_list = list(caps)
    if increment <= 0:
        raise ValueError("Increment must be positive for rounding.")
    if len(values) != len(caps_list):
        raise ValueError("Percentages and caps must share the same length.")

    units_target = int(round(sum(values) / increment))
    scaled = [value / increment for value in values]
    base_units = [int(math.floor(val)) for val in scaled]
    remainders = [scaled[idx] - base_units[idx] for idx in range(len(values))]
    cap_units = [
        int(math.floor(caps_list[idx] / increment + 1e-9)) if caps_list[idx] >= 0 else 0
        for idx in range(len(values))
    ]

    for idx in range(len(base_units)):
        base_units[idx] = min(base_units[idx], cap_units[idx])

    units_total = sum(base_units)
    units_to_distribute = units_target - units_total

    if units_to_distribute > 0:
        order = sorted(range(len(values)), key=lambda idx: remainders[idx], reverse=True)
        for idx in order:
            if units_to_distribute <= 0:
                break
            room = cap_units[idx] - base_units[idx]
            if room <= 0:
                continue
            take = min(room, units_to_distribute)
            base_units[idx] += take
            units_to_distribute -= take
    elif units_to_distribute < 0:
        order = sorted(range(len(values)), key=lambda idx: remainders[idx])
        for idx in order:
            if units_to_distribute >= 0:
                break
            removable = base_units[idx]
            if removable <= 0:
                continue
            take = min(removable, -units_to_distribute)
            base_units[idx] -= take
            units_to_distribute += take

    if units_to_distribute != 0:
        logging.debug(
            "Rounding residual of %d units after primary distribution; applying fallback.",
            units_to_distribute,
        )
        if units_to_distribute > 0:
            for _ in range(abs(units_to_distribute)):
                updated = False
                for idx in range(len(values)):
                    if base_units[idx] < cap_units[idx]:
                        base_units[idx] += 1
                        units_to_distribute -= 1
                        updated = True
                        break
                if not updated:
                    break
        else:
            for _ in range(abs(units_to_distribute)):
                updated = False
                for idx in range(len(values)):
                    if base_units[idx] > 0:
                        base_units[idx] -= 1
                        units_to_distribute += 1
                        updated = True
                        break
                if not updated:
                    break
    if units_to_distribute != 0:
        logging.warning(
            "Unable to fully round allocations to %.1f%% increments under caps; residual units=%d",
            increment,
            units_to_distribute,
        )

    rounded = [unit * increment for unit in base_units]
    return tuple(rounded)


def select_best_split_from_grid(
    target_split: Tuple[float, ...],
    splits: Iterable[SplitMetrics],
    caps: Tuple[float, ...],
    tolerance: float = 1e-6,
) -> Tuple[SplitMetrics, bool]:
    """Return the closest grid split, preferring cap-compliant entries."""

    cap_candidates: List[Tuple[float, Tuple[float, ...], SplitMetrics]] = []
    overflow_candidates: List[Tuple[float, Tuple[float, ...], SplitMetrics]] = []

    for split in splits:
        diff = tuple(abs(split.percentages[idx] - target_split[idx]) for idx in range(len(target_split)))
        dist = sum(diff)
        if all(split.percentages[idx] <= caps[idx] + tolerance for idx in range(len(target_split))):
            cap_candidates.append((dist, diff, split))
        else:
            overflow_candidates.append((dist, diff, split))

    def pick_best(candidates: List[Tuple[float, Tuple[float, ...], SplitMetrics]]) -> SplitMetrics:
        candidates.sort(key=lambda item: (item[0], item[1], item[2].percentages))
        return candidates[0][2]

    if cap_candidates:
        return pick_best(cap_candidates), True

    if overflow_candidates:
        chosen = pick_best(overflow_candidates)
        logging.warning(
            "No cap-compliant split found; selecting closest overflow split %s",
            make_column_name(chosen.percentages),
        )
        return chosen, False

    raise ValueError("No candidate splits available for selection")


def propose_gpu(current_pct: float, avg_throughput: float, arrival_target: float, cap: float) -> Tuple[float, float, float]:
    """Throughput-only controller proposal.

    Returns ``(proposal, residual, change_factor)``.
    """

    residual = avg_throughput - arrival_target
    if avg_throughput <= 0:
        change_factor = 0.0
    else:
        change_factor = abs(residual) / avg_throughput

    if residual < 0:  # under target => increase
        proposal = current_pct * (1.0 + change_factor)
    else:  # over target => decrease
        proposal = current_pct / (1.0 + change_factor) if change_factor > 0 else current_pct

    proposal = min(cap, max(0.0, proposal))
    return proposal, residual, change_factor


def allocate_to_100(proposals: Dict[str, float], caps: Dict[str, float]) -> Dict[str, float]:
    """Adjust proposed allocations to sum to 100% in a cap-aware fashion."""

    total = sum(proposals.values())
    allocations = proposals.copy()
    if math.isclose(total, 100.0):
        return allocations

    if total < 100.0:
        remaining = 100.0 - total
        weights = {name: max(value, 1e-6) for name, value in allocations.items()}
        for _ in range(16):  # iterative fill
            if remaining <= 1e-9:
                break
            weight_sum = sum(weights.values())
            if weight_sum <= 1e-9:
                break
            for name in list(allocations.keys()):
                if remaining <= 1e-9:
                    break
                increment = remaining * (weights[name] / weight_sum)
                room = max(0.0, caps[name] - allocations[name])
                take = min(increment, room)
                allocations[name] += take
                remaining -= take
                if allocations[name] >= caps[name] - 1e-9:
                    weights[name] = 0.0
        return allocations

    # total > 100% => shrink proportionally while respecting minimum of zero
    scale = 100.0 / total
    for name in allocations:
        allocations[name] = max(0.0, allocations[name] * scale)
    return allocations


def run_allocator_for_group(
    row: pd.Series,
    splits: List[SplitMetrics],
    arrival_map: Dict[str, float],
    caps_map: Dict[str, float],
    start_split: Tuple[float, ...],
    num_workloads: int,
) -> AllocationResult:
    """Iteratively apply the allocator until splits converge or data runs out."""

    workloads = tuple(str(row[f"workload{idx + 1}"]).strip() for idx in range(num_workloads))
    if any(not name for name in workloads):
        raise ValueError("Throughput row missing workload identifiers")

    freqs = tuple(float(row.get(f"freq{idx + 1}", 0.0)) for idx in range(num_workloads))

    split_lookup = {
        rounded_split(split.percentages): split
        for split in splits
    }

    start_key = rounded_split(start_split)
    if start_key not in split_lookup:
        raise ValueError(
            f"Starting split {make_column_name(start_split)} not available for workloads {workloads}"
        )

    arrivals: List[float] = []
    caps: List[float] = []
    missing_arrivals: List[str] = []
    for name in workloads:
        arrival_value = arrival_map.get(name)
        if arrival_value is None:
            missing_arrivals.append(name)
        else:
            arrivals.append(arrival_value)
        caps.append(caps_map.get(name, 100.0))
    if missing_arrivals:
        raise ValueError(f"Arrival rates missing for workloads: {', '.join(missing_arrivals)}")

    rounding_increment = (
        ROUNDING_INCREMENT_COMB2 if num_workloads == 2 else ROUNDING_INCREMENT_OTHER
    )

    current_split = tuple(start_split)
    previous_selected: Tuple[int, ...] | None = None
    iterations = 0
    max_iterations = 20
    converged = False

    initial_throughput: Tuple[float, ...] | None = None
    initial_residual: Tuple[float, ...] | None = None

    last_result: Dict[str, object] | None = None

    while iterations < max_iterations:
        current_key = rounded_split(current_split)
        if current_key not in split_lookup:
            raise ValueError(
                f"Metrics for split {make_column_name(current_split)} not available for workloads {workloads}"
            )

        metrics = split_lookup[current_key]
        throughput_current = metrics.throughputs

        proposals: List[float] = []
        residuals: List[float] = []
        change_factors_list: List[float] = []

        for idx in range(num_workloads):
            proposal, residual, change_factor = propose_gpu(
                current_pct=current_split[idx],
                avg_throughput=throughput_current[idx],
                arrival_target=arrivals[idx],
                cap=caps[idx],
            )
            proposals.append(proposal)
            residuals.append(residual)
            change_factors_list.append(change_factor)

        if iterations == 0:
            initial_throughput = throughput_current
            initial_residual = tuple(residuals)

        label_map = {f"w{idx + 1}": proposals[idx] for idx in range(num_workloads)}
        caps_map_weighted = {f"w{idx + 1}": caps[idx] for idx in range(num_workloads)}
        fair_alloc_map = allocate_to_100(label_map, caps_map_weighted)
        fair_split_raw = tuple(fair_alloc_map[f"w{idx + 1}"] for idx in range(num_workloads))
        fair_split = round_split_to_increment(
            fair_split_raw,
            caps,
            rounding_increment,
        )

        try:
            selected_metrics, cap_compliant = select_best_split_from_grid(
                target_split=fair_split,
                splits=splits,
                caps=tuple(caps),
            )
            candidate_split = selected_metrics.percentages
            candidate_matches_fair = all(
                math.isclose(candidate_split[idx], fair_split[idx], abs_tol=1e-6)
                for idx in range(num_workloads)
            )
            if cap_compliant and candidate_matches_fair:
                selected_split_values = candidate_split
                selected_throughput = selected_metrics.throughputs
                residual_selected = tuple(
                    selected_throughput[idx] - arrivals[idx] for idx in range(num_workloads)
                )
                metrics_source = selected_metrics.column
                fallback_used = False
            elif not cap_compliant:
                selected_split_values = candidate_split
                selected_throughput = selected_metrics.throughputs
                residual_selected = tuple(
                    selected_throughput[idx] - arrivals[idx] for idx in range(num_workloads)
                )
                metrics_source = selected_metrics.column
                fallback_used = False
            else:
                selected_split_values = fair_split
                selected_throughput = None
                residual_selected = None
                metrics_source = None
                fallback_used = True
        except ValueError:
            selected_split_values = fair_split
            selected_throughput = None
            residual_selected = None
            metrics_source = None
            fallback_used = True

        last_result = {
            "proposal_split": tuple(proposals),
            "fair_split": fair_split,
            "selected_split": tuple(selected_split_values),
            "selected_throughput": selected_throughput,
            "residual_selected": residual_selected,
            "change_factors": tuple(change_factors_list),
            "metrics_source": metrics_source,
            "fallback_used": fallback_used,
        }

        iterations += 1

        if fallback_used:
            converged = True
            break

        current_selected_key = rounded_split(selected_split_values)
        if previous_selected is not None and current_selected_key == previous_selected:
            converged = True
            break

        previous_selected = current_selected_key
        current_split = selected_split_values

    if not last_result:
        raise RuntimeError(f"Allocator failed to evaluate workloads {workloads}")

    if not converged and iterations >= max_iterations:
        logging.warning(
            "Reached maximum iterations (%d) before convergence for workloads %s",
            max_iterations,
            workloads,
        )

    assert initial_throughput is not None and initial_residual is not None

    return AllocationResult(
        workloads=workloads,
        freqs=freqs,
        start_split=tuple(start_split),
        proposal_split=cast(Tuple[float, ...], last_result["proposal_split"]),
        fair_split=cast(Tuple[float, ...], last_result["fair_split"]),
        selected_split=cast(Tuple[float, ...], last_result["selected_split"]),
        start_throughput=initial_throughput,
        selected_throughput=cast(Tuple[float, ...] | None, last_result["selected_throughput"]),
        arrival=tuple(arrivals),
        residual_start=initial_residual,
        residual_selected=cast(Tuple[float, ...] | None, last_result["residual_selected"]),
        change_factors=cast(Tuple[float, ...], last_result["change_factors"]),
        caps=tuple(caps),
        metrics_source=cast(str | None, last_result["metrics_source"]),
        iterations=iterations,
    )


def render_figure(result: AllocationResult, output_dir: Path) -> None:
    """Produce a diagnostic figure summarising allocator behaviour."""

    output_dir.mkdir(parents=True, exist_ok=True)

    num_workloads = len(result.workloads)
    workloads_label = "\nvs\n".join(result.workloads)
    before = np.array(result.start_throughput, dtype=float)
    if result.selected_throughput is None:
        after = np.full(num_workloads, np.nan, dtype=float)
    else:
        after = np.array(result.selected_throughput, dtype=float)
    arrivals = np.array(result.arrival, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(max(11, 6 + num_workloads), 4.5))
    ax_bar, ax_text = axes

    x = np.arange(num_workloads)
    width = 0.25

    ax_bar.bar(x - width, before, width, label="Before (start)")
    ax_bar.bar(x, after, width, label="After (selected)")
    ax_bar.bar(x + width, arrivals, width, label="Arrival target")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"w{idx + 1}" for idx in range(num_workloads)])
    ax_bar.set_ylabel("Throughput")
    ax_bar.set_title("Throughput comparison")
    ax_bar.legend()

    for idx, value in enumerate(before):
        if not math.isnan(value):
            ax_bar.annotate(
                f"{value:.2f}",
                (x[idx] - width, value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
            )
    for idx, value in enumerate(after):
        if not math.isnan(value):
            ax_bar.annotate(
                f"{value:.2f}",
                (x[idx], value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
            )
    for idx, value in enumerate(arrivals):
        if not math.isnan(value):
            ax_bar.annotate(
                f"{value:.2f}",
                (x[idx] + width, value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
            )

    ax_text.axis("off")
    residual_start = ", ".join(f"w{idx + 1}: {result.residual_start[idx]:.2f}" for idx in range(num_workloads))
    if result.residual_selected is not None:
        residual_selected = ", ".join(
            f"w{idx + 1}: {result.residual_selected[idx]:.2f}" for idx in range(num_workloads)
        )
    else:
        residual_selected = "N/A"
    change_lines = ", ".join(
        f"w{idx + 1}: {result.change_factors[idx]:.4f}" for idx in range(num_workloads)
    )
    caps_line = ", ".join(f"w{idx + 1}≤{result.caps[idx]:.1f}%" for idx in range(num_workloads))

    def format_split(split: Tuple[float, ...]) -> str:
        return ", ".join(f"w{idx + 1}={split[idx]:.2f}%" for idx in range(num_workloads))

    text_lines = [
        f"Group: {workloads_label}",
        f"Start split: {format_split(result.start_split)}",
        f"Proposal (post change factor): {format_split(result.proposal_split)}",
        f"Fair allocation (post weighting to 100): {format_split(result.fair_split)}",
        f"Selected split: {format_split(result.selected_split)}",
        f"Iterations to converge: {result.iterations}",
        f"Caps: {caps_line}",
        "",
        "Residuals (throughput - arrival):",
        f"  Start – {residual_start}",
        f"  Selected – {residual_selected}",
        "",
        f"Change factors (|residual| / throughput): {change_lines}",
        "",
        f"Throughput source column: {result.metrics_source or 'N/A'}",
    ]
    ax_text.text(0.0, 1.0, "\n".join(text_lines), va="top")

    fig.tight_layout()

    file_name = "__".join(sanitize_filename(name) for name in result.workloads) + "_alloc.png"
    output_path = output_dir / file_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary(summary_path: Path, records: List[AllocationResult]) -> None:
    """Write allocator results to a CSV summary."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    num_workloads = len(records[0].workloads) if records else 0
    for result in records:
        row: Dict[str, object] = {"iterations": result.iterations, "metrics_source": result.metrics_source}
        for idx in range(num_workloads):
            workload_label = f"workload{idx + 1}"
            freq_label = f"freq{idx + 1}"
            prefix = f"w{idx + 1}"
            row[workload_label] = result.workloads[idx]
            row[freq_label] = result.freqs[idx]
            row[f"{prefix}_proposal_percentage"] = result.proposal_split[idx]
            row[f"{prefix}_fair_percentage"] = result.fair_split[idx]
            row[f"{prefix}_optimal_percentage"] = result.selected_split[idx]
            row[f"{prefix}_arrival"] = result.arrival[idx]
            row[f"{prefix}_change_factor"] = result.change_factors[idx]
            if result.selected_throughput is not None:
                row[f"{prefix}_throughput_at_selected"] = result.selected_throughput[idx]
            else:
                row[f"{prefix}_throughput_at_selected"] = None
            if result.residual_selected is not None:
                row[f"{prefix}_residual_selected"] = result.residual_selected[idx]
            else:
                row[f"{prefix}_residual_selected"] = None
        if result.selected_throughput is not None:
            row["total_throughput_at_selected"] = sum(result.selected_throughput)
        else:
            row["total_throughput_at_selected"] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    if num_workloads >= 1:
        sort_columns = [f"workload{idx + 1}" for idx in range(num_workloads)]
        df.sort_values(sort_columns, inplace=True)
    df.to_csv(summary_path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

#sample run 
"""
python artifacts/baseline_evaluations/gslice/gslice_allocator.py --throughput_csv data/colocations/0311_freq1530_nonDL_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv data/colocations/09152025_freq1530_nodvfs_share_comb2_freqscale_throughput_individual_avg.csv
--caps_csv artifacts/baseline_evaluations/mudi/curvefit/summary/0206_xput_FREQ1530_kneedle_params_baseline.csv --arrival_csv data/baseline_metrics/0206_FREQ1530_baseline_metrics.csv --summary_name freq1530_comb2_summary.csv --fig_dir colocated_freq1530 --log_level WARNING --n_comb 2
n comb ==3
python artifacts/baseline_evaluations/gslice/gslice_allocator.py --throughput_csv data/colocations/09152025_freq300_DL_comb3_share_comb3_freqscale_throughput_individual_avg.csv
--caps_csv artifacts/baseline_evaluations/mudi/curvefit/summary/single_freq300.csv --arrival_csv data/baseline_metrics/0206_FREQ1530_baseline_metrics.csv --summary_name freq1530_comb3_summary.csv --fig_dir colocated_comb3_freq1530 --log_level WARNING --n_comb 3
"""
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--throughput_csv",
        type=Path,
        nargs="+",
        required=True,
        help="Stage-2 throughput grid CSV(s) with (w1_x, w2_y, ... ) columns.",
    )
    parser.add_argument(
        "--caps_csv",
        type=Path,
        required=True,
        help="Caps CSV providing delta0 per workload (e.g., kneepoint summary).",
    )
    parser.add_argument(
        "--arrival_csv",
        type=Path,
        required=True,
        help="Baseline metrics CSV containing Exclusive100 columns for arrivals.",
    )
    parser.add_argument(
        "--n_comb",
        type=int,
        required=True,
        help="Number of workloads colocated together (e.g., 2 for pairs, 3 for triples).",
    )
    parser.add_argument(
        "--start_split",
        type=str,
        default=None,
        help="Starting GPU percentage split expressed as comma-separated percentages. "
        "Defaults to 50/50 for n=2, 30/30/40 for n=3, and uniform otherwise.",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="gslice_optimal_allocation.csv",
        help="Filename for the summary CSV (written under summary/).",
    )
    parser.add_argument(
        "--fig_dir",
        type=Path,
        default=Path("gslice"),
        help="Sub-directory (within fig/) for diagnostic figures.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for console output.",
    )
    return parser


def resolve_figure_directory(fig_dir: Path) -> Path:
    if fig_dir.is_absolute():
        target = fig_dir
    else:
        target = FIG_ROOT / fig_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    num_workloads = args.n_comb
    if num_workloads < 2:
        raise ValueError("--n_comb must be at least 2")

    if args.start_split is None:
        if num_workloads == 2:
            start_split_values = [50.0, 50.0]
        elif num_workloads == 3:
            start_split_values = [30.0, 30.0, 40.0]
        else:
            base = 100.0 / num_workloads
            start_split_values = [base] * num_workloads
            start_split_values[-1] = 100.0 - sum(start_split_values[:-1])
    else:
        try:
            start_split_values = [float(part.strip()) for part in args.start_split.split(",")]
        except ValueError as exc:
            raise ValueError(
                "--start_split must contain comma-separated numeric percentages"
            ) from exc
        if len(start_split_values) != num_workloads:
            raise ValueError(
                f"--start_split expects {num_workloads} values, got {len(start_split_values)}"
            )
    if not math.isclose(sum(start_split_values), 100.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("--start_split must sum to 100")
    start_split = tuple(start_split_values)

    caps_map = load_caps(args.caps_csv)
    arrivals_map = load_arrivals(args.arrival_csv)

    figure_dir = resolve_figure_directory(args.fig_dir)
    summary_path = SUMMARY_ROOT / args.summary_name

    results: List[AllocationResult] = []

    throughput_frames = []
    workload_columns = {f"workload{idx + 1}" for idx in range(num_workloads)}
    for csv_path in args.throughput_csv:
        df = pd.read_csv(csv_path)
        if not workload_columns.issubset(df.columns):
            raise ValueError(
                f"Throughput CSV {csv_path} missing required columns: {workload_columns}"
            )
        throughput_frames.append(df)

    if not throughput_frames:
        raise ValueError("No throughput CSVs provided")

    base_columns = set(throughput_frames[0].columns)
    for csv_path, frame in zip(args.throughput_csv, throughput_frames):
        if set(frame.columns) != base_columns:
            raise ValueError(
                f"Columns in {csv_path} do not match the first throughput CSV"
            )

    throughput_df = pd.concat(throughput_frames, ignore_index=True)

    for _, row in throughput_df.iterrows():
        splits = extract_splits(row, num_workloads=num_workloads)
        try:
            result = run_allocator_for_group(
                row=row,
                splits=splits,
                arrival_map=arrivals_map,
                caps_map=caps_map,
                start_split=start_split,
                num_workloads=num_workloads,
            )
        except ValueError as exc:
            workload_names = [
                str(row.get(f"workload{idx + 1}", "?")).strip() for idx in range(num_workloads)
            ]
            logging.error("Skipping workloads %s due to error: %s", " | ".join(workload_names), exc)
            continue

        results.append(result)
        render_figure(result, figure_dir)

    if not results:
        raise SystemExit("No allocator results produced; see logs for details")

    write_summary(summary_path, results)
    logging.info("Wrote summary to %s", summary_path)
    logging.info("Generated %d figures under %s", len(results), figure_dir)


if __name__ == "__main__":
    main()
