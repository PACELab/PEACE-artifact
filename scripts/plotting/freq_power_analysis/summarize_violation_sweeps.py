#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate violation sweep metrics into a single CSV row per configuration."
    )
    parser.add_argument(
        "output_root",
        nargs="?",
        default="test",
        help="Root directory containing powercap_<cap>_window<window> subdirectories (default: %(default)s).",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Path for the consolidated CSV (default: <output_root>/violation_sweep_summary.csv).",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            metric_name = row.get("metric")
            value = row.get("value")
            if not metric_name:
                continue
            try:
                metrics[metric_name] = float(value)
            except (TypeError, ValueError):
                continue
    return metrics


def main() -> None:
    args = parse_args()

    root = Path(args.output_root).resolve()
    if not root.exists():
        raise SystemExit(f"Output root does not exist: {root}")

    summary_path = (
        Path(args.summary).resolve()
        if args.summary is not None
        else root / "violation_sweep_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"powercap_(\d+)_window(\d+)")
    records: list[dict[str, float | int | None]] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.fullmatch(entry.name)
        if not match:
            continue

        powercap = int(match.group(1))
        window_seconds = int(match.group(2))
        metrics_file = entry / "violation_metrics_summary.csv"
        if not metrics_file.exists():
            print(f"Warning: missing {metrics_file}")
            continue

        metrics = load_metrics(metrics_file)
        avg_pct_value = metrics.get("window_violation_pct_mean")
        running_violation_rate_pct = metrics.get("running_violation_rate_pct")
        overcap_mean_watts = metrics.get("running_avg_overcap_mean_mean_watts")

        records.append(
            {
                "powercap": powercap,
                "window_seconds": window_seconds,
                "window_violation_pct_mean": avg_pct_value,
                "workload_with_violation_rate_pct": running_violation_rate_pct,
                "running_avg_overcap_mean_mean_watts": overcap_mean_watts,
                "running_avg_overcap_minuspowercap_mean_watts": overcap_mean_watts - powercap if overcap_mean_watts is not None else None,
                #add percentage of running_avg_overcap_minuspowercap_mean_watts / powercap
                "running_avg_overcap_minuspowercap_mean_pct": (overcap_mean_watts - powercap) / powercap * 100 if overcap_mean_watts is not None else None,
            }
        )

    records.sort(key=lambda item: (item["powercap"], item["window_seconds"]))

    fieldnames = [
        "powercap",
        "window_seconds",
        "window_violation_pct_mean",
        "workload_with_violation_rate_pct",
        "running_avg_overcap_mean_mean_watts",
        "running_avg_overcap_minuspowercap_mean_watts",
        "running_avg_overcap_minuspowercap_mean_pct",
    ]
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
