"""Utility to derive latency CSVs from throughput outputs."""
from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Iterable, Union

Number = Union[int, float]


def format_inverse(value: float) -> str:
    """Return the latency representation for the provided numeric value."""
    return f"{value:.6f}"


def invert_float(value: float) -> str:
    """Invert a floating-point throughput value, handling zeros gracefully."""
    if value == 0:
        return "inf"
    return format_inverse(1.0 / value)


def invert_sequence(values: Iterable[Number]) -> str:
    """Invert every numeric element of a tuple/list and serialise to string."""
    inverted = [invert_float(float(item)) for item in values]
    return f"({', '.join(inverted)})"


def transform_cell(cell: str) -> str:
    """Transform a CSV cell by inverting numeric values when possible."""
    stripped = cell.strip()
    if not stripped:
        return cell

    try:
        numeric = float(stripped)
    except ValueError:
        pass
    else:
        return invert_float(numeric)

    if stripped.startswith("(") and stripped.endswith(")"):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return cell

        if isinstance(parsed, (tuple, list)) and all(isinstance(item, (int, float)) for item in parsed):
            return invert_sequence(parsed)

    return cell


def derive_output_path(input_path: Path, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output

    replaced_name = input_path.name.replace("throughput", "latency")
    if replaced_name == input_path.name:
        raise ValueError(
            "Input file name must contain 'throughput' or specify --output to describe latency file."
        )

    return input_path.with_name(replaced_name)


def convert_file(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as src:
        reader = csv.reader(src)
        rows = []
        for row_index, row in enumerate(reader):
            converted_row = []
            for column_index, cell in enumerate(row):
                if row_index == 0 or column_index in {2, 3}:
                    converted_row.append(cell)
                else:
                    converted_row.append(transform_cell(cell))
            rows.append(converted_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst)
        writer.writerows(rows)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Path to the throughput CSV to convert.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit path for the latency CSV. Defaults to replacing 'throughput' with 'latency'.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    output_path = derive_output_path(args.input_csv, args.output)
    convert_file(args.input_csv, output_path)


if __name__ == "__main__":
    main()
