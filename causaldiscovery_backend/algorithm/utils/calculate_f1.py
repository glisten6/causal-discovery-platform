"""Utility script to compute F1-score from recall and precision columns in CSV files.

Usage:
    python calculate_f1.py --input path/to/results.csv [--output path/to/result_with_f1.csv]

The script reads the CSV, computes F1-score for each row using the formula:
    F1 = 2 * (precision * recall) / (precision + recall)
where precision and recall are taken from columns named "precision" and "recall" by default.

An optional --output argument allows saving the augmented CSV with a new
"f1_score" column. If --output is omitted, the script overwrites the input
file in-place.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def compute_f1(recall: float, precision: float, eps: float = 1e-12) -> float:
    """Return the F1-score given recall and precision.

    Parameters
    ----------
    recall : float
        Recall value (between 0 and 1).
    precision : float
        Precision value (between 0 and 1).
    eps : float, optional
        Small value to avoid division by zero. Defaults to 1e-12.
    """
    denom = recall + precision
    if denom <= eps:
        return 0.0
    return 2.0 * recall * precision / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute F1-score from recall and precision columns in a CSV file.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the CSV with the new f1_score column. Defaults to overwriting the input file.",
    )
    parser.add_argument(
        "--recall-column",
        default="recall",
        help="Column name for recall values. Defaults to 'recall'.",
    )
    parser.add_argument(
        "--precision-column",
        default="precision",
        help="Column name for precision values. Defaults to 'precision'.",
    )
    parser.add_argument(
        "--f1-column",
        default="f1_score",
        help="Column name for the computed F1-score. Defaults to 'f1_score'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input)

    if args.recall_column not in df.columns:
        print(f"Recall column '{args.recall_column}' not found in CSV.", file=sys.stderr)
        return 1
    if args.precision_column not in df.columns:
        print(f"Precision column '{args.precision_column}' not found in CSV.", file=sys.stderr)
        return 1

    df[args.f1_column] = [
        compute_f1(rec, prec)
        for rec, prec in zip(df[args.recall_column], df[args.precision_column])
    ]

    output_path = args.output if args.output is not None else args.input
    df.to_csv(output_path, index=False)
    print(f"Saved CSV with F1-score column to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python D:\data\python_project\causal_discovery\algorithm\utils\calculate_f1.py --input D:\data\python_project\causal_discovery\algorithm\CIR\exp\pytorch_optimizer\different_active_constraint_method\lambda\4\200\results_by_seed.csv
# # 或另存
# python algorithm/CIR/utils/calculate_f1.py --input path/to/results.csv --output path/to/results_with_f1.csv