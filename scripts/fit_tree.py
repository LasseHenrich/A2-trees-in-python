# Fits a decision tree on a dataset and saves the model to a JSON file.
#
# Usage: python scripts/fit_tree.py <dataset> [options]
#
# <dataset> is a path to a directory that contains train.csv.
# Relative paths are resolved from the repo root.
#
# Example:
#   python scripts/fit_tree.py data/raw/moons --max-depth 3 --criterion entropy

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from decision_tree.factories import create_decision_tree
from decision_tree.io import model_filename


def fit_tree(
    dataset_path: str,
    max_depth: int | None,
    criterion: str,
    min_samples_split: int,
    output_dir: str | None,
) -> None:
    dataset_name = Path(dataset_path).name
    out_dir = output_dir or str(ROOT / "results" / "models")

    try:
        create_decision_tree(
            dataset=dataset_name,
            output_dir=out_dir,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    filename = model_filename(
        dataset_name, criterion, max_depth, min_samples_split
    )
    print(f"Model saved to {Path(out_dir) / filename}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a decision tree on a dataset and save the model."
    )
    parser.add_argument(
        "dataset",
        help=(
            "Path to a dataset directory containing train.csv. "
            "Relative paths are resolved from the repo root."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum depth of the tree. Omit to grow until leaves are pure."
        ),
    )
    parser.add_argument(
        "--criterion",
        choices=["gini", "entropy"],
        default="gini",
        help=("Impurity criterion used to evaluate splits (default: gini)."),
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        metavar="N",
        help=(
            "A node with fewer than N samples becomes a leaf without "
            "searching for a split (default: 2)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory to write the model JSON file. Defaults to "
            "results/models/. The filename encodes the dataset name, "
            "criterion, and any non-default hyperparameters."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    fit_tree(
        dataset_path=args.dataset,
        max_depth=args.max_depth,
        criterion=args.criterion,
        min_samples_split=args.min_samples_split,
        output_dir=args.output_dir,
    )
