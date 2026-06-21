# Evaluates a decision tree configuration using k-fold cross-validation.
#
# Usage: python scripts/evaluate.py <dataset> [options]
#
# <dataset> is a path to a directory that contains train.csv.
# Relative paths are resolved from the repo root.
#
# When any of --max-depth, --criterion, or --min-samples-split receives
# more than one value, a grid search is run over every combination.
#
# Examples:
#   python scripts/evaluate.py data/raw/moons --max-depth 5 --criterion gini
#   python scripts/evaluate.py data/raw/moons --max-depth 3 5 10 None

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from data.utils import load_csv
from evaluation.cross_validation import aggregate, cross_validate


def _parse_int_or_none(value: str) -> int | None:
    return None if value == "None" else int(value)


def _build_grid(args: argparse.Namespace) -> dict[str, list]:
    """
    Builds the grid dict from parsed CLI arguments.

    Maps CLI flag values to the keys expected by cross_validate:
        args.max_depth           → "stop_depth"
        args.criterion           → "criterion"
        args.min_samples_split   → "stop_below"
    """
    # ------ WRITE YOUR CODE HERE ------
    pass


def _is_grid_search(args: argparse.Namespace) -> bool:
    """
    Returns True if any flag was given more than one value.

    Determines whether main calls _print_single or _print_grid.
    """
    # ------ WRITE YOUR CODE HERE ------
    pass


def _print_single(aggregated: list[dict]) -> None:
    """Prints full metrics for a single configuration."""
    r = aggregated[0]
    print(f"criterion:  {r['criterion']}")
    print(f"stop_depth: {r['stop_depth']}")
    print(f"stop_below: {r['stop_below']}")
    print()

    for i, acc in enumerate(r["fold_accuracies"]):
        print(f"  fold {i + 1}: {acc:.4f}")
    print()
    print(f"  mean: {r['mean_accuracy']:.4f}")
    print(f"  std:  {r['std_accuracy']:.4f}")
    print()
    print(f"  avg prediction depth: {r['avg_pred_depth']:.2f}")
    print()

    cm = r["confusion_matrix"]
    print("confusion matrix (summed over folds):")
    for row in cm:
        print("  " + "  ".join(f"{v:5d}" for v in row))

    if len(cm) > 2:
        print()
        print("per-class accuracy:")
        for i, row in enumerate(cm):
            total = sum(row)
            print(
                f"  class {i}: {row[i] / total:.4f}"
                if total
                else f"  class {i}: n/a"
            )


def _print_grid(aggregated: list[dict]) -> None:
    """Prints all combinations sorted by mean accuracy, then the winner."""
    ranked = sorted(aggregated, key=lambda r: r["mean_accuracy"], reverse=True)
    header = (
        f"{'criterion':<10}  {'stop_depth':>10}  {'stop_below':>10}"
        f"  {'mean_acc':>8}  {'std_acc':>7}  {'avg_depth':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in ranked:
        print(
            f"{r['criterion']:<10}  {str(r['stop_depth']):>10}"
            f"  {str(r['stop_below']):>10}"
            f"  {r['mean_accuracy']:>8.4f}  {r['std_accuracy']:>7.4f}"
            f"  {r['avg_pred_depth']:>9.2f}"
        )
    best = ranked[0]
    print()
    print(
        f"best: criterion={best['criterion']}"
        f"  stop_depth={best['stop_depth']}"
        f"  stop_below={best['stop_below']}"
        f"  mean_acc={best['mean_accuracy']:.4f}"
    )


def main() -> None:
    # ------ WRITE YOUR CODE HERE ------
    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a decision tree configuration using k-fold "
            "cross-validation."
        )
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
        nargs="+",
        type=_parse_int_or_none,
        default=[None],
        metavar="N",
        help=(
            "Depth value(s) to evaluate. Use 'None' for unlimited. "
            "Multiple values trigger a grid search (default: None)."
        ),
    )
    parser.add_argument(
        "--criterion",
        nargs="+",
        choices=["gini", "entropy"],
        default=["gini"],
        metavar="C",
        help=(
            "Criterion value(s) to evaluate. Multiple values trigger "
            "a grid search (default: gini)."
        ),
    )
    parser.add_argument(
        "--min-samples-split",
        nargs="+",
        type=_parse_int_or_none,
        default=[None],
        metavar="N",
        help=(
            "Minimum samples-per-node value(s) to evaluate. Use 'None' "
            "for the tree default. Multiple values trigger a grid search "
            "(default: None)."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        metavar="K",
        help="Number of CV folds (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for fold shuffle (default: 0).",
    )
    return parser


if __name__ == "__main__":
    main()
