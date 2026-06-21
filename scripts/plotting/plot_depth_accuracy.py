# Plots training and test accuracy across a range of tree depths.
#
# Usage: python scripts/plotting/plot_depth_accuracy.py <name> [options]
#
# <name> is a dataset name resolved against data/raw/ and data/processed/.
# A saved model must exist in results/models/ trained without a depth
# limit (fit_tree.py without --max-depth).
#
# Example:
#   python scripts/plotting/plot_depth_accuracy.py moons --max-depth 15

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from config import MODELS_DIR
from data.utils import find_dataset, load_csv
from decision_tree.io import find_compatible_model, load_tree
from plots import plot_depth_accuracy


def main() -> None:
    args = _build_parser().parse_args()

    dataset_dir = find_dataset(args.dataset)
    if dataset_dir is None:
        print(f"Error: dataset '{args.dataset}' not found.", file=sys.stderr)
        sys.exit(1)

    model_path = find_compatible_model(
        ROOT / MODELS_DIR,
        args.dataset,
        args.criterion,
        max_depth=args.max_depth,
    )
    if model_path is None:
        print(
            f"Error: no model found for '{args.dataset}' "
            f"(criterion='{args.criterion}'). "
            "Run fit_tree.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_train, y_train = load_csv(str(dataset_dir / "train.csv"))
    X_test, y_test = load_csv(str(dataset_dir / "test.csv"))
    clf, _ = load_tree(str(model_path))

    depths: list[int | None] = list(range(1, args.max_depth + 1))
    title = f"{args.dataset}  |  {args.criterion}  |  accuracy vs. depth"
    plot_depth_accuracy(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        depths,
        title=title,
        save_path=args.save,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training and test accuracy across a range of tree depths."
        )
    )
    parser.add_argument(
        "dataset",
        metavar="NAME",
        help="Dataset name found in data/raw/ or data/processed/.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=15,
        metavar="N",
        help="Largest depth value to plot (default: 15).",
    )
    parser.add_argument(
        "--criterion",
        default="gini",
        choices=["gini", "entropy"],
        help="Criterion of the saved model (default: gini).",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to this path instead of displaying it.",
    )
    return parser


if __name__ == "__main__":
    main()
