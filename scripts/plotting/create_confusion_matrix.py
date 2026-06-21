# Plots a confusion matrix for a saved model evaluated on a test set.
#
# Usage:
#   python scripts/plotting/create_confusion_matrix.py <dataset> [options]
#
# <dataset> is a dataset name resolved to data/raw/<dataset>/test.csv.
# A saved model must exist at results/models/<dataset>_<criterion>.json.
# Pass --max-depth to cap prediction depth (default: no limit).
#
# Examples:
#   python scripts/plotting/create_confusion_matrix.py moons
#   python scripts/plotting/create_confusion_matrix.py mnist \
#       --criterion entropy --max-depth 10 --save results/figures/mnist_cm.png

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from config import MODELS_DIR
from data.utils import find_dataset, load_csv
from decision_tree.io import find_compatible_model, load_tree
from evaluation.metrics import confusion_matrix
from plots import plot_confusion_matrix


def _run(
    name: str,
    criterion: str,
    max_depth: int | None,
    save_path: str | None,
) -> None:
    dataset_dir = find_dataset(name)
    if dataset_dir is None:
        print(f"Error: dataset '{name}' not found.", file=sys.stderr)
        sys.exit(1)
    test_csv = dataset_dir / "test.csv"
    if not test_csv.exists():
        print(f"Error: test set not found at {test_csv}", file=sys.stderr)
        sys.exit(1)

    model_path = find_compatible_model(
        ROOT / MODELS_DIR, name, criterion, max_depth
    )
    if model_path is None:
        print(
            f"Error: no compatible model found for '{name}' "
            f"(criterion={criterion!r}, max_depth={max_depth}).",
            file=sys.stderr,
        )
        sys.exit(1)

    X, y_true = load_csv(str(test_csv))
    clf, _ = load_tree(str(model_path))
    y_pred = clf.predict(X, stop_depth=max_depth)

    classes, M = confusion_matrix(y_true, y_pred)
    depth_label = str(max_depth) if max_depth is not None else "full"
    title = f"{name}  |  {criterion}  |  depth={depth_label}"
    plot_confusion_matrix(M, classes, title=title, save_path=save_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a confusion matrix for a saved model on a test set."
        )
    )
    parser.add_argument(
        "dataset",
        metavar="NAME",
        help="Dataset name found in data/raw/.",
    )
    parser.add_argument(
        "--criterion",
        default="gini",
        choices=["gini", "entropy"],
        help="Impurity criterion used by the saved model (default: gini).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap prediction depth to N levels. "
            "Omit to use the full trained tree."
        ),
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to this path instead of displaying it.",
    )
    args = parser.parse_args()
    _run(args.dataset, args.criterion, args.max_depth, args.save)


if __name__ == "__main__":
    main()
