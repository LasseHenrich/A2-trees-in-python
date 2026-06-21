# Plots a grid of decision boundaries across datasets and tree depths.
#
# Usage:
#   python scripts/plotting/plot_boundary_grid.py <name> [<name> ...] \
#       --depths <int|None> [<int|None> ...] [--save PATH]
#
# Each <name> is a dataset name resolved to data/raw/<name>/test.csv.
# A saved model must exist at results/models/<name>.json and must have
# been trained to at least the maximum requested depth.
# Pass None as a depth value to use the full trained tree (no depth limit).

import argparse
import sys
from pathlib import Path


def _parse_int_or_none(value: str) -> int | None:
    return None if value == "None" else int(value)


ROOT = Path(__file__).resolve().parent.parent.parent

from config import RAW_DATA_DIR, MODELS_DIR
from data.utils import find_dataset, load_csv
from decision_tree.io import find_compatible_model, load_tree
from plots import plot_boundary_grid


def _validate(
    names: list[str], max_depth: int | None, criterion: str
) -> list[str]:
    errors = []
    for name in names:
        if find_dataset(name) is None:
            errors.append(f"Dataset not found: {name}")

        if (
            find_compatible_model(
                ROOT / MODELS_DIR, name, criterion, max_depth
            )
            is None
        ):
            errors.append(
                f"No compatible model found for '{name}' "
                f"(criterion={criterion!r}, max_depth={max_depth})."
            )
    return errors


def _run(
    dataset_names: list[str],
    depths: list[int | None],
    criterion: str,
    save_path: str | None,
) -> None:

    int_depths = [d for d in depths if d is not None]
    max_depth = None if None in depths else max(int_depths)
    errors = _validate(dataset_names, max_depth, criterion)

    if errors or len(dataset_names) == 0:
        for msg in errors:
            print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)

    names = []
    datasets = []
    classifiers = []

    for name in dataset_names:
        dataset_dir = find_dataset(name)
        assert dataset_dir is not None
        test_csv = dataset_dir / "test.csv"
        model_json = find_compatible_model(
            ROOT / MODELS_DIR, name, criterion, max_depth
        )
        X, y = load_csv(str(test_csv))
        clf, _ = load_tree(str(model_json))
        names.append(name)
        datasets.append((X, y))
        classifiers.append(clf)

    plot_boundary_grid(
        names,
        datasets,
        classifiers,
        depths,
        criterion=criterion,
        save_path=save_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a grid of decision boundaries for one or more datasets "
            "at one or more tree depths."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="NAME",
        help=(
            "Dataset name(s) to plot. Resolved against "
            "data/processed/ first, then data/raw/."
        ),
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=_parse_int_or_none,
        required=True,
        metavar="N",
        help=(
            "Tree depth(s) to evaluate. "
            "Use None to indicate no depth limit (full tree)."
        ),
    )
    parser.add_argument(
        "--criterion",
        default="gini",
        choices=["gini", "entropy"],
        help="Impurity criterion used by the saved model (default: gini).",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to this path instead of displaying it.",
    )
    args = parser.parse_args()

    int_depths = sorted(d for d in args.depths if d is not None)
    has_none = None in args.depths
    depths = int_depths + ([None] if has_none else [])
    _run(args.datasets, depths, args.criterion, args.save)


if __name__ == "__main__":
    main()
