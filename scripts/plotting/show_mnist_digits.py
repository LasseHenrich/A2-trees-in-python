# Displays one example of each MNIST digit alongside downsampled versions.
#
# Each row shows one version of the dataset (original, 14x14, 7x7).
# Each column shows one digit (0-9). Skips any version whose data
# directory does not yet exist.
#
# Usage: python scripts/plotting/show_mnist_digits.py [--save PATH]

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from data.utils import load_csv
from plots.mnist import plot_digit_samples

VERSIONS = [
    ("Original 28x28", 28, ROOT / "data" / "raw" / "mnist"),
    ("14x14 (factor=2)", 14, ROOT / "data" / "processed" / "mnist-2x"),
    ("7x7 (factor=4)", 7, ROOT / "data" / "processed" / "mnist-4x"),
]


def find_examples(
    X: list[list[float]], y: list, n_classes: int = 10
) -> list[list[float]]:
    """Returns the first row found for each class, ordered 0 to n_classes-1."""
    examples: dict[int, list[float]] = {}
    for row, label in zip(X, y):
        if label not in examples:
            examples[label] = row
        if len(examples) == n_classes:
            break
    return [examples[i] for i in range(n_classes)]


def main() -> None:
    args = _build_parser().parse_args()

    versions = []
    for title, width, data_dir in VERSIONS:
        csv_path = data_dir / "train.csv"
        if not csv_path.exists():
            print(f"Skipping {title!r}: {csv_path} not found.")
            continue
        X, y = load_csv(csv_path)
        versions.append((title, width, find_examples(X, y)))

    if not versions:
        print(
            "No data found. Run "
            "'uv run python scripts/create_datasets.py mnist' first."
        )
        sys.exit(1)

    plot_digit_samples(versions, save_path=args.save)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Display one example of each MNIST digit alongside "
            "downsampled versions."
        )
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
