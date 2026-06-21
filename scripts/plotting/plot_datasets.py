# Plots multiple 2D datasets side-by-side in a single figure.
#
# Usage: python scripts/plot_datasets.py <dir1> [<dir2> ...] [--save PATH] [--split {train,test}]
#
# Each directory must contain a train.csv (or test.csv, via --split) with a
# header row. All columns except the last are treated as features; the last
# column is the class label. The script validates that there are exactly two
# feature columns and that every feature value is numeric before plotting.

import csv
import argparse
import sys
from pathlib import Path

from plots import plot_datasets


def load_dataset(
    directory: Path, split: str
) -> tuple[list[list[float]], list[int], str]:
    """Load and validate a 2D dataset from <directory>/<split>.csv.

    Returns (X, y, title) where title is the directory name.

    Raises:
        FileNotFoundError -- if the CSV does not exist
        ValueError        -- if features are not exactly 2-D or not numeric
    """
    csv_path = directory / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No {split}.csv found in {directory}")

    X: list[list[float]] = []
    y: list[int] = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        feature_cols = header[:-1]
        n_features = len(feature_cols)
        if n_features != 2:
            raise ValueError(
                f"{csv_path}: expected 2 feature columns, "
                f"found {n_features} ({feature_cols})"
            )

        for lineno, row in enumerate(reader, start=2):
            features = row[:-1]
            label = row[-1]
            parsed: list[float] = []
            for col, val in zip(feature_cols, features):
                try:
                    parsed.append(float(val))
                except ValueError:
                    raise ValueError(
                        f"{csv_path} line {lineno}: "
                        f"column '{col}' value {val!r} is not numeric"
                    )
            X.append(parsed)
            y.append(int(label))

    return X, y, directory.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot multiple 2D datasets as subplots in one figure."
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        metavar="DIR",
        help="Dataset directories, each containing a train.csv (or test.csv)",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to this path instead of displaying it",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which CSV split to load (default: train)",
    )
    args = parser.parse_args()

    datasets: list[tuple[list[list[float]], list[int]]] = []
    titles: list[str] = []
    errors: list[str] = []

    for raw_dir in args.dirs:
        directory = Path(raw_dir).resolve()
        if not directory.is_dir():
            errors.append(f"{raw_dir!r} is not a directory")
            continue
        try:
            X, y, name = load_dataset(directory, args.split)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            continue
        datasets.append((X, y))
        titles.append(name)

    if errors:
        for msg in errors:
            print(f"Error: {msg}", file=sys.stderr)
        if not datasets:
            sys.exit(1)

    plot_datasets(datasets, titles=titles, save_path=args.save)


if __name__ == "__main__":
    main()
