"""
Data loading, saving, and train/test splitting.

`load_csv`, `write_processed_csv`, `to_tuples`, `infer_feature_types`,
`shuffle_and_split`, and `write_raw_csvs` are provided.
Implement `train_test_split`.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import csv
import json
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent


def find_dataset(name: str) -> Path | None:
    """
    Finds a dataset directory by name.

    Checks data/processed/<name> first, then data/raw/<name>.

    Arguments:
        name -- dataset directory name (e.g. "moons")

    Returns:
        Path to the dataset directory, or None if not found in either
        location
    """
    for subdir in ("processed", "raw"):
        path = _ROOT / "data" / subdir / name
        if path.is_dir():
            return path
    return None


def load_csv(path: str) -> tuple[list[list], list[int]]:
    """
    Loads a CSV into feature and label lists.

    The first row is treated as a header and skipped. All columns except
    the last are treated as features; the last column is the integer label.

    Each feature value is parsed as a float if possible, and kept as a
    string otherwise. This allows datasets with a mix of numeric and
    categorical features to be loaded without pre-processing.

    Arguments:
        path -- filesystem path to a CSV file

    Returns:
        tuple(X, y) -- X is a list[list[float | str]], y is a list[int]
    """
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            *features, label = row
            parsed: list[float | str] = []
            for v in features:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(v)
            X.append(parsed)
            y.append(int(label))
    return X, y


def _write_csv_file(
    path: str | Path,
    header: list[str],
    rows: list[list],
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_processed_csv(path: str | Path, X: list[list], y: list[int]) -> None:
    """
    Writes preprocessed feature and label lists to a CSV file.

    Column names are generated as f0, f1, ...; the label column is
    named y. This is the inverse of load_csv.

    Arguments:
        path -- filesystem path to write
        X    -- list of feature vectors
        y    -- list of integer labels (same length as X)
    """
    n_features = len(X[0]) if X else 0
    header = [f"f{i}" for i in range(n_features)] + ["y"]
    rows = [list(row) + [label] for row, label in zip(X, y)]
    _write_csv_file(path, header, rows)


def to_tuples(X: list[list], y: list[int]) -> tuple[list[tuple], list[int]]:
    """
    Converts feature and label lists to tuple form.

    The decision tree model expects each feature vector to be a tuple
    (or any sequence). This helper makes that conversion explicit.

    Arguments:
        X -- list of feature vectors (lists)
        y -- list of integer labels

    Returns:
        tuple(X_tuples, y) -- X_tuples is a list[tuple]
    """
    return [tuple(row) for row in X], y


def infer_feature_types(X: list[list]) -> list[str]:
    """
    Infers whether each feature is numeric or categorical.

    Inspects the first row of X. A feature is "categorical" if its value
    is a string, and "numeric" otherwise.

    Arguments:
        X -- list of feature vectors; must be non-empty

    Returns:
        list[str] -- one "numeric" or "categorical" entry per feature
    """
    if not X:
        return []
    return ["categorical" if isinstance(v, str) else "numeric" for v in X[0]]


def shuffle_and_split(
    rows: list[list],
    seed: int,
    train_split: float,
) -> tuple[list[list], list[list]]:
    """
    Shuffles rows and splits them into train and test subsets.

    Arguments:
        rows        -- list of rows to split
        seed        -- random seed for reproducibility
        train_split -- fraction of rows for the train set

    Returns:
        tuple(train_rows, test_rows)
    """
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_train = int(len(rows) * train_split)
    return rows[:n_train], rows[n_train:]


def write_raw_csvs(
    out_dir: Path,
    header: list[str],
    train_rows: list[list],
    test_rows: list[list],
) -> None:
    """
    Writes raw train and test rows to train.csv and test.csv in out_dir.

    Arguments:
        out_dir    -- directory to write into (created if absent)
        header     -- list of column names
        train_rows -- rows for train.csv
        test_rows  -- rows for test.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train_rows), ("test", test_rows)]:
        _write_csv_file(out_dir / f"{name}.csv", header, rows)
    print(
        f"Wrote {len(train_rows)} train / {len(test_rows)} test rows"
        f" to {out_dir}/"
    )


def write_metadata(
    out_dir: Path,
    n_train: int,
    n_test: int,
    gen_args: dict,
) -> None:
    """
    Writes dataset metadata to metadata.json in out_dir.

    Arguments:
        out_dir  -- directory to write into
        n_train  -- number of training samples
        n_test   -- number of test samples
        gen_args -- generation arguments to record
    """
    meta = {"n_train": n_train, "n_test": n_test, "args": gen_args}
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {out_dir}/metadata.json")


def train_test_split(
    X: list[list[float]],
    y: list[int],
    test_size: float = 0.2,
    seed: int | None = None,
) -> tuple[list[list[float]], list[list[float]], list[int], list[int]]:
    """
    Splits parallel feature and label lists into train and test sets.

    Shuffles the data before splitting using a local `random.Random(seed)`
    instance so the shuffle is independent of global RNG state. The first
    `round(len(X) * test_size)` shuffled samples form the test set.

    Arguments:
        X         -- list of feature vectors
        y         -- list of labels (same length as X)
        test_size -- fraction of samples for the test set (0 < test_size < 1)
        seed      -- optional int; fixes the shuffle for reproducibility

    Returns:
        tuple(X_train, X_test, y_train, y_test)

    Raises:
        ValueError -- if X and y differ in length
        ValueError -- if test_size is not strictly between 0 and 1
    """

    # ------ WRITE YOUR CODE HERE ------
    pass
