"""
Real dataset downloaders.

Each function downloads a dataset via scikit-learn's fetch_openml and
writes train.csv / test.csv to data/raw/<name>/.
"""

from pathlib import Path

from config import DEFAULT_SEED, RAW_DATA_DIR
from .utils import write_raw_csvs

MNIST_TRAIN_PER_DIGIT = 80
MNIST_TEST_PER_DIGIT = 60


def download_mnist(
    seed: int = DEFAULT_SEED,
    train_per_digit: int = MNIST_TRAIN_PER_DIGIT,
    test_per_digit: int = MNIST_TEST_PER_DIGIT,
) -> None:
    """Downloads a balanced MNIST subset and writes train/test CSVs."""
    import numpy as np  # type: ignore[import-untyped]
    from sklearn.datasets import fetch_openml  # type: ignore[import-untyped]

    print("Downloading MNIST from OpenML (this may take a moment)...")
    X_full, y_full = fetch_openml(
        "mnist_784", as_frame=False, return_X_y=True, parser="auto"
    )
    y_int = y_full.astype(int)

    rng = np.random.RandomState(seed)
    n_per_digit = train_per_digit + test_per_digit

    train_idx: list[int] = []
    test_idx: list[int] = []
    for digit in range(10):
        idx = np.where(y_int == digit)[0]
        chosen = rng.choice(idx, size=n_per_digit, replace=False)
        train_idx.extend(chosen[:train_per_digit].tolist())
        test_idx.extend(chosen[train_per_digit:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    n_pixels = X_full.shape[1]
    header = [f"pixel{i}" for i in range(n_pixels)] + ["y"]
    train_rows = [
        list(map(float, X_full[i])) + [int(y_int[i])] for i in train_idx
    ]
    test_rows = [
        list(map(float, X_full[i])) + [int(y_int[i])] for i in test_idx
    ]

    write_raw_csvs(Path(RAW_DATA_DIR) / "mnist", header, train_rows, test_rows)
