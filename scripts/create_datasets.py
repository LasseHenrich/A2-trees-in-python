# Generates or downloads a dataset and writes it to disk as train/test CSVs.
#
# Usage: python scripts/create_datasets.py <dataset> [options]
#
# Synthetic datasets (generated locally):
#   moons, circles, blobs, xor, checkerboard, linear
#
# Real datasets (downloaded via scikit-learn):
#   mnist     -- MNIST digits; 784 pixel features, 10 classes (subset)
#
# Special:
#   all -- generate/download every dataset above
#
# Each dataset is written to RAW_DATA_DIR/<dataset>/ as two CSV files
# (train.csv, test.csv).  Numeric features are stored as floats;
# categorical features are stored as strings.  The last column is always
# the integer label y.

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from data.downloaders import download_mnist
from data.generators import (
    make_moons,
    make_circles,
    make_blobs,
    make_xor,
    make_checkerboard,
    make_linear,
)
from data.utils import shuffle_and_split, write_raw_csvs, write_metadata
from plots.datasets import plot_train_test
from config import RAW_DATA_DIR, DATASET_FIGURES_DIR, DEFAULT_SEED

NUM_SAMPLES = 500


# ---------------------------------------------------------------------------
# Synthetic dataset defaults
# ---------------------------------------------------------------------------

dataset_defaults: dict[str, Any] = {
    "moons": {
        "dir_path": "moons",
        "gen_func": make_moons,
        "gen_func_args": {"noise": 0.20},
    },
    "circles": {
        "dir_path": "circles",
        "gen_func": make_circles,
        "gen_func_args": {"noise": 0.10, "factor": 0.5},
    },
    "blobs": {
        "dir_path": "blobs",
        "gen_func": make_blobs,
        "gen_func_args": {"std": 0.5},
    },
    "xor": {
        "dir_path": "xor",
        "gen_func": make_xor,
        "gen_func_args": {"noise": 0.10},
    },
    "checkerboard": {
        "dir_path": "checkerboard",
        "gen_func": make_checkerboard,
        "gen_func_args": {"grid": 4, "noise": 0.05},
    },
    "linear": {
        "dir_path": "linear",
        "gen_func": make_linear,
        "gen_func_args": {"noise": 0.15},
    },
}

SYNTHETIC_DATASETS = set(dataset_defaults.keys())
REAL_DATASETS = {"mnist"}
ALL_DATASETS = SYNTHETIC_DATASETS | REAL_DATASETS


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def run(
    targets: list[str],
    seed: int,
    train_split: float,
    n: int,
    overrides: dict,
) -> None:
    real_downloaders: dict[str, Any] = {
        "mnist": download_mnist,
    }

    for name in targets:
        if name in REAL_DATASETS:
            kwargs: dict[str, Any] = {"seed": seed}
            if name != "mnist":
                kwargs["train_split"] = train_split
            real_downloaders[name](**kwargs)
            out_dir = Path(RAW_DATA_DIR) / name
            with open(out_dir / "train.csv") as f:
                n_train = sum(1 for _ in f) - 1
            with open(out_dir / "test.csv") as f:
                n_test = sum(1 for _ in f) - 1
            write_metadata(
                out_dir,
                n_train,
                n_test,
                {"seed": seed, "train_split": train_split},
            )
        else:
            config = dataset_defaults[name]
            gen_func_args = {**config["gen_func_args"], **overrides}
            X, y = config["gen_func"](n=n, seed=seed, **gen_func_args)
            rows = [[x1, x2, label] for (x1, x2), label in zip(X, y)]
            train_rows, test_rows = shuffle_and_split(rows, seed, train_split)
            out_dir = Path(RAW_DATA_DIR) / config["dir_path"]
            write_raw_csvs(out_dir, ["x1", "x2", "y"], train_rows, test_rows)
            gen_args = {
                "n": n,
                "seed": seed,
                "train_split": train_split,
                **gen_func_args,
            }
            write_metadata(out_dir, len(train_rows), len(test_rows), gen_args)
            figures_dir = Path(DATASET_FIGURES_DIR)
            figures_dir.mkdir(parents=True, exist_ok=True)
            plot_train_test(
                name,
                [[r[0], r[1]] for r in train_rows],
                [r[2] for r in train_rows],
                [[r[0], r[1]] for r in test_rows],
                [r[2] for r in test_rows],
                str(figures_dir / f"{name}.png"),
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic dataset or download a real dataset "
            "and write train/test CSV files."
        )
    )
    parser.add_argument(
        "dataset",
        choices=sorted(ALL_DATASETS) + ["all"],
        help="Dataset to generate/download, or 'all' for every dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data used for training (default: %(default)s)",
    )
    # Synthetic-only options
    parser.add_argument(
        "--n",
        type=int,
        default=NUM_SAMPLES,
        help="Total number of samples, synthetic datasets only "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        help="Noise level (moons, circles, xor, checkerboard, linear)",
    )
    parser.add_argument(
        "--factor",
        type=float,
        help="Inner-ring scale factor (circles only)",
    )
    parser.add_argument(
        "--std",
        type=float,
        help="Blob standard deviation (blobs only)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        help="Grid size (checkerboard only)",
    )
    parser.add_argument(
        "--centers",
        nargs="+",
        metavar="X,Y",
        type=lambda s: tuple(float(v) for v in s.split(",")),
        help="Blob centers as X,Y pairs e.g. --centers -2,-2 2,-2 0,2 "
        "(blobs only)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = sorted(ALL_DATASETS) if args.dataset == "all" else [args.dataset]
    overrides = {
        key: getattr(args, key)
        for key in ("noise", "factor", "std", "grid", "centers")
        if getattr(args, key) is not None
    }
    run(targets, args.seed, args.train_split, args.n, overrides)


if __name__ == "__main__":
    main()
