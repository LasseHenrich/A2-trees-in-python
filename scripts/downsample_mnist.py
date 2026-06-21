# Applies average-pool downsampling to the MNIST dataset.
#
# Usage: python scripts/downsample_mnist.py [--factor N]
#
# Reads data/raw/mnist/train.csv and test.csv.
# Writes downsampled CSVs to data/processed/mnist-{factor}x/.
#
# Example:
#   python scripts/downsample_mnist.py --factor 2   # 784 → 196 features
#   python scripts/downsample_mnist.py --factor 4   # 784 → 49 features

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from data.feature_map import make_downsample_img
from data.utils import load_csv, write_processed_csv

MNIST_WIDTH = 28


def downsample(factor: int) -> None:
    transform = make_downsample_img(MNIST_WIDTH, factor)
    out_dir = ROOT / "data" / "processed" / f"mnist-{factor}x"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        src = ROOT / "data" / "raw" / "mnist" / f"{split}.csv"
        X, y = load_csv(src)
        X_new = [transform(row) for row in X]
        write_processed_csv(out_dir / f"{split}.csv", X_new, y)
        n = len(X_new)
        f = len(X_new[0]) if X_new else 0
        print(f"{split}: {n} rows, {f} features → {out_dir / f'{split}.csv'}")


def main() -> None:
    args = _build_parser().parse_args()
    downsample(args.factor)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average-pool downsample the MNIST dataset."
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Pooling block size (default: 2). "
            "Must divide 28 evenly (2 → 196 features, 4 → 49 features)."
        ),
    )
    return parser


if __name__ == "__main__":
    main()
