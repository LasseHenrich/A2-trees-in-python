# Applies a preprocessing pipeline to a raw dataset and saves the result.
#
# Usage: python scripts/preprocess.py <config>
#
# <config> is a path to a feature map JSON config file (see
# src/data/feature_map.py for the format).
#
# The processed data is written to data/processed/<output>/ as train.csv
# and test.csv.  It can then be loaded with load_csv like any raw dataset.
#
# Example:
#   python scripts/preprocess.py configs/preprocessing/circles_radius.json

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from data.feature_map import load_feature_map
from data.utils import infer_feature_types, load_csv, write_processed_csv
from plots import plot_original_and_processed


def preprocess(config_path: str) -> None:
    cfg_path = Path(config_path)
    transform = load_feature_map(str(cfg_path))

    with open(cfg_path) as f:
        cfg = json.load(f)
    dataset: str = cfg["dataset"]
    output: str = cfg["output"]
    feature_names: list[str] = cfg.get("columns", [])
    proc_feature_names: list[str] = cfg.get("select", [])

    raw_dir = ROOT / "data" / "raw" / dataset
    out_dir = ROOT / "data" / "processed" / output
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train: list[list[float]] = []
    y_train: list[int] = []
    X_proc_train: list[list[float]] = []
    X_test: list[list[float]] = []
    y_test: list[int] = []
    X_proc_test: list[list[float]] = []

    for split in ("train", "test"):
        X, y = load_csv(str(raw_dir / f"{split}.csv"))
        feature_types = infer_feature_types(X)
        X_new, _ = transform(X, feature_types)
        write_processed_csv(out_dir / f"{split}.csv", X_new, y)
        print(
            f"  {split}: {len(X_new)} rows, "
            f"{len(X_new[0]) if X_new else 0} features"
        )
        if split == "train":
            X_train, y_train, X_proc_train = X, y, X_new
        else:
            X_test, y_test, X_proc_test = X, y, X_new

    figures_dir = ROOT / "results" / "figures" / "datasets"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = str(figures_dir / f"{output}.png")
    plot_original_and_processed(
        dataset,
        X_train,
        y_train,
        X_proc_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        X_proc_test=X_proc_test,
        feature_names=feature_names or None,
        proc_feature_names=proc_feature_names or None,
        save_path=fig_path,
    )

    print(f"Saved to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a feature map config to a raw dataset and save "
            "the processed train/test CSVs to data/processed/."
        )
    )
    parser.add_argument(
        "config",
        help="Path to a feature map JSON config file",
    )
    args = parser.parse_args()
    preprocess(args.config)


if __name__ == "__main__":
    main()
