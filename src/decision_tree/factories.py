"""
Factory functions for constructing fitted decision tree models.
"""

from pathlib import Path

from data.utils import find_dataset, infer_feature_types, load_csv

from .io import model_filename, save_tree
from .tree import DecisionTreeClassifier


def create_decision_tree(
    dataset: str,
    output_dir: str,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    criterion: str = "gini",
    feature_types: list[str] | None = None,
) -> DecisionTreeClassifier:
    """
    Finds a dataset by name, fits a decision tree, saves it, and returns it.

    Looks for the dataset in data/processed/<dataset> first, then
    data/raw/<dataset>. Infers feature types from the data when
    feature_types is None. The model is written to
    output_dir/<filename> where filename is produced by model_filename.

    Arguments:
        dataset           -- dataset directory name (e.g. "moons")
        output_dir        -- directory to write the model JSON file
        max_depth         -- maximum tree depth; None grows until pure
        min_samples_split -- nodes with fewer samples become leaves
        criterion         -- "gini" or "entropy"
        feature_types     -- one "numeric" or "categorical" per feature;
                             inferred from the data when None

    Returns:
        DecisionTreeClassifier -- fitted classifier

    Raises:
        FileNotFoundError -- if dataset is not found in data/processed
                             or data/raw
    """
    dataset_dir = find_dataset(dataset)
    if dataset_dir is None:
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found in data/processed or data/raw"
        )
    train_path = dataset_dir / "train.csv"

    X, y = load_csv(str(train_path))

    if feature_types is None:
        feature_types = infer_feature_types(X)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        feature_types=feature_types,
    )
    clf.fit(X, y)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = model_filename(dataset, criterion, max_depth, min_samples_split)
    out_path = out_dir / filename
    save_tree(clf, str(out_path), str(train_path))

    return clf
