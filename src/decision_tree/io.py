"""
Tree model serialization.

save_tree             -- write a fitted DecisionTreeClassifier to a JSON file
load_tree             -- load a DecisionTreeClassifier from a JSON file
find_compatible_model -- locate a usable saved model in a directory
"""

import json
from pathlib import Path

from .tree import DecisionTreeClassifier, Node


def _node_to_dict(node: Node | None) -> dict | None:
    if node is None:
        return None
    return {
        "prediction": node.prediction,
        "n_samples": node.n_samples,
        "feature": node.feature,
        "threshold": node.threshold,
        "category_value": node.category_value,
        "left": _node_to_dict(node.left),
        "right": _node_to_dict(node.right),
    }


def _dict_to_node(d: dict | None) -> Node | None:
    if d is None:
        return None
    return Node(
        prediction=d["prediction"],
        n_samples=d.get("n_samples", 0),
        feature=d["feature"],
        threshold=d["threshold"],
        category_value=d["category_value"],
        left=_dict_to_node(d["left"]),
        right=_dict_to_node(d["right"]),
    )


def model_filename(
    dataset_name: str,
    criterion: str,
    max_depth: int | None,
    min_samples_split: int,
) -> str:
    parts = [dataset_name, criterion]
    if max_depth is not None:
        parts.append(f"md_{max_depth}")
    if min_samples_split != 2:
        parts.append(f"mss_{min_samples_split}")
    return "_".join(parts) + ".json"


def save_tree(
    clf: DecisionTreeClassifier,
    path: str,
    dataset: str,
) -> None:
    """
    Saves a fitted DecisionTreeClassifier to a JSON file.

    The file stores the dataset identifier, hyperparameters,
    and the full tree structure so that load_tree can
    reconstruct a ready-to-use classifier without retraining.

    Arguments:
        clf     -- a DecisionTreeClassifier (fitted or not)
        path    -- filesystem path to write the JSON file to
        dataset -- path or name of the training dataset
    """
    payload = {
        "dataset": dataset,
        "hyperparams": {
            "max_depth": clf.max_depth,
            "min_samples_split": clf.min_samples_split,
            "criterion": clf.criterion,
            "feature_types": clf.feature_types,
        },
        "tree": _node_to_dict(clf.root),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_tree(
    path: str,
) -> tuple[DecisionTreeClassifier, str]:
    """
    Loads a DecisionTreeClassifier from a JSON file written
    by save_tree.

    Arguments:
        path -- filesystem path to the JSON file

    Returns:
        (clf, dataset) where clf is ready to call .predict()
        and dataset is the training dataset identifier.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    clf = DecisionTreeClassifier(**payload["hyperparams"])
    clf.root = _dict_to_node(payload["tree"])
    return clf, payload["dataset"]


def find_compatible_model(
    root_dir: str | Path,
    dataset_name: str,
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_split: int = 2,
) -> Path | None:
    """
    Finds the best saved model in root_dir that is compatible with
    the given hyperparameters.

    A model is compatible when:
      - Its dataset name and criterion match exactly.
      - Its max_depth is None, or >= the requested max_depth (when
        max_depth is not None). A deeper tree can simulate a shallower
        one at inference; an unrestricted tree can simulate any depth.
        If max_depth is None, only an unrestricted model is accepted.
      - Its min_samples_split is <= the requested min_samples_split.
        A model trained with a smaller (more permissive) threshold has
        more splits available, so a higher threshold can be simulated
        at inference.

    Among compatible models, those without restrictions (max_depth=None,
    min_samples_split=2) are preferred, then larger max_depth, then
    smaller min_samples_split.

    Returns the path to the best match, or None if none is found.
    """
    candidates = []
    for path in Path(root_dir).glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError, OSError:
            continue

        hp = payload.get("hyperparams", {})
        if Path(payload.get("dataset", "")).parent.name != dataset_name:
            continue
        if hp.get("criterion") != criterion:
            continue

        m_md = hp.get("max_depth")
        m_mss = hp.get("min_samples_split", 2)

        if max_depth is None:
            if m_md is not None:
                continue
        else:
            if m_md is not None and m_md < max_depth:
                continue

        if m_mss > min_samples_split:
            continue

        candidates.append((path, m_md, m_mss))

    if not candidates:
        return None

    def _sort_key(item: tuple) -> tuple:
        _, md, mss = item
        # None (unrestricted) sorts before any integer max_depth.
        # Among integer values, larger md is preferred (negate for asc sort).
        md_val = md if md is not None else float("inf")
        return (md is not None, -md_val, mss)

    candidates.sort(key=_sort_key)
    return candidates[0][0]
