import pytest

from decision_tree.io import load_tree, save_tree
from decision_tree.impurity import entropy
from decision_tree.tree import DecisionTreeClassifier

DATASET = "data/raw/moons/train.csv"


def test_round_trip_numeric(tmp_path):
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    preds_before = clf.predict(X)

    path = str(tmp_path / "tree.json")
    save_tree(clf, path, DATASET)
    loaded, _ = load_tree(path)

    assert loaded.predict(X) == preds_before


def test_round_trip_categorical(tmp_path):
    X = [["red"], ["red"], ["blue"], ["blue"]]
    y = [0, 0, 1, 1]
    clf = DecisionTreeClassifier(
        max_depth=2, feature_types=["categorical"]
    ).fit(X, y)
    preds_before = clf.predict(X)

    path = str(tmp_path / "tree.json")
    save_tree(clf, path, DATASET)
    loaded, _ = load_tree(path)

    assert loaded.predict(X) == preds_before


def test_round_trip_mixed(tmp_path):
    X = [[1.0, "a"], [2.0, "b"], [3.0, "a"], [4.0, "b"]]
    y = [0, 1, 0, 1]
    clf = DecisionTreeClassifier(
        max_depth=3, feature_types=["numeric", "categorical"]
    ).fit(X, y)
    preds_before = clf.predict(X)

    path = str(tmp_path / "tree.json")
    save_tree(clf, path, DATASET)
    loaded, _ = load_tree(path)

    assert loaded.predict(X) == preds_before


def test_hyperparams_preserved(tmp_path):
    clf = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=5,
        criterion="entropy",
        feature_types=["numeric"],
    ).fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])

    path = str(tmp_path / "tree.json")
    save_tree(clf, path, DATASET)
    loaded, _ = load_tree(path)

    assert loaded.max_depth == 4
    assert loaded.min_samples_split == 5
    assert loaded.criterion == "entropy"
    assert loaded.feature_types == ["numeric"]


def test_dataset_preserved(tmp_path):
    clf = DecisionTreeClassifier(max_depth=1).fit([[0.0], [1.0]], [0, 1])
    path = str(tmp_path / "tree.json")
    save_tree(clf, path, DATASET)
    _, dataset = load_tree(path)

    assert dataset == DATASET


def test_unfitted_tree_round_trip(tmp_path):
    clf = DecisionTreeClassifier(max_depth=3, criterion="gini")
    path = str(tmp_path / "tree.json")
    save_tree(clf, path, DATASET)
    loaded, _ = load_tree(path)

    assert loaded.root is None
    assert loaded.criterion == "gini"


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_tree("nonexistent.json")


def test_entropy_criterion_preserved(tmp_path):
    clf = DecisionTreeClassifier(criterion="entropy").fit(
        [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]
    )
    path = str(tmp_path / "tree.json")
    save_tree(clf, path, DATASET)
    loaded, _ = load_tree(path)

    assert loaded._impurity is entropy
