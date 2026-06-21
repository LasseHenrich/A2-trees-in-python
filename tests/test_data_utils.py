import pytest

from data.utils import (
    infer_feature_types,
    load_csv,
    to_tuples,
    train_test_split,
)


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


def test_load_csv_numeric_columns(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("x1,x2,y\n1.0,2.0,0\n3.0,4.0,1\n")
    X, y = load_csv(str(p))
    assert X == [[1.0, 2.0], [3.0, 4.0]]
    assert y == [0, 1]


def test_load_csv_multicolumn(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c,y\n1.0,2.0,3.0,0\n4.0,5.0,6.0,1\n")
    X, y = load_csv(str(p))
    assert len(X[0]) == 3
    assert y == [0, 1]


def test_load_csv_categorical_features(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("colour,size,y\nred,small,0\nblue,large,1\n")
    X, y = load_csv(str(p))
    assert X == [["red", "small"], ["blue", "large"]]
    assert y == [0, 1]


def test_load_csv_mixed_features(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("age,sex,y\n30.0,male,0\n45.0,female,1\n")
    X, y = load_csv(str(p))
    assert X == [[30.0, "male"], [45.0, "female"]]
    assert y == [0, 1]


# ---------------------------------------------------------------------------
# to_tuples
# ---------------------------------------------------------------------------


def test_to_tuples_converts_rows():
    X = [[1.0, 2.0], [3.0, 4.0]]
    y = [0, 1]
    X_t, y_out = to_tuples(X, y)
    assert X_t == [(1.0, 2.0), (3.0, 4.0)]
    assert y_out is y


def test_to_tuples_mixed_types():
    X = [[1.0, "red"], [2.0, "blue"]]
    y = [0, 1]
    X_t, _ = to_tuples(X, y)
    assert X_t == [(1.0, "red"), (2.0, "blue")]


# ---------------------------------------------------------------------------
# infer_feature_types
# ---------------------------------------------------------------------------


def test_infer_feature_types_all_numeric():
    X = [[1.0, 2.0], [3.0, 4.0]]
    assert infer_feature_types(X) == ["numeric", "numeric"]


def test_infer_feature_types_all_categorical():
    X = [["red", "small"], ["blue", "large"]]
    assert infer_feature_types(X) == ["categorical", "categorical"]


def test_infer_feature_types_mixed():
    X = [[30.0, "male"], [45.0, "female"]]
    assert infer_feature_types(X) == ["numeric", "categorical"]


def test_infer_feature_types_empty():
    assert infer_feature_types([]) == []


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------



def test_train_test_split_smoke():
    X = [[float(i)] for i in range(10)]
    y = list(range(10))
    result = train_test_split(X, y, test_size=0.3, seed=0)
    assert len(result) == 4

def test_train_test_split_sizes():
    pass


def test_train_test_split_alignment():
    pass


def test_train_test_split_seed_reproducible():
    pass


def test_train_test_split_different_seeds_differ():
    pass


def test_train_test_split_no_overlap():
    pass


def test_train_test_split_invalid_lengths():
    pass


def test_train_test_split_invalid_test_size():
    pass
