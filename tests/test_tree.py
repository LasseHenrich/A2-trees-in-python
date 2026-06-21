import pytest

from decision_tree.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Numeric features (existing behaviour)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_tree_fit_smoke():
    tree = DecisionTreeClassifier(max_depth=3).fit([[0.0], [1.0]], [0, 1])
    assert tree.root is not None


def test_tree_predict_smoke():
    tree = DecisionTreeClassifier(max_depth=3).fit([[0.0], [1.0]], [0, 1])
    result = tree.predict([[0.0]])
    assert isinstance(result, list)


def test_tree_fits_perfectly_on_separable_data():
    pass


def test_tree_max_depth_one_produces_single_split():
    pass


def test_tree_pure_input_becomes_leaf():
    pass


def test_tree_min_samples_split_prevents_growth():
    pass


def test_tree_predict_before_fit_raises():
    pass


def test_tree_nonlinear_pattern_requires_depth():
    pass


def test_tree_entropy_criterion():
    pass


# ---------------------------------------------------------------------------
# Categorical features
# ---------------------------------------------------------------------------


def test_tree_categorical_features():
    pass


def test_tree_categorical_node_stores_category_value():
    pass


def test_tree_mixed_numeric_and_categorical():
    pass


# ---------------------------------------------------------------------------
# Node metadata
# ---------------------------------------------------------------------------


def test_tree_root_n_samples_equals_training_size():
    pass


# ---------------------------------------------------------------------------
# predict stop_depth
# ---------------------------------------------------------------------------


def test_predict_stop_depth_zero_uses_root_prediction():
    pass


def test_predict_stop_depth_limits_tree_depth():
    pass


# ---------------------------------------------------------------------------
# predict stop_below
# ---------------------------------------------------------------------------


def test_predict_stop_below_large_value_uses_root_prediction():
    pass


# ---------------------------------------------------------------------------
# predict_with_depth
# ---------------------------------------------------------------------------


def test_predict_with_depth_returns_prediction_and_depth():
    pass


def test_predict_with_depth_root_leaf_has_depth_zero():
    pass


def test_predict_with_depth_stop_depth_zero_has_depth_zero():
    pass


def test_predict_with_depth_deeper_tree_reaches_greater_depths():
    pass


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_criterion_raises():
    pass


def test_invalid_max_depth_raises():
    pass


def test_invalid_min_samples_split_raises():
    pass
