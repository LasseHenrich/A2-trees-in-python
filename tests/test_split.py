from decision_tree.impurity import entropy, gini
from decision_tree.split import best_split, candidate_thresholds


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_candidate_thresholds_smoke():
    result = candidate_thresholds([1.0, 2.0, 3.0])
    assert isinstance(result, list)


def test_best_split_smoke():
    result = best_split([[0.0], [1.0]], [0, 1], gini)
    assert result is not None


# ---------------------------------------------------------------------------
# Numeric splits
# ---------------------------------------------------------------------------


def test_best_split_numeric_perfect_separation():
    pass


def test_best_split_numeric_prefers_better_feature():
    pass


def test_best_split_numeric_returns_none_on_pure_node():
    pass


def test_best_split_numeric_returns_none_when_features_constant():
    pass


def test_best_split_numeric_works_with_entropy():
    pass


# ---------------------------------------------------------------------------
# Categorical splits
# ---------------------------------------------------------------------------


def test_best_split_categorical_perfect_separation():
    pass


def test_best_split_categorical_no_gain():
    pass


def test_best_split_categorical_constant_feature():
    pass


def test_best_split_mixed_prefers_informative_feature():
    pass
