import pytest

from decision_tree.tree import DecisionTreeClassifier
from evaluation.tuning import grid_search, stopping_search


# ---------------------------------------------------------------------------
# stopping_search
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_stopping_search_smoke():
    tree = DecisionTreeClassifier(max_depth=3).fit(
        [[0.0], [1.0], [2.0]], [0, 0, 1]
    )
    results = stopping_search(tree, [[0.0]], [0], [None], [None])
    assert isinstance(results, list)


def test_grid_search_smoke():
    grid = {
        "criterion": ["gini"],
        "stop_depth": [None],
        "stop_below": [None],
    }
    results = grid_search(
        [[0.0], [1.0]], [0, 1], [[0.0], [1.0]], [0, 1], grid
    )
    assert isinstance(results, list)


def test_stopping_search_returns_one_result_per_combination():
    pass


def test_stopping_search_result_has_expected_keys():
    pass


def test_stopping_search_zero_depth_gives_avg_depth_zero():
    pass


# ---------------------------------------------------------------------------
# grid_search
# ---------------------------------------------------------------------------


def test_grid_search_finds_best_params():
    pass


def test_grid_search_result_has_expected_keys():
    pass


def test_grid_search_empty_grid_raises():
    pass
