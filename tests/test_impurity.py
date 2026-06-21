import math

import pytest

from decision_tree.impurity import entropy, gini, information_gain


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_gini_smoke():
    assert isinstance(gini([0, 1]), float)


def test_entropy_smoke():
    assert isinstance(entropy([0, 1]), float)


def test_information_gain_smoke():
    result = information_gain([0, 1], [0], [1], gini)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# gini
# ---------------------------------------------------------------------------


def test_gini_pure_node():
    pass


def test_gini_50_50_binary():
    pass


def test_gini_three_class_uniform():
    pass


def test_gini_empty():
    pass


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------


def test_entropy_pure_node():
    pass


def test_entropy_50_50_binary():
    pass


def test_entropy_three_class_uniform():
    pass


def test_entropy_empty():
    pass


# ---------------------------------------------------------------------------
# information_gain
# ---------------------------------------------------------------------------


def test_information_gain_perfect_split():
    pass


def test_information_gain_no_split():
    pass


def test_information_gain_empty_parent_raises():
    pass


def test_information_gain_children_mismatch_raises():
    pass
