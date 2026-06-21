import pytest

from evaluation.metrics import accuracy, confusion_matrix


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_accuracy_all_correct():
    assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0


def test_accuracy_all_wrong():
    assert accuracy([0, 0, 1, 1], [1, 1, 0, 0]) == 0.0


def test_accuracy_partial():
    assert accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.75)


def test_accuracy_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        accuracy([], [])


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_returns_classes_and_matrix():
    classes, M = confusion_matrix([0, 1], [0, 1])
    assert classes == [0, 1]
    assert len(M) == 2
    assert len(M[0]) == 2


def test_confusion_matrix_binary_perfect():
    classes, M = confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1])
    assert classes == [0, 1]
    assert M == [[2, 0], [0, 2]]


def test_confusion_matrix_binary_all_wrong():
    classes, M = confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
    assert classes == [0, 1]
    assert M == [[0, 2], [2, 0]]


def test_confusion_matrix_multiclass():
    classes, M = confusion_matrix([0, 1, 2], [0, 2, 1])
    assert classes == [0, 1, 2]
    assert M[0][0] == 1
    assert M[1][2] == 1
    assert M[2][1] == 1


def test_confusion_matrix_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0])


def test_confusion_matrix_empty_raises():
    with pytest.raises(ValueError):
        confusion_matrix([], [])
