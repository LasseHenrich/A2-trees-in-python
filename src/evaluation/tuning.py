"""
Hyperparameter Search

Task D: Implement `stopping_search` and `grid_search`.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

from itertools import product

from decision_tree.tree import DecisionTreeClassifier
from evaluation.metrics import accuracy, confusion_matrix


def stopping_search(
    tree: DecisionTreeClassifier,
    X: list[list],
    y: list,
    stop_depths: list[int | None],
    stop_belows: list[int | None],
) -> list[dict]:
    """
    Evaluates a fitted tree over every (stop_depth, stop_below) pair
    without retraining.

    Arguments:
        tree        -- a fitted DecisionTreeClassifier
        X           -- feature vectors
        y           -- true labels (same length as X)
        stop_depths -- list of stop_depth values to try
        stop_belows -- list of stop_below values to try

    Returns:
        A list of dicts of the format below with one dict per combination:
        {
            "stop_depth":       int | None,
            "stop_below":       int | None,
            "accuracy":         float,
            "confusion_matrix": list[list[int]],
            "avg_pred_depth":   float,
        }
    """

    # ------ WRITE YOUR CODE HERE ------
    pass


def grid_search(
    X_train: list[list],
    y_train: list,
    X_val: list[list],
    y_val: list,
    grid: dict[str, list],
) -> list[dict]:
    """
    Searches over criterion, stop_depth, and stop_below by fitting one
    tree per criterion on the training data and calling stopping_search
    on each against the validation data.

    Each tree is trained only as deep as the grid requires:
    - max_depth  = largest stop_depth value (None if None is in the list)
    - min_samples_split = smallest stop_below value (default if None is
      in the list)

    Arguments:
        X_train -- training feature vectors
        y_train -- training labels
        X_val   -- validation feature vectors
        y_val   -- validation labels
        grid    -- dict with keys "criterion", "stop_depth", "stop_below",
                   each mapping to a list of values to try

    Returns:
        A list of dicts, one per (criterion, stop_depth, stop_below)
        combination, each with keys:
        {
            "criterion":        str,
            "stop_depth":       int | None,
            "stop_below":       int | None,
            "accuracy":         float,
            "confusion_matrix": list[list[int]],
        }
    """

    # ------ WRITE YOUR CODE HERE ------
    pass
