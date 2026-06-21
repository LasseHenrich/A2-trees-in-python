"""
Evaluation Utilities (provided)

Accuracy and confusion matrix for measuring classifier quality.
Use these as-is.
"""


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the fraction of predictions that match the true label.

    Arguments:
        y_true -- list of ground-truth labels
        y_pred -- list of predicted labels (same length as y_true)

    Returns:
        float -- accuracy in [0, 1]

    Raises:
        ValueError -- if the lists differ in length or are empty
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Label lists must not be empty")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(
    y_true: list[int], y_pred: list[int]
) -> tuple[list[int], list[list[int]]]:
    """
    Returns the multiclass confusion matrix as (classes, M), where
    M[i][j] is the number of samples with true class classes[i]
    predicted as class classes[j].

    Arguments:
        y_true -- list of ground-truth labels
        y_pred -- list of predicted labels (same length as y_true)

    Returns:
        tuple(classes, M) -- sorted unique classes and the
                             n_classes x n_classes matrix

    Raises:
        ValueError -- if the lists differ in length or are empty
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Label lists must not be empty")

    classes = sorted(set(y_true) | set(y_pred))
    idx = {c: i for i, c in enumerate(classes)}
    M = [[0] * len(classes) for _ in classes]
    for t, p in zip(y_true, y_pred):
        M[idx[t]][idx[p]] += 1
    return classes, M
