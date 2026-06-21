"""
K-Fold Cross-Validation

Task E: Implement `k_fold_indices`.
        `cross_validate` is provided.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import random

from .tuning import grid_search


def k_fold_indices(
    n: int, k: int, seed: int | None = None
) -> list[tuple[list[int], list[int]]]:
    """
    Returns k (train_indices, val_indices) pairs for k-fold cross-validation.

    Shuffles indices [0, ..., n-1] once with a local `random.Random(seed)`
    instance, then splits them into k contiguous chunks of as-equal size as
    possible. For fold i, chunk i is the validation set and the remaining
    chunks form the training set. When n is not divisible by k, the first
    n % k folds receive one extra sample.

    Arguments:
        n    -- number of samples
        k    -- number of folds (must satisfy 2 <= k <= n)
        seed -- optional int; fixes the shuffle for reproducibility

    Returns:
        list of k tuples (train_indices, val_indices); val_indices across
        folds form a partition of [0, n)

    Raises:
        ValueError -- if k < 2 or k > n
    """

    # ------ WRITE YOUR CODE HERE ------
    pass


def aggregate(fold_results: list[list[dict]]) -> list[dict]:
    """
    Combines per-fold results into one summary dict per configuration.

    Arguments:
        fold_results -- list of k lists returned by cross_validate; each
                        inner list contains one result dict per grid
                        configuration

    Each result dict has keys: "criterion", "stop_depth", "stop_below",
    "accuracy", "confusion_matrix", "avg_pred_depth".

    For each unique (criterion, stop_depth, stop_below) combination,
    collect the per-fold accuracy values and compute their mean and
    population standard deviation, sum confusion matrices across folds,
    and average avg_pred_depth.

    Returns a list of dicts, one per configuration, each with keys:
        "criterion", "stop_depth", "stop_below",
        "fold_accuracies", "mean_accuracy", "std_accuracy",
        "confusion_matrix", "avg_pred_depth"
    """
    # ------ WRITE YOUR CODE HERE ------
    pass


def cross_validate(
    X: list[list],
    y: list,
    grid: dict[str, list],
    k: int = 5,
    seed: int | None = None,
) -> list[list[dict]]:
    """
    Runs grid_search on each fold of a k-fold split. Provided.

    Arguments:
        X    -- feature vectors
        y    -- labels (same length as X)
        grid -- dict with keys "criterion", "stop_depth", "stop_below"
        k    -- number of folds (default 5)
        seed -- optional int; fixes the fold shuffle

    Returns:
        A list of k lists, one per fold. Each inner list is the return
        value of grid_search for that fold's train/val split.
    """
    folds = k_fold_indices(len(X), k=k, seed=seed)
    results = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]
        results.append(grid_search(X_train, y_train, X_val, y_val, grid))
    return results
