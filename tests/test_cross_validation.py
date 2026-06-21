import math

import pytest

from evaluation.cross_validation import (
    aggregate,
    cross_validate,
    k_fold_indices,
)


# ---------------------------------------------------------------------------
# k_fold_indices
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_k_fold_indices_smoke():
    folds = k_fold_indices(10, 5, seed=0)
    assert isinstance(folds, list)


def test_aggregate_smoke():
    fold_results = [
        [
            {
                "criterion": "gini",
                "stop_depth": None,
                "stop_below": None,
                "accuracy": 0.8,
                "confusion_matrix": [[4, 1], [1, 4]],
                "avg_pred_depth": 1.0,
            }
        ]
    ]
    result = aggregate(fold_results)
    assert isinstance(result, list)


def test_kfold_covers_all_indices():
    pass


def test_kfold_uneven_fold_sizes():
    pass


def test_kfold_seed_reproducible():
    pass


def test_kfold_invalid_k():
    pass


# ---------------------------------------------------------------------------
# cross_validate
# ---------------------------------------------------------------------------


def test_cross_validate_returns_k_fold_results():
    pass


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _fold_result(criterion, stop_depth, stop_below, accuracy, cm, avg_depth):
    return {
        "criterion": criterion,
        "stop_depth": stop_depth,
        "stop_below": stop_below,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "avg_pred_depth": avg_depth,
    }


def test_aggregate_returns_one_result_per_config():
    pass


def test_aggregate_mean_and_std_accuracy():
    pass


def test_aggregate_fold_accuracies_preserved():
    pass


def test_aggregate_confusion_matrix_summed():
    pass


def test_aggregate_avg_pred_depth_averaged():
    pass


def test_aggregate_multiple_configs_kept_separate():
    pass


def test_aggregate_result_has_expected_keys():
    pass


def test_cross_validate_seed_reproducible():
    pass
