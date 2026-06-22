"""
Finding the Best Split

`candidate_thresholds` is provided. Implement `best_split`, which performs
an exhaustive search over every feature and candidate threshold, returning
the split with the highest information gain.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

from collections.abc import Callable
from functools import reduce
import sys

from .impurity import information_gain


def candidate_thresholds(values: list[float]) -> list[float]:
    """
    Returns candidate split thresholds for a single feature column.

    Candidates are the midpoints between every pair of consecutive unique
    sorted values. Any threshold within a gap produces the same partition,
    so the midpoint is a canonical representative.

    Arguments:
        values -- feature values from one column

    Returns:
        list[float] -- sorted candidate thresholds; empty if all values
                       are identical
    """
    unique_sorted = sorted(set(values))
    return [(a + b) / 2 for a, b in zip(unique_sorted, unique_sorted[1:])]


def best_split(
    X: list[list],
    y: list[int],
    criterion: Callable[[list[int]], float],
    feature_types: list[str] | None = None,
) -> tuple[int, float | str, float] | None:
    """
    Finds the (feature, split_value) pair that maximizes information gain.

    For numeric features, searches every candidate threshold from
    `candidate_thresholds`; samples with feature value <= threshold go
    left, the rest go right.

    For categorical features, each unique value is a candidate; samples
    where feature == value go left, the rest go right.

    Arguments:
        X             -- feature vectors (n_samples x n_features)
        y             -- labels (length n_samples)
        criterion     -- impurity callable, e.g. `gini` or `entropy`
        feature_types -- per-feature type strings, either 'numeric' or
                         'categorical'; None treats all as numeric

    Returns:
        tuple(feature_index, split_value, gain) -- the best split found,
            where split_value is a float threshold for numeric features
            and a string category value for categorical features
        None -- if no split produces strictly positive gain
    """
    
    if not X or not y:
        return None    
    feat_count = len(X[0])
    
    if feature_types is not None and len(feature_types) != feat_count:
        raise ValueError("feature_types length incorrect")

    best_gain = 0
    curr_best_split = None
    
    
    for feat_idx in range(feat_count):
        feat_type = feature_types[feat_idx] if feature_types is not None else "numeric"
        
        feat_column = [x[feat_idx] for x in X]
        
        if feat_type == "numeric":
            thresholds = candidate_thresholds(feat_column)
            for threshold in thresholds:
                left = [label for label_idx, label in enumerate(y) if feat_column[label_idx] <= threshold]
                right = [label for label_idx, label in enumerate(y) if feat_column[label_idx] > threshold]
                gain = information_gain(y, left, right, criterion)
                if gain > best_gain:
                    best_gain = gain
                    curr_best_split = (feat_idx, threshold, gain)

        elif feat_type == "categorical":
            unique_vals = list(set(feat_column))
            for threshold in unique_vals:
                left = [label for label_idx, label in enumerate(y) if feat_column[label_idx] == threshold]
                right = [label for label_idx, label in enumerate(y) if feat_column[label_idx] != threshold]
                gain = information_gain(y, left, right, criterion)
                if gain > best_gain:
                    best_gain = gain
                    curr_best_split = (feat_idx, threshold, gain)
        else:
            raise ValueError("feature_types must be a list of either 'numeric' or 'categorical'")
    
    return curr_best_split
    
