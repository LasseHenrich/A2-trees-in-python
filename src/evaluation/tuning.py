"""
Hyperparameter Search

Task D: Implement `stopping_search` and `grid_search`.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

from itertools import product
import statistics

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
    
    X_tuples = [tuple(feature_as_list) for feature_as_list in X]
    
    stop_depths_unique = list(dict.fromkeys(stop_depths))
    stop_belows_unique = list(dict.fromkeys(stop_belows))
    
    dicts = []
    for stop_depth in stop_depths_unique:
        for stop_below in stop_belows_unique:
            pred_result = tree.predict_with_depth(X=X_tuples, stop_depth=stop_depth, stop_below=stop_below)
            
            predictions = [entry[0] for entry in pred_result]
            depths = [entry[1] for entry in pred_result]
            
            avg_pred_depth = statistics.mean(depths)
            acc = accuracy(y_true=y, y_pred=predictions)
            conf_mat_result = confusion_matrix(y_true=y, y_pred=predictions)
            conf_mat = conf_mat_result[1]
            
            dicts.append({
                "stop_depth": stop_depth,
                "stop_below": stop_below,
                "accuracy": acc,
                "confusion_matrix": conf_mat,
                "avg_pred_depth": avg_pred_depth
            })
            
    return dicts
    
    
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
    
    dicts = []    
    for criterion in grid["criterion"]:
        max_depth = max(grid["stop_depth"]) if None not in grid["stop_depth"] else None
        min_samples_split = min(grid["stop_below"]) if None not in grid["stop_below"] else None
        
        if min_samples_split is None:
            tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        else:
            tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)
            
        tree.fit(X_train, y_train)
        
        results = stopping_search(tree=tree, X=X_val, y=y_val, stop_depths=grid["stop_depth"], stop_belows=grid["stop_below"])
        
        for result in results:
            dicts.append({
                "criterion": criterion,
                "stop_depth": result["stop_depth"],
                "stop_below": result["stop_below"],
                "accuracy": result["accuracy"],
                "confusion_matrix": result["confusion_matrix"]
            })
            
    return dicts
