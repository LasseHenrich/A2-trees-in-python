"""
Impurity Measures and Information Gain

For a label set S with class proportions p_c:

    Gini(S)    = 1 - sum_c p_c^2
    Entropy(S) = - sum_c p_c * log2(p_c)

A split partitions S into left subset S_L and right subset S_R.
The information gain of that split is:

    IG(S, S_L, S_R) = I(S) - (|S_L| / |S|) * I(S_L) - (|S_R| / |S|) * I(S_R)

where I is whichever impurity measure was passed in.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import math
from collections.abc import Callable
from collections import Counter


def gini(y: list[int]) -> float:
    """
    Computes the Gini impurity of a list of labels.

    By convention, returns 0.0 for an empty list.

    Arguments:
        y -- a list of integer class labels

    Returns:
        float -- Gini impurity in [0, 1)
    """

    if len(y) == 0:
        return 0.0
    
    class_counts = Counter(y)
    p = [el/len(y) for el in class_counts.values()]
    
    # Gini(S)    = 1 - sum_c p_c^2
    return 1.0 - sum([el ** 2 for el in p])


def entropy(y: list[int]) -> float:
    """
    Computes the Shannon entropy (in bits) of a list of labels.

    Terms with p_c == 0 are excluded from the sum. Returns 0.0 for an
    empty list.

    Arguments:
        y -- a list of integer class labels

    Returns:
        float -- entropy in bits, in [0, log2(num_classes)]
    """

    if len(y) == 0:
        return 0.0
    
    class_counts = Counter(y)
    p = [el/len(y) for el in class_counts.values()]
    
    # Entropy(S) = - sum_c p_c * log2(p_c)
    return -sum([el * math.log(el, 2) for el in p])


def information_gain(
    y: list[int],
    y_left: list[int],
    y_right: list[int],
    criterion: Callable[[list[int]], float],
) -> float:
    """
    Computes the information gain of a split using the given
    impurity criterion.

    Arguments:
        y         -- parent label list (before the split)
        y_left    -- labels in the left child
        y_right   -- labels in the right child
        criterion -- impurity callable, e.g. `gini` or `entropy`

    Returns:
        float -- information gain (>= 0)

    Raises:
        ValueError -- if y is empty
        ValueError -- if len(y_left) + len(y_right) != len(y)
    """
    
    if len(y) == 0:
        raise ValueError("y cannot be empty")
    
    if len(y_left) + len(y_right) != len(y):
        raise ValueError("y must be decomposed into y_left and y_right")


    # IG(S, S_L, S_R) = I(S) - (|S_L| / |S|) * I(S_L) - (|S_R| / |S|) * I(S_R)
    
    y_imp = criterion(y)
    left = (len(y_left) / len(y)) * criterion(y_left)
    right = (len(y_right) / len(y)) * criterion(y_right)
    return y_imp - left - right
