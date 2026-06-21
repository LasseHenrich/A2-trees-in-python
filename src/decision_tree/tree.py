"""
Decision Tree Classifier

The Node dataclass and DecisionTreeClassifier are provided. Implement
fit, predict, and _grow.

More on the CART algorithm this loosely follows:

    https://en.wikipedia.org/wiki/Decision_tree_learning
"""

from collections.abc import Callable
from dataclasses import dataclass

from .impurity import entropy, gini
from .split import best_split


@dataclass
class Node:
    """
    A single node of the decision tree.

    Internal nodes store the split used and references to their children.
    Leaves store only a prediction. Every node stores n_samples, the
    number of training samples that reached it during fit.

    Numeric split  : feature <= threshold  (threshold is set, category_value
                     is None)
    Categorical split: feature == category_value  (category_value is set,
                     threshold is None)
    """

    prediction: int | None = None
    n_samples: int = 0
    feature: int | None = None
    threshold: float | None = None
    category_value: str | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifier:
    """
    A classification tree built by greedy, top-down information-gain
    maximization (CART-style).

    Hyperparameters:
        max_depth         -- maximum depth of the tree; None means grow
                             until another stopping condition fires
        min_samples_split -- a node with fewer than this many samples
                             becomes a leaf without searching for a split
        criterion         -- "gini" or "entropy"
        feature_types     -- list of "numeric" or "categorical" strings,
                             one per feature; None treats all as numeric
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        feature_types: list[str] | None = None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy', got {criterion!r}"
            )
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.feature_types = feature_types
        self._impurity: Callable[[list[int]], float] = (
            gini if criterion == "gini" else entropy
        )
        self.root: Node | None = None

    def fit(self, X: list[list], y: list[int]) -> "DecisionTreeClassifier":
        """
        Builds the tree by recursively splitting starting from the root.

        Raises ValueError if X or y is empty, or if their lengths differ.

        Arguments:
            X -- feature vectors to train on
            y -- class labels, one per row in X

        Returns:
            DecisionTreeClassifier -- self, to allow method chaining
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    def predict(
        self,
        X: list[tuple],
        stop_depth: int | None = None,
        stop_below: int | None = None,
    ) -> list[int]:
        """
        Predicts a class label for each sample in X.

        Raises RuntimeError if the tree has not been fitted yet.

        Arguments:
            X          -- feature vectors to classify
            stop_depth -- if given, stops traversal at this depth and
                          uses the prediction stored at that node.
                          Useful for simulating a tree trained with a
                          smaller max_depth without retraining.
            stop_below -- if given, stops traversal at any node whose
                          n_samples is less than this value. Useful for
                          simulating a tree trained with a larger
                          min_samples_split without retraining.

        Returns:
            list[int] -- predicted class label for each sample
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    def predict_with_depth(
        self,
        X: list[tuple],
        stop_depth: int | None = None,
        stop_below: int | None = None,
    ) -> list[tuple[int, int]]:
        """
        Predicts a class label and traversal depth for each sample in X.

        Returns a list of (prediction, depth) pairs, where depth is the
        number of edges traversed from the root to the decision node.

        Accepts the same stop_depth and stop_below arguments as predict.
        """
        # ------ WRITE YOUR CODE HERE ------
        pass

    @staticmethod
    def _predict_one(
        x: tuple,
        node: Node,
        stop_depth: int | None = None,
        stop_below: int | None = None,
    ) -> tuple[int, int]:
        """
        Walks one sample down the tree and returns its prediction and
        the depth of the node where traversal stopped.

        Arguments:
            x          -- a single feature vector
            node       -- the node to start traversal from
            stop_depth -- stop at this depth if given; see predict
            stop_below -- stop at nodes with fewer samples if given;
                          see predict

        Returns:
            tuple[int, int] -- (predicted class label, traversal depth)
        """

        # ------ WRITE YOUR CODE HERE ------
        pass

    @staticmethod
    def _majority(y: list[int]) -> int:
        """
        Returns the most common label in y, breaking ties by smallest
        label.
        """
        counts: dict[int, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    def _grow(self, X: list[list], y: list[int], depth: int) -> Node:
        """
        Recursively builds a subtree and returns its root Node.

        Returns a leaf node (prediction = majority class) if any stopping
        condition holds: the node is pure, fewer than min_samples_split
        samples remain, max_depth is reached, or no split improves impurity.
        Otherwise splits on the best feature and recurses.

        For numeric features the split is feature <= threshold; for
        categorical features it is feature == category_value.

        Arguments:
            X     -- feature vectors at this node
            y     -- labels at this node
            depth -- depth of this node (root is 0)

        Returns:
            Node -- root of the subtree grown from (X, y)
        """

        # ------ WRITE YOUR CODE HERE ------
        pass
