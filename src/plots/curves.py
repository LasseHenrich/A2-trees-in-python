import matplotlib.pyplot as plt

from evaluation.metrics import accuracy
from ._utils import _save_or_show


def plot_depth_accuracy(
    clf,
    X_train: list[list[float]],
    y_train: list[int],
    X_test: list[list[float]],
    y_test: list[int],
    depths: list[int | None],
    title: str = "Accuracy vs. depth",
    save_path: str | None = None,
) -> None:
    """Plot training and test accuracy as a function of prediction depth.

    Arguments:
        clf        -- a fitted DecisionTreeClassifier
        X_train    -- training feature vectors
        y_train    -- training labels
        X_test     -- test feature vectors
        y_test     -- test labels
        depths     -- stop_depth values to evaluate; None means no limit
        title      -- figure title
        save_path  -- path to save the figure; displays if None
    """
    int_depths = sorted(d for d in depths if d is not None)
    ordered: list[int | None] = int_depths + ([None] if None in depths else [])

    labels = [str(d) if d is not None else "full" for d in ordered]
    train_accs = []
    test_accs = []

    for depth in ordered:
        train_preds = clf.predict(X_train, stop_depth=depth)
        test_preds = clf.predict(X_test, stop_depth=depth)
        train_accs.append(accuracy(y_train, train_preds))
        test_accs.append(accuracy(y_test, test_preds))

    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, train_accs, marker="o", label="train")
    ax.plot(xs, test_accs, marker="o", label="test")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("max depth")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)
