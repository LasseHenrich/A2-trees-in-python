import matplotlib.pyplot as plt

from ._utils import _save_or_show


def plot_confusion_matrix(
    M: list[list[int]],
    classes: list[int],
    title: str = "Confusion matrix",
    normalize: bool = False,
    save_path: str | None = None,
) -> None:
    """Plot a confusion matrix as a labelled heatmap.

    Rows are the true classes, columns are the predicted classes.
    Cell values are the integer counts; if `normalize` is True, each
    row is divided by its sum and the cells show fractions in [0, 1].

    Arguments:
        M         -- n_classes x n_classes confusion matrix from
                     `confusion_matrix`
        classes   -- list of class labels in the same order as the rows
                     and columns of M
        title     -- figure title
        normalize -- if True, each row is divided by its sum
        save_path -- path to write the figure to; if None, shows it
    """
    n = len(classes)
    if normalize:
        cells: list[list[float]] = []
        for row in M:
            total = sum(row)
            if total == 0:
                cells.append([0.0] * n)
            else:
                cells.append([v / total for v in row])
        fmt = "{:.2f}"
    else:
        cells = [[float(v) for v in row] for row in M]
        fmt = "{:.0f}"

    fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.2 * n + 2))
    im = ax.imshow(cells, cmap="Blues", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    cmax = max((max(row) for row in cells), default=0.0)
    threshold = cmax / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if cells[i][j] > threshold else "black"
            ax.text(
                j,
                i,
                fmt.format(cells[i][j]),
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    fig.tight_layout()
    _save_or_show(fig, save_path)
