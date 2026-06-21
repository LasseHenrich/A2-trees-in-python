import random
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from evaluation.metrics import accuracy
from ._utils import _scatter, _save_or_show, _FILL_COLORS, _POINT_COLORS


def plot_decision_boundary(
    ax: Axes,
    model,
    X: list[list[float]],
    y: list[int],
    title: str = "Decision Boundary",
    resolution: int = 200,
    stop_depth: int | None = None,
    legend: bool = True,
) -> None:
    """Renders the decision boundary of a fitted 2D classifier on ax."""
    x1 = [row[0] for row in X]
    x2 = [row[1] for row in X]

    pad = 0.5
    x1_min, x1_max = min(x1) - pad, max(x1) + pad
    x2_min, x2_max = min(x2) - pad, max(x2) + pad

    step1 = (x1_max - x1_min) / resolution
    step2 = (x2_max - x2_min) / resolution
    grid_x1 = [x1_min + i * step1 for i in range(resolution + 1)]
    grid_x2 = [x2_min + i * step2 for i in range(resolution + 1)]

    mesh_points = [[a, b] for b in grid_x2 for a in grid_x1]
    mesh_preds = model.predict(mesh_points, stop_depth=stop_depth)

    nx, ny = len(grid_x1), len(grid_x2)
    Z = [mesh_preds[row * nx : (row + 1) * nx] for row in range(ny)]

    classes = sorted(set(y))
    levels = [c - 0.5 for c in classes] + [classes[-1] + 0.5]
    ax.contourf(
        grid_x1,
        grid_x2,
        Z,
        alpha=0.3,
        levels=levels,
        colors=_FILL_COLORS[: len(classes)],
    )

    _scatter(ax, X, y)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    if legend:
        ax.legend(loc="best")
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)


def plot_decision_boundary_1d(
    ax: Axes,
    model,
    X: list[list[float]],
    y: list[int],
    title: str = "Decision Boundary",
    resolution: int = 200,
    stop_depth: int | None = None,
    legend: bool = True,
    seed: int = 0,
) -> None:
    """Renders the decision boundary of a fitted 1D classifier on ax."""
    rng = random.Random(seed)
    x1 = [row[0] for row in X]
    classes = sorted(set(y))

    pad = 0.5
    x_min, x_max = min(x1) - pad, max(x1) + pad

    step = (x_max - x_min) / resolution
    grid = [[x_min + i * step] for i in range(resolution + 1)]
    preds = model.predict(grid, stop_depth=stop_depth)

    span_start = x_min
    span_cls = preds[0]
    for pt, pred in zip(grid[1:], preds[1:]):
        if pred != span_cls:
            ci = classes.index(span_cls)
            ax.axvspan(
                span_start,
                pt[0],
                alpha=0.3,
                color=_FILL_COLORS[ci % len(_FILL_COLORS)],
                zorder=0,
            )
            span_start, span_cls = pt[0], pred
    ci = classes.index(span_cls)
    ax.axvspan(
        span_start,
        x_max,
        alpha=0.3,
        color=_FILL_COLORS[ci % len(_FILL_COLORS)],
        zorder=0,
    )

    for i, cls in enumerate(classes):
        pts = [x for x, lbl in zip(x1, y) if lbl == cls]
        jitter = [rng.uniform(0.2, 0.8) for _ in pts]
        ax.scatter(
            pts,
            jitter,
            c=_POINT_COLORS[i % len(_POINT_COLORS)],
            marker="o",
            s=18,
            edgecolors="white",
            linewidth=0.4,
            label=f"class {cls}",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("x1")
    ax.set_title(title)
    if legend:
        ax.legend(loc="best")


def plot_boundary_grid(
    names: list[str],
    datasets: list[tuple[list[list[float]], list[int]]],
    classifiers: list,
    depths: list[int | None],
    criterion: str = "gini",
    resolution: int = 200,
    save_path: str | None = None,
) -> None:
    """Plot a grid of decision boundaries across datasets and depths.

    Rows correspond to depths and columns to datasets. Each subplot
    title states the accuracy on the provided data split. Dataset
    names label the top of each column and depth values label the
    left of each row.

    Arguments:
        names       -- dataset labels for each column
        datasets    -- parallel list of (X, y) data pairs
        classifiers -- parallel list of fitted DecisionTreeClassifiers
        depths      -- depth values to evaluate (row labels); None means
                       no depth limit
        criterion   -- impurity criterion used; shown as figure title
        resolution  -- mesh resolution for each boundary plot
        save_path   -- file path to save the figure; displays if None
    """
    sorted_depths = sorted(d for d in depths if d is not None)
    if None in depths:
        sorted_depths.append(None)

    n_rows = len(sorted_depths)
    n_cols = len(names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    for row, depth in enumerate(sorted_depths):
        depth_label = "no limit" if depth is None else str(depth)
        for col, (name, (X, y), clf) in enumerate(
            zip(names, datasets, classifiers)
        ):
            ax = axes[row][col]
            y_pred = clf.predict(X, stop_depth=depth)
            acc = accuracy(y, y_pred)
            plot_fn = (
                plot_decision_boundary
                if len(X[0]) == 2
                else plot_decision_boundary_1d
            )
            plot_fn(
                ax,
                clf,
                X,
                y,
                title=f"acc: {acc:.2f}",
                resolution=resolution,
                stop_depth=depth,
                legend=False,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"acc: {acc:.2f}", fontsize=24)
            if row == 0:
                ax.set_title(f"{name}\nacc: {acc:.2f}", fontsize=28)
            if col == 0:
                ax.set_ylabel(f"depth = {depth_label}", fontsize=28)
    fig.suptitle(f"Decision Boundary with {criterion} criterion", fontsize=32)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save_or_show(fig, save_path)
