import random
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ._utils import _scatter, _save_or_show, _POINT_COLORS


def plot_dataset(
    ax: Axes,
    X: list[list[float]],
    y: list[int],
    title: str = "2D Dataset",
    xlabel: str = "x1",
    ylabel: str = "x2",
    marker: str = "o",
    legend: bool = True,
) -> None:
    """Scatter-plots a 2D dataset coloured by class label on ax."""
    _scatter(ax, X, y, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend(loc="best")


def plot_jitter(
    ax: Axes,
    X_train: list[list[float]],
    y_train: list[int],
    X_test: list[list[float]] | None = None,
    y_test: list[int] | None = None,
    title: str = "",
    xlabel: str = "x1",
    seed: int = 0,
) -> None:
    """Jitter plot for 1D data with per-class, per-split bands on the y-axis.

    Bands are stacked vertically: test bands at the bottom, train bands
    above. Each band's height is proportional to its share of total samples.
    Points are scattered horizontally by feature value and jittered
    vertically within their band.
    """
    rng = random.Random(seed)
    x_te: list[float] = [row[0] for row in X_test] if X_test else []
    lbl_te: list[int] = y_test if y_test is not None else []
    classes = sorted(set(y_train) | set(lbl_te))
    n_total = len(y_train) + len(lbl_te)
    x_tr = [row[0] for row in X_train]

    tick_positions: list[float] = []
    tick_labels: list[str] = []
    y = 0.0

    if x_te:
        for i, cls in enumerate(classes):
            pts = [x for x, lbl in zip(x_te, lbl_te) if lbl == cls]
            h = len(pts) / n_total
            ax.axhline(y, color="lightgray", linewidth=0.5, zorder=0)
            jitter = [rng.uniform(y, y + h) for _ in pts]
            ax.scatter(
                pts,
                jitter,
                c=_POINT_COLORS[i % len(_POINT_COLORS)],
                marker="^",
                s=18,
                edgecolors="white",
                linewidth=0.4,
            )
            tick_positions.append(y + h / 2)
            tick_labels.append(f"test {cls}")
            y += h
        ax.axhline(y, color="gray", linewidth=1.5, zorder=1)

    for i, cls in enumerate(classes):
        pts = [x for x, lbl in zip(x_tr, y_train) if lbl == cls]
        h = len(pts) / n_total
        jitter = [rng.uniform(y, y + h) for _ in pts]
        ax.scatter(
            pts,
            jitter,
            c=_POINT_COLORS[i % len(_POINT_COLORS)],
            marker="o",
            s=18,
            edgecolors="white",
            linewidth=0.4,
        )
        tick_positions.append(y + h / 2)
        tick_labels.append(f"train {cls}")
        y += h
        ax.axhline(y, color="lightgray", linewidth=0.5, zorder=0)

    ax.set_ylim(0, 1)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)


def plot_train_test(
    name: str,
    X_train: list[list[float]],
    y_train: list[int],
    X_test: list[list[float]],
    y_test: list[int],
    save_path: str | None = None,
) -> None:
    """Scatter-plots train (circles) and test (triangles) on one axes."""
    fig, ax = plt.subplots(figsize=(6, 5))
    classes = sorted(set(y_train) | set(y_test))
    _scatter(
        ax,
        X_train,
        y_train,
        marker="o",
        label_suffix=" (train)",
        classes=classes,
    )
    _scatter(
        ax, X_test, y_test, marker="^", label_suffix=" (test)", classes=classes
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(name)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_original_and_processed(
    name: str,
    X: list[list[float]],
    y: list[int],
    X_proc: list[list[float]],
    y_proc: list[int],
    X_test: list[list[float]] | None = None,
    y_test: list[int] | None = None,
    X_proc_test: list[list[float]] | None = None,
    feature_names: list[str] | None = None,
    proc_feature_names: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot the original 2D dataset alongside its processed form.

    The number of subplots depends on the dimensionality of X_proc:
    - 1D: two subplots (original scatter, processed jitter).
          X_test, y_test, and X_proc_test are required for this case.
    - 2D: two subplots (original scatter, processed scatter)
    """
    n_features = len(X_proc[0])
    if n_features not in (1, 2):
        raise ValueError(f"Unsupported processed dimensionality: {n_features}")

    fn = feature_names or ["x1", "x2"]
    pfn = proc_feature_names or [f"x{i + 1}" for i in range(n_features)]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4))
    plot_dataset(
        axes[0],
        X,
        y,
        f"{name} (original)",
        xlabel=fn[0],
        ylabel=fn[1],
        marker="o",
        legend=False,
    )
    plot_dataset(
        axes[0],
        X,
        y,
        f"{name} (original)",
        xlabel=fn[0],
        ylabel=fn[1],
        marker="^",
        legend=False,
    )

    if n_features == 1:
        plot_jitter(
            axes[1],
            X_proc,
            y_proc,
            X_proc_test,
            y_test,
            title=f"{name} (processed)",
            xlabel=pfn[0],
        )
    else:
        plot_dataset(
            axes[1],
            X_proc,
            y_proc,
            f"{name} (processed)",
            xlabel=pfn[0],
            ylabel=pfn[1],
            marker="o",
        )
        plot_dataset(
            axes[1],
            X_proc_test,
            y_test,
            f"{name} (processed)",
            xlabel=pfn[0],
            ylabel=pfn[1],
            marker="^",
        )
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_datasets(
    datasets: list[tuple[list[list[float]], list[int]]],
    titles: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot multiple datasets side-by-side, one subplot per dataset."""
    n = len(datasets)
    if titles is None:
        titles = [f"Dataset {i + 1}" for i in range(n)]
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (X, y), title in zip(axes, datasets, titles):
        plot_dataset(ax, X, y, title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
