import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_POINT_COLORS = ["#1d4ed8", "#b91c1c", "#15803d", "#a21caf", "#b45309"]
_FILL_COLORS = ["#3b82f6", "#ef4444", "#4ade80", "#d946ef", "#fbbf24"]


def _scatter(
    ax: Axes,
    X: list[list[float]],
    y: list[int],
    marker: str = "o",
    label_suffix: str = "",
    classes: list[int] | None = None,
) -> None:
    if classes is None:
        classes = sorted(set(y))
    x1 = [row[0] for row in X]
    x2 = [row[1] for row in X]
    for i, cls in enumerate(classes):
        pts_x = [a for a, lbl in zip(x1, y) if lbl == cls]
        pts_y = [b for b, lbl in zip(x2, y) if lbl == cls]
        ax.scatter(
            pts_x,
            pts_y,
            c=_POINT_COLORS[i % len(_POINT_COLORS)],
            marker=marker,
            s=18,
            label=f"class {cls}{label_suffix}",
            edgecolors="white",
            linewidth=0.4,
        )


def _save_or_show(fig: Figure, save_path: str | None) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)
