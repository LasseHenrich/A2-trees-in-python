"""
Visualizations for MNIST and its downsampled variants.
"""

import matplotlib.pyplot as plt

from ._utils import _save_or_show


def plot_digit_samples(
    versions: list[tuple[str, int, list[list[float]]]],
    save_path: str | None = None,
) -> None:
    """
    Plots one example per digit for each image version in a grid.

    Each row shows one version (e.g. original, 14x14, 7x7).
    Each column shows one digit (0-9).

    Arguments:
        versions  -- list of (title, width, examples) where:
                     title    = row label (e.g. "Original 28x28")
                     width    = side length of the square image in pixels
                     examples = list of 10 pixel rows, one per digit 0-9,
                                in order
        save_path -- path to save the figure; displays it if None
    """
    n_versions = len(versions)
    n_digits = 10

    fig, axes = plt.subplots(
        n_versions,
        n_digits,
        figsize=(n_digits * 1.2, n_versions * 1.6),
    )

    if n_versions == 1:
        axes = [axes]

    for row_idx, (title, width, examples) in enumerate(versions):
        for col_idx, row in enumerate(examples):
            ax = axes[row_idx][col_idx]
            img = [row[r * width : (r + 1) * width] for r in range(width)]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(str(col_idx), fontsize=10)
        axes[row_idx][0].set_ylabel(
            title, fontsize=8, rotation=90, labelpad=4, va="center"
        )

    plt.tight_layout()
    _save_or_show(fig, save_path)
