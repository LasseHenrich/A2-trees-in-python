"""
Functions used to create synthetic datasets and to download already existing
datasets.
"""

import math
import random


def make_moons(
    n: int = 300, noise: float = 0.20, seed: int = 0
) -> tuple[list[tuple[float, float]], list[int]]:
    rng = random.Random(seed)
    n0 = n // 2
    n1 = n - n0
    X, y = [], []
    for i in range(n0):
        t = math.pi * i / max(1, n0 - 1)
        x1 = math.cos(t) + rng.gauss(0, noise)
        x2 = math.sin(t) + rng.gauss(0, noise)
        X.append((x1, x2))
        y.append(0)
    for i in range(n1):
        t = math.pi * i / max(1, n1 - 1)
        x1 = 1 - math.cos(t) + rng.gauss(0, noise)
        x2 = 0.5 - math.sin(t) + rng.gauss(0, noise)
        X.append((x1, x2))
        y.append(1)
    return X, y


def make_circles(
    n: int = 300, noise: float = 0.10, factor: float = 0.5, seed: int = 0
) -> tuple[list[tuple[float, float]], list[int]]:
    """Outer ring = class 0, inner ring = class 1 (scaled by factor)."""
    rng = random.Random(seed)
    n0 = n // 2
    n1 = n - n0
    X, y = [], []
    for i in range(n0):
        angle = 2 * math.pi * i / max(1, n0 - 1)
        X.append(
            (
                math.cos(angle) + rng.gauss(0, noise),
                math.sin(angle) + rng.gauss(0, noise),
            )
        )
        y.append(0)
    for i in range(n1):
        angle = 2 * math.pi * i / max(1, n1 - 1)
        X.append(
            (
                factor * math.cos(angle) + rng.gauss(0, noise),
                factor * math.sin(angle) + rng.gauss(0, noise),
            )
        )
        y.append(1)
    return X, y


def make_blobs(
    n: int = 300,
    centers: list[tuple[float, float]] | None = None,
    std: float = 0.5,
    seed: int = 0,
) -> tuple[list[tuple[float, float]], list[int]]:
    """Gaussian blobs at the given centers; label = center index."""
    if centers is None:
        centers = [(-2.0, -2.0), (2.0, -2.0), (0.0, 2.0)]
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n):
        label = i % len(centers)
        cx, cy = centers[label]
        X.append((cx + rng.gauss(0, std), cy + rng.gauss(0, std)))
        y.append(label)
    return X, y


def make_xor(
    n: int = 300, noise: float = 0.10, seed: int = 0
) -> tuple[list[tuple[float, float]], list[int]]:
    """Four-quadrant XOR: class 0 on the diagonal quadrants,
    class 1 on the off-diagonal."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x1 = rng.uniform(-1.0, 1.0)
        x2 = rng.uniform(-1.0, 1.0)
        label = 0 if (x1 > 0) == (x2 > 0) else 1
        X.append((x1 + rng.gauss(0, noise), x2 + rng.gauss(0, noise)))
        y.append(label)
    return X, y


def make_checkerboard(
    n: int = 300, grid: int = 3, noise: float = 0.05, seed: int = 0
) -> tuple[list[tuple[float, float]], list[int]]:
    """grid×grid checkerboard; label = (col + row) % 2."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x1 = rng.uniform(0.0, float(grid))
        x2 = rng.uniform(0.0, float(grid))
        label = (int(x1) + int(x2)) % 2
        X.append((x1 + rng.gauss(0, noise), x2 + rng.gauss(0, noise)))
        y.append(label)
    return X, y


def make_linear(
    n: int = 300, noise: float = 0.15, seed: int = 0
) -> tuple[list[tuple[float, float]], list[int]]:
    """Linearly separable on x2 = x1 with Gaussian label noise."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x1 = rng.uniform(-2.0, 2.0)
        x2 = rng.uniform(-2.0, 2.0)
        label = 0 if x2 > x1 else 1
        X.append((x1 + rng.gauss(0, noise), x2 + rng.gauss(0, noise)))
        y.append(label)
    return X, y
