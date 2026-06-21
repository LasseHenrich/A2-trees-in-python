"""
Named, composable feature definitions for 2D datasets.

A feature map config (JSON) defines a new dataset as a sequence of
named steps. Each step computes one value from earlier names in the
working namespace (original column values or prior step outputs).

Config format::

    {
      "dataset":  "<source dataset name>",
      "output":   "<output dataset name>",
      "columns":  ["x1", "x2"],
      "steps": [
        {
          "output": "<name>",
          "fn":     "<registry key>",
          "inputs": ["<name>", ...],
          "<kwarg>": <value>
        }
      ],
      "select":   ["<name>", ...]
    }

``select`` is optional; if omitted, all step outputs are included in
order. Original column names are available in ``select`` and as inputs
to any step. Extra keys in a step (beyond ``output``, ``fn``,
``inputs``) are forwarded to the function as keyword arguments. For
example, ``"angle": 45`` in a step using ``rotate_x`` passes
``angle=45`` to the function.

``REGISTRY`` maps function names to callables. Add entries to make new
functions available in configs.
"""

import json
import math
from collections.abc import Callable


def _shift(x: float, *, scalar: float) -> float:
    return x + scalar


def _scale(x: float, *, scalar: float) -> float:
    return x * scalar


def _radius(*args: float) -> float:
    return float(sum(x**2 for x in args) ** 0.5)


def _angle(x: float, y: float) -> float:
    return math.atan2(y, x)


def _rotate_x(x: float, y: float, *, angle: float) -> float:
    a = math.radians(angle)
    return x * math.cos(a) - y * math.sin(a)


def _rotate_y(x: float, y: float, *, angle: float) -> float:
    a = math.radians(angle)
    return x * math.sin(a) + y * math.cos(a)


REGISTRY: dict[str, Callable] = {
    "identity": lambda x: x,
    "square": lambda x: x * x,
    "product": lambda x, y: x * y,
    "shift": _shift,
    "scale": _scale,
    "radius": _radius,
    "angle": _angle,
    "rotate_x": _rotate_x,
    "rotate_y": _rotate_y,
    "sin": math.sin,
    "cos": math.cos,
}


def _apply_steps(
    row: list,
    columns: list[str],
    steps: list[dict],
    select: list[str],
) -> list[float]:
    ns: dict[str, float] = {name: row[i] for i, name in enumerate(columns)}
    for step in steps:
        fn = REGISTRY[step["fn"]]
        args = [ns[name] for name in step["inputs"]]
        kwargs = {
            k: v
            for k, v in step.items()
            if k not in {"output", "fn", "inputs"}
        }
        ns[step["output"]] = fn(*args, **kwargs)
    return [ns[name] for name in select]


def load_feature_map(path: str):
    """
    Loads a feature map JSON config and returns a callable transform.

    Arguments:
        path -- path to a feature map JSON config file

    Returns:
        callable (X, feature_types) -> (X_new, feature_types_new)
    """
    with open(path) as f:
        cfg = json.load(f)

    columns: list[str] = cfg["columns"]
    steps: list[dict] = cfg["steps"]
    select: list[str] = cfg.get("select", [s["output"] for s in steps])

    def transform(
        X: list[list], feature_types: list[str]
    ) -> tuple[list[list], list[str]]:
        X_new = [_apply_steps(row, columns, steps, select) for row in X]
        return X_new, ["numeric"] * len(select)

    return transform


def make_downsample_img(width: int, factor: int) -> Callable:
    """
    Returns a transform that average-pools a flat square-image row.

    Given a row of width * width values stored row-major, replaces each
    non-overlapping factor x factor block with its mean. The output has
    (width // factor) ** 2 values.

    Arguments:
        width  -- side length of the square image in pixels
        factor -- pooling block size

    Returns:
        callable (row: list[float]) -> list[float]
    """
    # ------ WRITE YOUR CODE HERE ------
    pass
