import json

import pytest

from data.feature_map import (
    _apply_steps,
    load_feature_map,
    make_downsample_img,
)


def test_apply_steps_single_derived_feature():
    row = [3.0, 4.0]
    columns = ["x1", "x2"]
    steps = [{"output": "r", "fn": "radius", "inputs": ["x1", "x2"]}]
    select = ["x1", "x2", "r"]
    assert _apply_steps(row, columns, steps, select) == [3.0, 4.0, 5.0]


def test_apply_steps_composition():
    row = [3.0, 4.0]
    columns = ["x1", "x2"]
    steps = [
        {"output": "r", "fn": "radius", "inputs": ["x1", "x2"]},
        {"output": "r_sq", "fn": "square", "inputs": ["r"]},
    ]
    select = ["r", "r_sq"]
    assert _apply_steps(row, columns, steps, select) == [5.0, 25.0]


def test_apply_steps_kwargs():
    row = [1.0, 0.0]
    columns = ["x1", "x2"]
    steps = [
        {
            "output": "rx",
            "fn": "rotate_x",
            "inputs": ["x1", "x2"],
            "angle": 90,
        }
    ]
    result = _apply_steps(row, columns, steps, ["rx"])
    assert result[0] == pytest.approx(0.0, abs=1e-10)



def test_make_downsample_img_smoke():
    fn = make_downsample_img(4, 2)
    assert callable(fn)

def test_make_downsample_img():
    pass


def test_load_feature_map(tmp_path):
    cfg = {
        "dataset": "circles",
        "output": "circles_radius",
        "columns": ["x1", "x2"],
        "steps": [{"output": "r", "fn": "radius", "inputs": ["x1", "x2"]}],
        "select": ["r"],
    }
    path = tmp_path / "fm.json"
    path.write_text(json.dumps(cfg))
    transform = load_feature_map(str(path))
    X_new, ft = transform([[3.0, 4.0], [0.0, 0.0]], ["numeric", "numeric"])
    assert X_new == [[5.0], [0.0]]
    assert ft == ["numeric"]
