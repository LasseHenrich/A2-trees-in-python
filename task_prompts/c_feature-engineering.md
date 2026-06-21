# Task C — Feature Engineering on the 2D Toy Datasets

A decision tree splits one feature at a time, and each split is
axis-aligned: every cut in feature space is a vertical or horizontal
line. This task is the first time you will *feel* that limitation,
and the first time you will fix it by **engineering new features**.

## Concepts

### Preprocessing

Preprocessing is any transformation applied to raw data before it is
fed to a model. Common forms include:

- **Normalization**: scale each feature to a fixed range, usually
  [0, 1], by dividing by the range of that feature.
- **Standardization**: shift and scale each feature so it has mean 0
  and standard deviation 1.
- **Feature engineering**: add new columns computed from existing
  ones.

All three change what the model sees without changing the model
itself. Today you will use feature engineering.

### Model bias

Every model can only express certain shapes. A decision tree can only
draw axis-aligned rectangles. **Bias** in this sense is the set of
shapes the model can produce.

When the true boundary in your data does not match the model's bias,
you usually pay for it with depth (more rectangles to approximate the
true shape) and with overfitting (the deep tree learns noise).

### Feature engineering

If the model cannot express the shape you need, **change the inputs
so it does not have to**. The model is unchanged. You add columns
computed from the original ones, and the model now only has to draw
axis-aligned splits in this new, friendlier space.

Two examples:

- **circles** has two concentric rings. The classes differ in their
  distance from the origin. Add `r = sqrt(x1**2 + x2**2)` as a
  third column — a single split on `r` separates them.
- **xor** labels samples by whether `x1` and `x2` have the same sign.
  No split on `x1` alone or `x2` alone reduces impurity. A split on
  `x1 * x2` (positive vs negative) does the whole job.

## Feature map configs

Feature engineering in this project is done by writing JSON config
files that describe which new columns to compute. The format is
defined in `src/data/feature_map.py`. A config looks like this:

```json
{
  "dataset": "circles",
  "output": "circles-processed",
  "columns": ["x1", "x2"],
  "steps": [
    { "output": "r", "fn": "radius", "inputs": ["x1", "x2"] }
  ],
  "select": ["x1", "x2", "r"]
}
```

Each step computes one new column from earlier columns using a
function named by `fn`. `select` lists which columns to keep in the
output. Original columns do not have to be included in `select` — if
a new feature is all the tree needs, you can drop the originals. If
`select` is omitted, only the step outputs are kept.

The available functions are listed in `REGISTRY` in
`src/data/feature_map.py`:

| key        | what it computes                        |
| ---------- | --------------------------------------- |
| `identity` | `x`                                     |
| `square`   | `x * x`                                 |
| `product`  | `x * y`                                 |
| `shift`    | `x + scalar` (kwarg `scalar`)           |
| `scale`    | `x * scalar` (kwarg `scalar`)           |
| `radius`   | `sqrt(x1**2 + x2**2 + ...)`             |
| `angle`    | `atan2(y, x)`                           |
| `rotate_x` | rotated x-coordinate (kwarg `angle`)    |
| `rotate_y` | rotated y-coordinate (kwarg `angle`)    |
| `sin`      | `sin(x)`                                |
| `cos`      | `cos(x)`                                |

Some functions take extra parameters beyond their inputs. Pass these
as additional keys in the step object. For example, `rotate_x` takes
an `angle` parameter (in degrees):

```json
{ "output": "x1_rot", "fn": "rotate_x", "inputs": ["x1", "x2"], "angle": 45 }
```

If you need a function that is not in the table, add it to `REGISTRY`
in `src/data/feature_map.py`.

The config `configs/preprocessing/circles-processed.json` is already
provided as a worked example.

## What you will do

### 1 — Run the worked example

The config `configs/preprocessing/circles-processed.json` used the circles dataset
to create a new dataset with only the distance of each point to the origin, or
rather, the radius. The command to run it is:

```bash
uv run python scripts/preprocess.py \
    configs/preprocessing/circles-processed.json
```

or you can use the command in the Makefile:

```bash
make preprocess-circles
```

This writes processed CSVs to `data/processed/circles-processed/` and
saves a side-by-side scatter plot to
`results/figures/datasets/circles-processed.png`. If you open the plot you can see
that the two classes separate cleanly on the `r` axis, so a single split is
enough.

### 2 — Fill in configs for the remaining datasets

In `configs/preprocessing` you'll find a lot of files that are named
`{dataset}-processed.json`. There are blank preprocessing configs for you to
fill out to the best of your ability.

It should be noted that we'll exclude the circles dataset, as we used it as the
demo.

Before filling in a config, look at the scatter plot of the raw dataset in
`results/figures/datasets/`. These were created when you generated the
datasets in Task B. Studying the shape of the data will help you decide which
feature transformation is likely to separate the classes cleanly.

For each of the five remaining skeleton configs, fill them out so that the
included transformed features can more easily be separated by a decision tree.
In some cases, the data will already be well separated and will not benefit
from feature engineering.

**Requirement:** `select` must contain exactly one or two features.
This keeps the decision boundary plots meaningful.

Hints:
- `xor`: the sign of `x1 * x2` separates the two classes with a
  single split. Use a `product` step.
- `moons`: the two crescents differ in their angle from the origin.
  The `angle` function computes `atan2(x2, x1)`.
- `checkerboard`: product features can expose the repeating sign
  pattern in the labels.
- `blobs`: you might not need to change any of the features. If this is the
  case, insert the column labels in the `"select"` field.
- Due to nose in the datasets, there will almost always have some overlap, but
  you can get really close.


Once you're done filling out a preprocessing config, you can create the dataset
by running:

```bash
uv run python scripts/preprocess.py \
    configs/preprocessing/<dataset>-processed.json
```

or by running the shortcut:

```bash
make preprocess-<dataset>
```

If you want to preprocess all of the 2D datasets, you can run:

```bash
make preprocess-all
```

### 3 — Fit trees on the proposed datasets

You've done this before! In the last task. So don't fret. To fit a tree, run:

```bash
uv run python scripts/fit_tree.py data/processed/<dataset>-processed
```

or the directive in the Makefile:

```bash
make fit-<dataset>-processed
```

but realistically just run:

```bash
make fit-all-processed
```

### 4 — Plot the decision boundaries

```bash
uv run python scripts/plotting/plot_boundary_grid.py \
    moons-processed circles-processed blobs-processed \
    xor-processed checkerboard-processed linear-processed \
    --depths 1 2 3 5 7 None \
    --save results/figures/boundary_grid_2D_processed.png
```

```bash
make plot-boundary-grid-processed
```

### 5 — Read the depth-1 accuracy

Look at the depth-1 row in `results/figures/boundary_grid_2D_processed.png`.
Each subplot title shows the accuracy on the test set at that depth.

For every dataset, the depth-1 accuracy should be at least 0.90. If a
dataset falls short, revisit its preprocessing config. The `select`
field controls which features reach the tree — if you include a feature
the tree cannot split cleanly on, the useful one is buried.

The one exception is blobs. Blobs is already separated well enough that
a single axis-aligned split cannot cleanly partition all clusters at
once, so a depth-1 accuracy above 0.65 is sufficient. At depth 2,
however, the tree should reach 1.00.

Now open `results/figures/boundary_grid_2D.png` from Task B. For moons
and checkerboard, the depth-1 row of the raw grid will show a noticeably
lower accuracy. At which depth in the raw grid does each dataset first
reach or surpass the same accuracy as its processed version does at depth 1?

The plot in step 4 only shows a selection of depths. If you want a
precise answer, re-run `plot_boundary_grid.py` on the raw datasets with
a finer range, for example `--depths 1 2 3 4 5`:

```bash
uv run python scripts/plotting/plot_boundary_grid.py \
    moons checkerboard \
    --depths 1 2 3 4 5 \
    --save results/figures/boundary_grid_raw_fine.png
```

```bash
make plot-boundary-grid-raw-fine
```

The number of extra splits required is the cost of a poor feature
representation.

> **Record your answers** in `answers.py`:
> `c_moons_raw_min_depth` and `c_checkerboard_raw_min_depth`.
