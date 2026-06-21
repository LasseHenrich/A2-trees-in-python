# Task F — Writing the `evaluate.py` Script

`scripts/fit_tree.py` trains a tree and reports accuracy on the held-out
test set. In this task you will complete `scripts/evaluate.py`, a companion
script that uses cross-validation instead.

Cross-validation is the right tool when you want to compare training
configurations or search for good hyperparameters. `fit_tree.py` answers
*"how well does this fitted model do on the test set?"*. `evaluate.py`
answers *"how well will this configuration do on average, and how
consistent is it?"*

## argparse

`argparse` is a standard library module for parsing command-line arguments.
You define what arguments a script accepts, and `argparse` reads
`sys.argv`, validates the input, and returns the results as a
`Namespace` object whose attributes match the argument names.

```python
parser = argparse.ArgumentParser()
parser.add_argument("--max-depth", type=int, default=5)
args = parser.parse_args()
print(args.max_depth)  # value passed via --max-depth, or 5
```

Hyphens in flag names become underscores on the `Namespace`:
`--max-depth` is accessed as `args.max_depth`.

When a flag is declared with `nargs="+"`, it accepts one or more values and
stores them as a list:

```
python script.py --max-depth 3 5 10
# args.max_depth == [3, 5, 10]
```

`_build_parser` in `evaluate.py` is already written — read it to see how
the flags are declared, then use the resulting `args` object in `main`.

## What to implement

Three functions in `scripts/evaluate.py`:

- `_build_grid` — maps CLI arguments to the grid dict
- `_is_grid_search` — decides which print path to take
- `main` — ties everything together

The argument parser, print helpers (`_print_single`, `_print_grid`), and
the imports are all provided.

### `_build_grid`

Returns a `dict[str, list]` mapping grid keys to the corresponding argument
lists from `args`. The CLI attribute names differ slightly from the keys
expected by `cross_validate`:

| `args` attribute          | Grid key        |
|---------------------------|-----------------|
| `args.max_depth`          | `"stop_depth"`  |
| `args.criterion`          | `"criterion"`   |
| `args.min_samples_split`  | `"stop_below"`  |

`cross_validate` forms the Cartesian product of those lists internally —
you just need to pass them through.

### `_is_grid_search`

Returns `True` if any of the three argument lists has more than one
element, `False` otherwise. The result determines whether `main` calls
`_print_single` (detailed metrics for one configuration) or `_print_grid`
(ranked table of all combinations).

### `main`

`main` is the entry point for the script. In order:

1. Parse CLI arguments by calling `_build_parser().parse_args()`.
2. Resolve the dataset path. If `args.dataset` is not absolute, prepend
   `ROOT` (already defined at the top of the file).
3. Load `train.csv` from the dataset directory using `load_csv`.
4. Build the grid dict with `_build_grid(args)`.
5. Run `cross_validate(X, y, grid, k=args.folds, seed=args.seed)`.
6. Aggregate the fold results with `aggregate`.
7. Call `_print_grid` or `_print_single` depending on `_is_grid_search`.

## Script behaviour

The script accepts these flags:

| Flag                   | Default           | Stored as                |
|------------------------|-------------------|--------------------------|
| `--max-depth`          | `[None]`          | `args.max_depth`         |
| `--criterion`          | `[gini, entropy]` | `args.criterion`         |
| `--min-samples-split`  | `[None]`          | `args.min_samples_split` |
| `--folds`              | `5`               | `args.folds`             |
| `--seed`               | `0`               | `args.seed`              |

When any of `--max-depth`, `--criterion`, or `--min-samples-split` receives
more than one value, the script runs a grid search over every combination.
No separate flag is needed.

## Retrospective — feature engineering and minimum depth

Now that you have `evaluate.py`, you can measure what Task C showed
visually. Run a depth grid search on both the raw and processed moons
datasets:

```bash
uv run python scripts/evaluate.py data/raw/moons \
    --max-depth 1 2 3 4 5 --criterion gini --folds 5
```

```bash
make evaluate-moons-depth-fine
```

```bash
uv run python scripts/evaluate.py data/processed/moons-processed \
    --max-depth 1 2 3 4 5 --criterion gini --folds 5
```

```bash
make evaluate-moons-processed-depth-fine
```

In each table, find the smallest depth whose `mean_acc` is within one
`std_acc` of the best result. For which dataset is that threshold reached
at a lower depth?

The difference is the same gap Task C showed in the boundary plots, now
expressed as a number.

> **Record your answer** in `answers.py`: `f_retro_lower_depth_dataset`.

## The effect of k

Run the same single-configuration call three times, changing only
`--folds`:

```bash
uv run python scripts/evaluate.py data/raw/moons \
    --max-depth 5 --criterion gini --folds 2
```

```bash
make evaluate-moons-k2
```

```bash
uv run python scripts/evaluate.py data/raw/moons \
    --max-depth 5 --criterion gini --folds 5
```

```bash
make evaluate-moons-k5
```

```bash
uv run python scripts/evaluate.py data/raw/moons \
    --max-depth 5 --criterion gini --folds 10
```

```bash
make evaluate-moons-k10
```

Each run prints the per-fold accuracies and a `std` value. Notice two
things as k increases:

- **More estimates.** With k=2 you have two accuracy values; with k=10
  you have ten. A standard deviation computed from two numbers is not a
  reliable measure of spread — the two values may happen to be close or
  far apart by chance. You can see this directly: run the k=2 command
  several times with different `--seed` values (0, 1, 2, ...). The `std`
  will change drastically from run to run. Do the same with k=10 and the
  `std` stays much more consistent.
- **More training data per fold.** With k=2, each fold trains on half the
  data. With k=5, each fold trains on four-fifths. With k=10, each fold
  trains on nine-tenths. More training data per fold means each fold's
  accuracy estimate has lower bias.

The conventional choice of k=5 or k=10 balances these two effects. Very
small k (k=2) gives noisy estimates; very large k (k=n, leaving one sample
out per fold) is expensive and the folds become highly correlated with one
another.

## How to check your work

Single-configuration run:

```bash
make evaluate-moons
```

Grid search across four depth values:

```bash
make evaluate-moons-grid
```

Create a `submission.json` file by running

```bash
uv run python submit.py f
```

```bash
make submit_f
```

and drop it in the course webpage for this assignment
(https://py.ml.tu-berlin.de/quiz/#/homework).
