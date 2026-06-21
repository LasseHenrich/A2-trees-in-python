# Task E — Cross-Validation

A single train/test split gives you one estimate of accuracy. That estimate
is noisy: a different random split might give a noticeably different number,
especially on smaller datasets.

**K-fold cross-validation** gives a more reliable estimate by repeating the
evaluation k times. The training data is divided into k equal-sized chunks
called **folds**. In each round, one fold is held out as the validation set
and the model is trained on the remaining k−1 folds. This produces k
accuracy estimates. Their mean is more stable than any single estimate, and
their spread tells you how sensitive the model is to which data it sees.

```
k = 4 folds, 12 samples:

  round 1: [ val val val | train train train | train train train | train train train ]
  round 2: [ train train train | val val val | train train train | train train train ]
  round 3: [ train train train | train train train | val val val | train train train ]
  round 4: [ train train train | train train train | train train train | val val val ]
```

Every sample appears in exactly one validation set across all k rounds.

## Why not use the test set?

The test set is reserved for the final evaluation of your chosen model. Using
it during hyperparameter search — even indirectly — risks over-fitting to the
test set: you would be choosing the configuration that happened to do well on
*that specific* set of samples, not one that generalises. Cross-validation
lets you tune on the training data and keep the test set completely unseen
until the end.

## What to implement

In `src/evaluation/cross_validation.py`, implement `k_fold_indices` and
`aggregate`.

### `k_fold_indices`

Takes the number of samples `n`, the number of folds `k`, and an optional
random seed. Shuffles the indices `[0, ..., n-1]` once and splits them into
k contiguous chunks of as-equal size as possible. When `n` is not divisible
by `k`, the first `n % k` folds receive one extra sample. For fold `i`,
chunk `i` is the validation set and the remaining chunks form the training
set.

Use a local `random.Random(seed)` instance rather than the global `random`
module, so that calls to `k_fold_indices` do not affect any other random
state in the program.

The function raises `ValueError` if `k < 2` or `k > n`.

### `aggregate`

`cross_validate` is already implemented in `src/evaluation/cross_validation.py`.
It takes `(X, y, grid, k, seed)`, splits the data using `k_fold_indices`,
runs `grid_search` on each fold's train/val split, and returns a list of k
lists — one inner list per fold, each containing one result dict per grid
configuration. `aggregate` collects those scattered results and produces one
summary dict per configuration.

Each result dict has these keys:

- `"criterion"`, `"stop_depth"`, `"stop_below"` — the configuration
- `"accuracy"` — the validation accuracy for this fold
- `"confusion_matrix"` — 2-D list of ints where `M[i][j]` is the number
  of samples with true class `i` predicted as class `j`
- `"avg_pred_depth"` — average depth of leaf nodes reached during
  prediction

Steps:

1. **Group results by configuration.** Use a dict keyed by the tuple
   `(criterion, stop_depth, stop_below)`. Loop over every fold and every
   result in that fold, appending each result dict to the list for its key.

2. **For each group, compute:**
   - `fold_accuracies` — the list of `"accuracy"` values, one per fold
   - `mean_accuracy` — mean of those values
   - `std_accuracy` — population standard deviation (divide by n, not n−1)
   - `avg_pred_depth` — mean of `"avg_pred_depth"` across folds
   - `confusion_matrix` — element-wise sum of `"confusion_matrix"` across
     all folds

3. **Return** a list of summary dicts, one per configuration, each with
   keys: `"criterion"`, `"stop_depth"`, `"stop_below"`,
   `"fold_accuracies"`, `"mean_accuracy"`, `"std_accuracy"`,
   `"confusion_matrix"`, `"avg_pred_depth"`.

## How to check your work

Create a `submission.json` file by running

```bash
make submit_e
```

and drop it in the course webpage for this assignment
(https://py.ml.tu-berlin.de/quiz/#/homework).

Or run the tests for this task:

```bash
make test-e
```

Please note that the test code provided is minimal, and only checks
that the functions can be called without error. There is boilerplate
code that you are welcome to fill out to your own liking.
