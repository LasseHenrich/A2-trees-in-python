# Task B — The Decision Tree Classifier

In Task A you wrote the impurity helpers and `best_split`. Now you
will use them to build the tree itself.

## What you are building

A class that:

1. Takes a training set `(X, y)` and grows a tree by splitting nodes
   from the top down (`fit`).
2. Takes new feature vectors and walks each one down the tree to a
   leaf (`predict`).

The class lives in `src/decision_tree/tree.py` and is called
`DecisionTreeClassifier`. The `__init__` and the helper
`_predict_one` are provided. Your job is to implement `fit`, `predict`, and the
recursive worker they share, `_grow`.

## The `Node` dataclass

A `Node` is either a **leaf** (only with attributes `prediction` and `n_samples`) or an
**internal node** (additionally with a `feature`, a `split_value`, and two
children). The split value comes in two flavors:

- Numeric split: `feature <= threshold` sends the sample left.
- Categorical split: `feature == category_value` sends the sample left.

Exactly one of `threshold` or `category_value` is set on an internal
node. Both are `None` on a leaf. Every node also stores:

- `prediction` — the majority class of the samples that reached it.
- `n_samples` — the number of training samples that reached it.

## The `DecisionTreeClassifier` class

`DecisionTreeClassifier` is the main class you will be working with. It
is constructed with four parameters:

- `max_depth` — the maximum number of splits along any path from the
  root to a leaf. `None` means the tree grows until one of the other
  stopping conditions fires.
- `min_samples_split` — a node with fewer than this many samples becomes
  a leaf without searching for a split.
- `criterion` — the impurity measure used to score candidate splits.
  Either `"gini"` (default) or `"entropy"`.
- `feature_types` — a list of `"numeric"` or `"categorical"` strings,
  one per feature column. `None` treats all features as numeric.

Aside from setting these parameters, the `__init__` method also checks that
these passed vaules are within a valid range, and raises a `ValueError` if they
are not. We don't want a `"magic"` criterion after all (literally!).

Note that the `__init__` method doesn't take any data of grow the tree; that's
the job of the below method

### `fit`

This is the method that creates the structure of the decision tree. It takes
the feature vectors `X`,the lables `y`, and starts a recursive process of
growing itself in order to fit the data that it was given.

It also raises a `ValueError` if either `X` or `y` are empty, or if they are
not the same length.

Note that `fit` itself doesn't grow the tree, that's the job of the recursive
helper method `_grow`.

#### `_grow`

`_grow(X, y, depth)` returns the root of a subtree. It is recursive.
It returns a leaf (i.e. a `Node` with only a `prediction`) when **any** of
these stopping conditions hold:

- `y` has only one unique label (the node is pure)
- `len(y) < self.min_samples_split`
- `self.max_depth is not None` and `depth >= self.max_depth`
- `best_split` returns `None` (no split improves impurity)

Otherwise:

1. Call `best_split` to get `(feature, split_value, gain)`.
2. Partition `(X, y)` into a left half and a right half using the same
   rule that `best_split` used (numeric: `<=`, categorical: `==`).
3. Build a `Node` with `feature`, the right kind of split value, the
   majority class as `prediction`, and the two recursive children.

In both cases, the `prediction` attribute of the node is set using the
`_majority` static method to pick the majority class.

### `predict`, `predict_with_depth`, and their unusual behavior

`predict` accepts the required parameter `X`, which is a list of feature
vectors to classify, and two optional parameters that stop traversal early:

- `X` - A list of feature vectors to classify.
- `stop_depth` — stop at any node once `depth >= stop_depth`.
- `stop_below` — stop at any node once `n_samples < stop_below`.

Using a `stop_depth` or `stop_below` parameter during prediction are not
standard features of a decision tree. Normally you would
choose `max_depth` and `min_samples_split` before training and retrain
the tree whenever you want to try different values. We store `n_samples`
on every node and keep a `prediction` on every internal node so that we
can skip the retraining step entirely: one deep tree can answer queries
as if it had been trained with any shallower depth or larger minimum
sample count. This will let us explore many hyperparameter values
quickly during our experiments later in the assignment.

`predict` returns a list of integers, each being the prediction of the
corresponding feature vector in `X`.

`predict_with_depth` behaves exactly like `predict` however it returns a list
of tuples, each tuple corresponding to a feature vector in `X`. The first
element of the tuple is the prediction and the second element is the depth of the `Node` where the prediction was made. This will be used during evaluation.

Both `predict` and `predict_with_depth` will not directly traverse the tree.
Instead this is given to the helper method `_predict_one`.

#### `_predict_one`

`_predict_one` takes a single feature vector `x`, a `Node` object `node` to
start traversal from, and the `stop_depth` and `stop_below` parameters that
were passed to the prediction method that called `_predict_one`. Lastly,
`_predict_one` returns a tuple with the prediction and the depth of the node
where the prediction was made.

Note that
`_predict_one` has the `@staticmethod` decorator, which means that it doesn't
have `self` as the first arguments, and thus is why it has to be passed the
node that it has to traverse.

This method can be implemented in a variety of ways, however I would suggest
traversing the tree in a while loop, rather than say, using recursion.

## What to implement

In `src/decision_tree/tree.py`:

- `DecisionTreeClassifier.fit(X, y)` — validate inputs and call
  `_grow` to build the root.
- `DecisionTreeClassifier.predict(X, stop_depth=None, stop_below=None)`
  — raise if the tree has not been fitted yet, otherwise call
  `_predict_one` for every row of `X` and return a list of predicted
  labels.
- `DecisionTreeClassifier.predict_with_depth(X, stop_depth=None, stop_below=None)`
  — same as `predict`, but return a list of `(label, depth)` tuples
  instead of just labels.
- `DecisionTreeClassifier._grow(X, y, depth)` — the recursive worker.
- `DecisionTreeClassifier._predict_one(x, node, stop_depth, stop_below)`
  — walk a single feature vector down the tree, stopping early if
  `stop_depth` or `stop_below` conditions are met, and return a
  `(label, depth)` tuple.

Do not change the function signatures or the provided code.

## Demo: Comparing the boundary at different depths

#### Generating the datasets

First we need to create our datasets using `scripts/create_datasets.py`.
Specifically we'll be running this on synthetic (i.e. generated on your device)
2D datasets. Each will have 1000 samples and use the seed 42. There will be
six datasets in total: `moons`, `circles`, `blobs`, `xor`, `checkerboard`, and
`linear`. An example call to this is:

```bash
uv run python scripts/create_datasets.py moons --n 1000 --seed 42
```

But to type all of that six times in a row seems exhausting so you can also
instead run

```bash
make gen-2D-data
```

With this, if you want to play around more with synthetic dataset creation,
you can get `create_datasets.py`'s usage message by running

```bash
uv run python scripts/create_datasets.py -h
```

Or you can find the generating functions in

```
A2-trees-in-python/src/data/generators.py
```

Note that `create_datasets.py` also creates a scatter plot of the dataset
in `results/figures/datasets/`.

#### Fitting the trees

Second we need to fit all of the trees. To fit one, such as on the `moons`
dataset, run:

```bash
uv run python scripts/fit_tree.py data/raw/moons
```

Without arguments, the tree will use the gini criterion, have no maximum depth,
and will have a minimum sample split of 2. The tree will be saved in
`results/models` and is automatically named `{dataset_name}_{criterion}.json`.

Doing this for every dataset is, again, exhausting, and to do the same thing we
can run

```bash
make fit-2D-all
```
If you wanted to do this with an information entropy criterion, add the
`--criterion entropy` flag:

```bash
uv run python scripts/fit_tree.py data/raw/moons --criterion entropy
```

Also, adding the `-h` or `--help` flag prints the usage message, as is the
case with all scripts in the `scripts` directory.

```bash
uv run python scripts/fit_tree.py --help
```

#### Plot the decision boundaries

Run the script below

```bash
uv run python scripts/plotting/plot_boundary_grid.py \
	moons circles blobs xor checkerboard linear \
	--depths 1 2 3 5 7 None \
	--save results/figures/boundary_grid_2D.png
```

or you could run

```bash
make plot-boundary-grid-2D
```

however, I would encourage you to add depth values to the command to see how
the decision boundary evolves in more detail.

Lastly, it should be noted that if you want to do this all at once, run

```bash
make demo-task-b
```

## How to check your work

Create a `submission.json` file by running

```bash
uv run python submit.py b
```

```bash
make submit_b
```

and drop it in the course webpage for this assignment
(https://py.ml.tu-berlin.de/quiz/#/homework).

Or run the tests for this task with:

```bash
uv run pytest tests/test_tree.py
```

```bash
make test-b
```

although, please note that the test code provided is minimal, and only sees if
the functions can be called without error. There, however, is boilerplate code
that you're welcome to fill out to your own liking.
