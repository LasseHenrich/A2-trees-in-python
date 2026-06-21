# Task D — Hyperparameter Search

Your tree works. Now the question is: which configuration of
hyperparameters gives the best results, and how do you measure best?

This task introduces a way to search efficiently over `max_depth`
and `min_samples_split` without retraining the tree, and a more
informative way to measure classifier quality than accuracy alone.

## Stopping parameters simulate training hyperparameters

When you call `predict`, you can pass two optional arguments:

- `stop_depth` — stops traversal at this depth and returns the
  prediction stored at that node.
- `stop_below` — stops at any node whose `n_samples` is below
  this value.

These let you simulate a shallower or more conservative tree without
retraining. This works because every internal node stores a
`prediction` and an `n_samples` count — the same information a
shallower tree would store at its leaves.

For example, a tree trained with no depth limit and `stop_depth=3`
gives the same predictions as a tree trained with `max_depth=3`
(provided the full tree reaches past depth 3 everywhere on the
training data). The same logic applies to `stop_below` and
`min_samples_split`.

This means you only need to train **one tree per criterion** to
search over many values of `max_depth` and `min_samples_split`.
You need a separate tree for each `criterion` value (`"gini"` or
`"entropy"`) because the criterion changes which splits are chosen
during training, not just when to stop.

## Bias and variance

Every model has two sources of error.

**Bias** is the error that comes from a model that is too simple to
capture the true pattern in the data. A decision tree with
`max_depth=1` can draw only a single split — it will misclassify most
samples in a dataset like moons no matter how much training data you
give it. This is high-bias behaviour.

**Variance** is the error that comes from a model that is too sensitive
to the specific training data it saw. A fully grown tree often
memorises the training set: it classifies every training sample
correctly, but generalises poorly to new data. Different training sets
of the same size would produce very different trees. This is
high-variance behaviour.

The depth of the tree controls this tradeoff. A shallow tree has high
bias and low variance; a deep tree has low bias and high variance.
Somewhere in between is a depth that minimises the error on new data.

For further reading, see the
[bias–variance tradeoff on Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff).

## Confusion matrix

Accuracy is the fraction of predictions that match the true label.
It is a useful summary, but it hides *which* classes the model gets
wrong. A **confusion matrix** shows the full picture.

For a binary classifier with classes 0 and 1, the confusion matrix
is a 2×2 table:

```
                  Predicted 0   Predicted 1
  True class 0  [     TN             FP     ]
  True class 1  [     FN             TP     ]
```

- **TN** (true negative): true class 0, predicted 0.
- **FP** (false positive): true class 0, predicted 1.
- **FN** (false negative): true class 1, predicted 0.
- **TP** (true positive): true class 1, predicted 1.

The `confusion_matrix` function in `src/evaluation/metrics.py`
returns `(classes, M)` where `M[i][j]` is the count of samples with
true class `classes[i]` predicted as `classes[j]`. Read it before
you start.

## Type-I and Type-II error

Two rates from the confusion matrix are useful when the cost of each
kind of mistake is different.

**Type-I error** (false positive rate) — the fraction of true
negatives that were predicted positive:

    FPR = FP / (FP + TN)

**Type-II error** (false negative rate) — the fraction of true
positives that were predicted negative:

    FNR = FN / (FN + TP)

If false positives are costly (blocking a valid transaction), you
want a low FPR even if FNR rises. If false negatives are costly
(missing a medical finding), you want a low FNR even if FPR rises.
Accuracy does not capture this trade-off.

## What to implement

In `src/evaluation/tuning.py`, implement `stopping_search` and
`grid_search`. Do not modify function signatures.

**`stopping_search`** evaluates a fitted tree across every
`(stop_depth, stop_below)` pair and returns one result dict per
combination. Use `accuracy` and `confusion_matrix` from
`src/evaluation/metrics.py`, and `predict_with_depth` from the tree
to record the average depth at which predictions were made. The expected
keys are in the docstring.

**`grid_search`** accepts training data, validation data, and a `grid`
dict with keys `"criterion"`, `"stop_depth"`, and `"stop_below"` (each
mapping to a list of values to try). It fits one tree per criterion in
`grid["criterion"]`, then calls `stopping_search` on each tree against
the validation data. It tags every result dict with its criterion and
returns the full list of result dicts — one per
`(criterion, stop_depth, stop_below)` combination.

To avoid unnecessary training, each tree is trained with only the depth
and sample count it needs to support the given search space:

- `max_depth` is set to the largest value in `grid["stop_depth"]`, or
  `None` (unlimited) if `None` appears in that list.
- `min_samples_split` is set to the smallest value in
  `grid["stop_below"]`, or left at its default if `None` appears in
  that list.

This way, `grid_search` trains the most restricted tree that can still
answer every stopping-parameter query in the grid.

**Optional challenge**: implement `stopping_search` so that each
sample is walked through the tree only once. During that single
traversal, record the prediction at each depth and the depth at
which `n_samples` first drops below each `stop_below` threshold.
Then answer every `(stop_depth, stop_below)` query from that record
without calling `predict` again. This reduces the number of tree
traversals from `len(stop_depths) * len(stop_belows) * len(X)` to
`len(X)`.

## Optional — Visualize the bias-variance tradeoff

Once you have a fitted tree, you can see the bias-variance tradeoff
directly. Run:

```bash
make plot-depth-accuracy-moons
```

This loads the moons model you trained in Task B and plots accuracy on
both the training set and the test set across depths 1 to 15.

Training accuracy rises monotonically — a deeper tree can always fit
the training data more closely. Test accuracy rises to a point and then
plateaus. The growing gap between the two curves shows the tree
beginning to fit the training data more closely than the underlying
pattern, without gaining anything on unseen data.

## How to check your work

Create a `submission.json` file by running

```bash
make submit_d
```

and drop it in the course webpage for this assignment
(https://py.ml.tu-berlin.de/quiz/#/homework).

Or run the tests for this task:

```bash
make test-d
```

Please note that the test code provided is minimal, and only checks
that the functions can be called without error. There is boilerplate
code that you are welcome to fill out to your own liking.
