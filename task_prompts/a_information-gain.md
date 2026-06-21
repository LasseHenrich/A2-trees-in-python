# Task A — Impurity and Information Gain

We will start this assignment with the simple task of writing three numeric
functions and a utility function that are fundamental components of a
decision tree classifier. But since we aren't in a vacuum, first we must ask...

## What is a decision tree classifier?

A decision tree classifier sorts samples into classes by asking a
sequence of yes / no questions about their features. Each internal node
holds one question (e.g. *"is petal length <= 2.45?"*); the answer
sends the sample left or right. Each leaf holds a class label. To
classify a new sample, we walk it from the root to a leaf and return the
label stored there.

A small example. Imagine we want to label fruit as *apple* (`A`) or
*lemon* (`L`) from two features, `colour` and `weight_g`:

```
                 colour == "yellow" ?
                    /            \
                yes                no
                 /                  \
       weight_g <= 80 ?         predict A
           /        \
         yes         no
          /           \
     predict L     predict A
```

A new fruit that is yellow and weighs 60 g goes left at the root
(yellow), left at the next node (light), and lands on the leaf
predicting `L`. A red fruit goes right at the root and immediately
lands on `A`. Internal nodes hold questions; leaves hold predictions.

We **fit** a tree by looking at labelled training data and choosing,
for every node, the question that does the best job of separating the
classes. We then recurse on each side. We stop when a node is pure
(all samples have the same label), when too few samples remain to be
worth splitting, or when we have reached a maximum depth.

What does *"best job of separating the classes"* mean? That is what the
rest of this task is about.

## Impurity

For a list of labels `S`, an **impurity measure** is a number that says
how mixed the labels are. It is `0` when every label in `S` is the
same, and grows as the labels become more evenly mixed.

We will use two standard measures.

**Gini impurity.** If the proportion of class `c` in `S` is `p_c`, then

```
Gini(S) = 1 - sum_c p_c^2
```

For a two-class problem `Gini` ranges from `0` (pure) to `0.5` (50/50
mix).

**Shannon entropy.** Same setup, different formula:

```
Entropy(S) = - sum_c p_c * log2(p_c)
```

For a two-class problem `Entropy` ranges from `0` (pure) to `1` bit
(50/50 mix). Terms with `p_c == 0` are skipped (`0 * log 0` is taken
to be `0`).

The two measures usually rank splits the same way. Gini is slightly
cheaper to compute; entropy comes from information theory and is
easier to interpret as *"bits of uncertainty about the label"*.

## Information gain

A **split** on a node partitions its labels `S` into a left subset
`S_L` and a right subset `S_R`. The split's **information gain** is
how much the impurity drops, weighted by how many samples ended up on
each side:

```
IG(S, S_L, S_R) = I(S)
                  - (|S_L| / |S|) * I(S_L)
                  - (|S_R| / |S|) * I(S_R)
```

`I` is whichever impurity measure we have chosen. A larger
information gain means a cleaner split.

## Finding the best split

For every feature we consider every candidate split:

- For a **numeric** feature, the candidates are the midpoints between
  consecutive unique values. Any threshold inside a gap produces the
  exact same partition, so the midpoint is a fair representative.
- For a **categorical** feature, each unique value `v` is a candidate;
  samples with `feature == v` go left, the rest go right.

We compute the information gain of each candidate and keep the one
with the largest gain. If no split has strictly positive gain, the
node should become a leaf.

## What to implement

In `src/decision_tree/impurity.py`:

- `gini(y)` — Gini impurity of a label list. Empty list returns `0.0`.
- `entropy(y)` — Shannon entropy in bits. Empty list returns `0.0`.
- `information_gain(y, y_left, y_right, criterion)` — uses any
  impurity callable. Raises `ValueError` if `y` is empty or if the
  children do not partition `y`.

In `src/decision_tree/split.py`:

- `best_split(X, y, criterion, feature_types)` — returns
  `(feature_index, split_value, gain)` for the best split, or `None`
  if no split yields strictly positive gain.

The function signatures and the helper `candidate_thresholds` are
already provided. Do not change them.

## How to check your work

Create a `submission.json` file by running

```bash
uv run python submit.py a
```

```bash
make submit_a
```

and drop it in the course webpage for this assignment
(https://py.ml.tu-berlin.de/quiz/#/homework).

Or run the tests for this task:

```bash
uv run pytest tests/test_impurity.py tests/test_split.py
```

```bash
make test-a
```

although, please note that the test code provided is minimal, and only sees if
the functions can be called without error. There, however, is boilerplate code
that you're welcome to fill out to your own liking.
