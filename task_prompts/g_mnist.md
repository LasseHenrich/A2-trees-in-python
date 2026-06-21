# Task G — MNIST

MNIST is a dataset of handwritten digit images (0–9), each 28×28 pixels
stored as a flat row of 784 grayscale values (0–255). It is a standard
benchmark in machine learning and a good test of how well the tree
handles high-dimensional numeric data.

In this task you will implement image downsampling, apply it to MNIST,
and use the tools you have built to find a good configuration and
understand where the tree makes mistakes.

## Step 1 — Download the data

```bash
make download-mnist
```

This writes `train.csv` and `test.csv` to `data/raw/mnist/`.

## Step 2 — Downsampling

A 28×28 image has 784 features. Working with that many features is
expensive and most of the information is duplicated in neighbouring
pixels. **Average-pool downsampling** reduces the image by grouping
pixels into non-overlapping square blocks and replacing each block
with its mean.

A 2× downsample groups pixels into 2×2 blocks:

```
original 4×4 patch:     after 2× downsample:

  1   2   5   6           3.5   5.5
  3   4   7   8    →
  9  10  13  14          11.5  13.5
 11  12  15  16
```

Each block mean becomes one feature in the output. A 28×28 image
becomes 14×14 (196 features) at 2× and 7×7 (49 features) at 4×.

### What to implement

In `src/data/feature_map.py`, implement `make_downsample_img(width,
factor)`. It takes the side length of the square image and the pooling
block size, and returns a function. That function takes a flat image
row (a list of `width * width` floats) and returns the list of block
averages, in row-major order (left to right, top to bottom).

Do not modify the function signature.

Run the tests to check your implementation:

```bash
make test-feature-map
```

### Apply to MNIST

Once `make_downsample_img` works, implement
`scripts/downsample_mnist.py`. The script loads
`data/raw/mnist/train.csv` and `test.csv`, applies
`make_downsample_img(28, factor)` to every row, and writes the result
to `data/processed/mnist-{factor}x/`.

Generate both downsampled versions:

```bash
make downsample-mnist-2x
make downsample-mnist-4x
```

You can view one example of each digit for each version:

```bash
make plot-mnist-digits
```

## Step 3 — Find a good depth with cross-validation

Train a tree on the 14×14 (2×) version:

```bash
make fit-mnist-2x
```

Now search over several depth values using cross-validation:

```bash
uv run python scripts/evaluate.py data/processed/mnist-2x \
    --max-depth 5 10 15 20 None --criterion gini --folds 5
```

```bash
make evaluate-mnist-2x-grid
```

The output table ranks every depth by `mean_acc` — the mean accuracy
across the five folds — and includes a `std_acc` column.

`std_acc` is the standard deviation of accuracy across those five
folds. Because each fold trains on a different portion of the data
and validates on a different portion, accuracy varies slightly between
folds. `std_acc` measures how much it varies.

This matters when comparing configurations. If depth=15 has
`mean_acc=0.82` and `std_acc=0.03`, the accuracy you would expect on a
new split of the same data is somewhere around 0.82 ± 0.03. If
depth=20 has `mean_acc=0.83` with `std_acc=0.03`, those two intervals
overlap entirely — the data cannot reliably tell them apart.

Two questions to answer from the table:

1. **Where does accuracy plateau?** Find the smallest depth whose
   `mean_acc` is within `std_acc` of the best `mean_acc`. Using a
   deeper tree beyond this point does not reliably improve accuracy.

2. **Does a difference mean anything?** Pick two adjacent depths near
   the top of the ranking. Is the difference in `mean_acc` larger than
   the `std_acc` of the better one? If not, the improvement is within
   the expected variation across data splits.

> **Record your answers** in `answers.py`:
> `g_mnist_2x_plateau_depth` and `g_mnist_2x_diff_significant`.

## Step 4 — Visualize the train/test gap

```bash
make plot-depth-accuracy-mnist-2x
```

This plots training and test accuracy across depths 1 to 20 for the
2× dataset. Training accuracy rises steadily as the tree grows.
Test accuracy rises to a point and then plateaus while training
accuracy keeps climbing. The gap between the two curves is larger
here than on the toy datasets — MNIST has enough classes and features
that a deep tree can fit the training data closely without learning
anything that generalises to the test set.

Compare this to the moons plot from Task D. What is different about
the size of the gap and the depth at which the test curve levels off?

## Step 5 — Per-class accuracy

Run cross-validation with the single depth you chose above:

```bash
uv run python scripts/evaluate.py data/processed/mnist-2x \
    --max-depth <depth> --criterion gini --folds 5
```

Look at the per-class accuracy printed at the bottom of the output.
Which digits are easiest for the tree to classify? Which have the
lowest accuracy?

> **Record your answers** in `answers.py`:
> `g_mnist_2x_easiest_class` and `g_mnist_2x_hardest_class`.

To see which digits are most often confused with each other, generate
the confusion matrix plot:

```bash
make plot-confusion-matrix-mnist-2x
```

Each row of the matrix corresponds to a true class and each column to
a predicted class. A cell at row `i`, column `j` shows how many
digits of class `i` were predicted as class `j`. The diagonal holds
correct predictions; off-diagonal cells are mistakes.

Look at the off-diagonal cells with the highest counts. Do the pairs
of confused digits look visually similar in `make plot-mnist-digits`?

## Step 6 — Does resolution matter?

Run cross-validation on the 7×7 (4×) version with the same depth:

```bash
make evaluate-mnist-4x
```

Is the difference in `mean_acc` between the two resolutions larger
than either `std_acc`? What does this tell you about whether the extra
features in the 2× version carry useful information for this dataset?

> **Record your answer** in `answers.py`: `g_resolution_matters`.

## Deliverables

- `make_downsample_img` in `src/data/feature_map.py`
- `scripts/downsample_mnist.py`

## How to check your work

Create a `submission.json` file by running

```bash
make submit_g
```

and drop it in the course webpage for this assignment
(https://py.ml.tu-berlin.de/quiz/#/homework).
