"""
Answers to observation questions in the task prompts.

Fill in each variable below with the value you observed. Replace `None`
with your answer. Types are shown in the comments.

Run `uv run python submit.py <task>` to bundle your code and these
answers into submission.json.
"""

# ---------------------------------------------------------------------------
# Task C — Feature Engineering
# ---------------------------------------------------------------------------

# Step 5: Look at the depth-1 column of boundary_grid_2D.png (raw datasets)
# and boundary_grid_processed.png (processed datasets).
#
# At which depth in the RAW boundary grid does `moons` first reach or surpass
# the same test accuracy as `moons-processed` achieves at depth 1?
# Type: int
c_moons_raw_min_depth: int | None = None

# At which depth in the RAW boundary grid does `checkerboard` first reach or
# surpass the same test accuracy as `checkerboard-processed` achieves at depth 1?
# Type: int
c_checkerboard_raw_min_depth: int | None = None

# ---------------------------------------------------------------------------
# Task F — Scripting / Retrospective
# ---------------------------------------------------------------------------

# Run the depth grid search on both raw and processed moons (instructions in
# the task prompt). For each table, find the smallest depth whose mean_acc is
# within one std_acc of the best mean_acc.
#
# For which dataset is that threshold reached at a lower depth?
# Type: str — either "raw" or "processed"
f_retro_lower_depth_dataset: str | None = None

# ---------------------------------------------------------------------------
# Task G — MNIST
# ---------------------------------------------------------------------------

# Step 3: Run the cross-validation grid search on mnist-2x with
#   --max-depth 5 10 15 20 None --criterion gini --folds 5
#
# What is the smallest max_depth whose mean_acc is within std_acc of the
# best mean_acc? (Use None if the unlimited tree is the plateau point.)
# Type: int | None
g_mnist_2x_plateau_depth: int | None = None

# Step 3: Pick any two adjacent depths near the top of the ranking.
# Is the difference in their mean_acc larger than the std_acc of the better
# one? (True → yes it is significant; False → the gap is within normal variation)
# Type: bool
g_mnist_2x_diff_significant: bool | None = None

# Step 5: Run single-configuration cross-validation at your chosen depth.
# Which digit class (0–9) has the highest per-class accuracy?
# Type: int
g_mnist_2x_easiest_class: int | None = None

# Which digit class (0–9) has the lowest per-class accuracy?
# Type: int
g_mnist_2x_hardest_class: int | None = None

# Step 6: Run cross-validation on mnist-4x at the same depth.
# Is the difference in mean_acc between mnist-2x and mnist-4x larger than
# either of their std_acc values?
# Type: bool
g_resolution_matters: bool | None = None
