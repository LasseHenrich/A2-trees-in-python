# Assignment 2 — Trees in Python

A from-scratch decision tree classifier implemented in pure Python.
You will build the classifier piece by piece across seven tasks, ending
with an evaluation on the MNIST handwritten-digit dataset.

## Setup

```bash
make install
```

## Tasks

Each task has a prompt in `task_prompts/` that explains what to implement
and why. Ideally they should be worked through in order, as they build on each
other. It should be noted, however, that you are able to use reference mode to
experiment and interact with the parts of the assignment that require the
portions of code in `src` that you are expected to implement. Still, reference
mode does not cover `scripts/evaluate.py`, which contains functions that you
are expected to implement.

| Task | Topic | Files to implement |
|------|-------|--------------------|
| A | Impurity and Information Gain | `src/decision_tree/impurity.py`, `src/decision_tree/split.py` |
| B | The Decision Tree Classifier | `src/decision_tree/tree.py` |
| C | Feature Engineering | preprocessing configs in `configs/preprocessing/`, answers in `answers.py` |
| D | Hyperparameter Search | `src/evaluation/tuning.py` |
| E | Cross-Validation | `src/evaluation/cross_validation.py` |
| F | Writing the `evaluate.py` Script | `scripts/evaluate.py`, answer in `answers.py` |
| G | MNIST | `src/data/feature_map.py`, answers in `answers.py` |

## Running tests and demos

Each task has a corresponding `make` target for running its tests and
for producing demo outputs. For example, for task B:

```bash
make test-b       # run the tests for task B
make demo-task-b  # generate data, fit trees, and plot decision boundaries
```

## Submitting

```bash
uv run python submit.py a        # submit task A only
uv run python submit.py          # submit all tasks
```

This writes `submission.json`, which you upload to the course webpage.

## Reference mode

A compiled reference implementation is included in `src_reference/`. You
can use it to run demos and tests without having implemented anything
yourself. This is useful for understanding what the finished project looks
like before you start writing code.

```bash
make use-reference   # switch src/ to the reference implementation
make use-student     # switch back to your own implementation
```

Running `submit.py` in reference mode still produces a `submission.json`,
but all results that correspond to code you are supposed to implement will
be marked invalid. Only your written answers in `answers.py` are graded
from that file.
