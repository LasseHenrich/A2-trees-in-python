.PHONY: gen-data-moons gen-data-circles gen-data-blobs \
        gen-data-xor gen-data-checkerboard gen-data-linear \
        gen-2D-data \
        fit-moons fit-circles fit-blobs \
        fit-xor fit-checkerboard fit-linear \
        fit-2D-all \
        plot-boundary-grid-2D \
        preprocess-circles preprocess-xor preprocess-moons \
        preprocess-blobs preprocess-checkerboard preprocess-linear \
        preprocess-all \
        fit-circles-processed fit-blobs-processed fit-xor-processed \
        fit-moons-processed fit-checkerboard-processed \
        fit-linear-processed fit-all-processed \
        plot-boundary-grid-processed plot-boundary-grid-raw-fine \
        demo-task-b demo-task-c submit_c \
        test-d submit_d test-e submit_e \
        evaluate-moons evaluate-moons-grid \
        evaluate-moons-depth-fine evaluate-moons-processed-depth-fine \
        evaluate-moons-k2 evaluate-moons-k5 evaluate-moons-k10 \
        plot-confusion-matrix-moons submit_f \
        download-mnist downsample-mnist-2x downsample-mnist-4x \
        plot-mnist-digits fit-mnist-2x fit-mnist-4x evaluate-mnist-2x \
        evaluate-mnist-2x-grid \
        plot-confusion-matrix-mnist-2x \
        submit_g demo-task-g \
        test-impurity test-split test-a test-b submit_a submit_b \
        test-feature-map test-c \
        clean clean-results clean-datasets \
 use-reference use-student


# ---- Task B Demos -----------------------------------------------------------

gen-data-moons:
	uv run python scripts/create_datasets.py moons --n 1000 --seed 42

gen-data-circles:
	uv run python scripts/create_datasets.py circles --n 1000 --seed 42

gen-data-blobs:
	uv run python scripts/create_datasets.py blobs --n 1000 --seed 42

gen-data-xor:
	uv run python scripts/create_datasets.py xor --n 1000 --seed 42

gen-data-checkerboard:
	uv run python scripts/create_datasets.py checkerboard \
		--n 1000 --seed 42 --grid 4

gen-data-linear:
	uv run python scripts/create_datasets.py linear --n 1000 --seed 42

gen-2D-data: gen-data-moons gen-data-circles gen-data-blobs \
             gen-data-xor gen-data-checkerboard gen-data-linear

fit-moons:
	uv run python scripts/fit_tree.py data/raw/moons

fit-circles:
	uv run python scripts/fit_tree.py data/raw/circles

fit-blobs:
	uv run python scripts/fit_tree.py data/raw/blobs

fit-xor:
	uv run python scripts/fit_tree.py data/raw/xor

fit-checkerboard:
	uv run python scripts/fit_tree.py data/raw/checkerboard

fit-linear:
	uv run python scripts/fit_tree.py data/raw/linear

fit-2D-all: fit-moons fit-circles fit-blobs \
              fit-xor fit-checkerboard fit-linear

plot-boundary-grid-2D:
	uv run python scripts/plotting/plot_boundary_grid.py \
		moons circles blobs xor checkerboard linear \
		--depths 1 2 3 5 7 None \
		--save results/figures/boundary_grid_2D.png

demo-task-b: gen-2D-data fit-2D-all plot-boundary-grid-2D


# ---- Task C Demos -----------------------------------------------------------

preprocess-circles:
	uv run python scripts/preprocess.py \
		configs/preprocessing/circles-processed.json

preprocess-xor:
	uv run python scripts/preprocess.py \
		configs/preprocessing/xor-processed.json

preprocess-moons:
	uv run python scripts/preprocess.py \
		configs/preprocessing/moons-processed.json

preprocess-blobs:
	uv run python scripts/preprocess.py \
		configs/preprocessing/blobs-processed.json

preprocess-checkerboard:
	uv run python scripts/preprocess.py \
		configs/preprocessing/checkerboard-processed.json

preprocess-linear:
	uv run python scripts/preprocess.py \
		configs/preprocessing/linear-processed.json

preprocess-all: preprocess-circles preprocess-xor preprocess-moons \
                preprocess-blobs preprocess-checkerboard preprocess-linear

fit-circles-processed:
	uv run python scripts/fit_tree.py data/processed/circles-processed

fit-xor-processed:
	uv run python scripts/fit_tree.py data/processed/xor-processed

fit-moons-processed:
	uv run python scripts/fit_tree.py data/processed/moons-processed

fit-blobs-processed:
	uv run python scripts/fit_tree.py data/processed/blobs-processed

fit-checkerboard-processed:
	uv run python scripts/fit_tree.py data/processed/checkerboard-processed

fit-linear-processed:
	uv run python scripts/fit_tree.py data/processed/linear-processed

fit-all-processed: fit-circles-processed fit-xor-processed \
                   fit-moons-processed fit-checkerboard-processed \
                   fit-linear-processed fit-blobs-processed

plot-boundary-grid-processed:
	uv run python scripts/plotting/plot_boundary_grid.py \
		moons-processed circles-processed blobs-processed \
		xor-processed checkerboard-processed linear-processed \
		--depths 1 2 3 5 7 None \
		--save results/figures/boundary_grid_2D_processed.png

plot-boundary-grid-raw-fine:
	uv run python scripts/plotting/plot_boundary_grid.py \
		moons checkerboard \
		--depths 1 2 3 4 5 \
		--save results/figures/boundary_grid_raw_fine.png

demo-task-c: gen-data-circles preprocess-circles \
             fit-circles fit-circles-processed

submit_c:
	uv run python submit.py c


# ---- Task D Demos -----------------------------------------------------------

test-d:
	uv run pytest tests/test_tuning.py

submit_d:
	uv run python submit.py d

plot-depth-accuracy-moons:
	uv run python scripts/plotting/plot_depth_accuracy.py moons \
		--max-depth 15 \
		--save results/figures/depth_accuracy_moons.png


# ---- Task E Demos -----------------------------------------------------------

test-e:
	uv run pytest tests/test_cross_validation.py

submit_e:
	uv run python submit.py e


# ---- Task F Demos -----------------------------------------------------------

evaluate-moons:
	uv run python scripts/evaluate.py data/raw/moons \
		--max-depth 5 --criterion gini --folds 5 --seed 0

evaluate-moons-grid:
	uv run python scripts/evaluate.py data/raw/moons \
		--max-depth 3 5 10 None --folds 5 --seed 0

evaluate-moons-depth-fine:
	uv run python scripts/evaluate.py data/raw/moons \
		--max-depth 1 2 3 4 5 --criterion gini --folds 5

evaluate-moons-processed-depth-fine:
	uv run python scripts/evaluate.py data/processed/moons-processed \
		--max-depth 1 2 3 4 5 --criterion gini --folds 5

evaluate-moons-k2:
	uv run python scripts/evaluate.py data/raw/moons \
		--max-depth 5 --criterion gini --folds 2

evaluate-moons-k5:
	uv run python scripts/evaluate.py data/raw/moons \
		--max-depth 5 --criterion gini --folds 5

evaluate-moons-k10:
	uv run python scripts/evaluate.py data/raw/moons \
		--max-depth 5 --criterion gini --folds 10

plot-confusion-matrix-moons:
	uv run python scripts/plotting/create_confusion_matrix.py moons \
		--save results/figures/confusion_matrix_moons.png

submit_f:
	uv run python submit.py f


# ---- Task G Demos -----------------------------------------------------------

download-mnist:
	uv run python scripts/create_datasets.py mnist

downsample-mnist-2x:
	uv run python scripts/downsample_mnist.py --factor 2

downsample-mnist-4x:
	uv run python scripts/downsample_mnist.py --factor 4

create-all-mnist: download-mnist downsample-mnist-4x downsample-mnist-2x

plot-mnist-digits:
	uv run python scripts/plotting/show_mnist_digits.py \
		--save results/figures/mnist_digits.png

fit-mnist-2x:
	uv run python scripts/fit_tree.py data/processed/mnist-2x \
		--max-depth 20

fit-mnist-4x:
	uv run python scripts/fit_tree.py data/processed/mnist-4x \
		--max-depth 20

evaluate-mnist-2x:
	uv run python scripts/evaluate.py data/processed/mnist-2x \
		--max-depth 20

evaluate-mnist-2x-grid:
	uv run python scripts/evaluate.py data/processed/mnist-2x \
		--max-depth 5 10 15 20 None --criterion gini --folds 5

evaluate-mnist-4x:
	uv run python scripts/evaluate.py data/processed/mnist-4x \
		--max-depth 20

plot-confusion-matrix-mnist-2x:
	uv run python scripts/plotting/create_confusion_matrix.py mnist-2x \
		--max-depth 20 \
		--save results/figures/confusion_matrix_mnist_2x.png

submit_g:
	uv run python submit.py g

demo-task-g: download-mnist downsample-mnist-2x downsample-mnist-4x \
             plot-mnist-digits fit-mnist-2x evaluate-mnist-2x

plot-depth-accuracy-mnist-2k:
	uv run python scripts/plotting/plot_depth_accuracy.py mnist-2x \
		--max-depth 20 \
		--save results/figures/depth-accuracy-mnist-2k.png

plot-depth-accuracy-mnist-4k:
	uv run python scripts/plotting/plot_depth_accuracy.py mnist-4x \
		--max-depth 20 \
		--save results/figures/depth-accuracy-mnist-4k.png



# ---- Test Scripts -----------------------------------------------------------

test-impurity:
	uv run pytest tests/test_impurity.py

test-split:
	uv run pytest tests/test_split.py

test-a: test-impurity test-split

submit_a:
	uv run python submit.py a

test-b:
	uv run pytest tests/test_tree.py

submit_b:
	uv run python submit.py b

test-feature-map:
	uv run pytest tests/test_feature_map.py

test-c: test-feature-map


# ---- Utilities --------------------------------------------------------------

install:
	git init
	uv sync

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-results:
	find results -mindepth 1 -not -type d -not -name '.gitkeep' -delete

clean-datasets:
	find data -mindepth 1 -not -type d -not -name '.gitkeep' -delete


use-reference:
	@if [ -d src_student_backup ]; then \
	    echo "Already in reference mode. Run 'make use-student' first."; \
	    exit 1; \
	elif [ ! -d src_reference ]; then \
	    echo "src_reference/ not found. Run 'make' first."; \
	    exit 1; \
	else \
	    mv src src_student_backup && \
	    cp -r src_reference src && \
	    touch .reference_mode && \
	    echo "Reference mode active. Run 'make use-student' to switch back."; \
	fi

use-student:
	@if [ ! -d src_student_backup ]; then \
	    echo "Already in student mode."; \
	else \
	    rm -rf src && \
	    mv src_student_backup src && \
	    rm -f .reference_mode && \
	    echo "Student mode active."; \
	fi
