# Repository layout

- `scripts/current_analysis/`: peak extraction, coordinate alignment, current-target fitting, and alpha–sigma analysis.
- `scripts/mnist_noise/`: MNIST download plus coarse, inference-only, and fine alpha-sweep experiments.
- `scripts/continual_learning/`: separate sequential continual-learning experiments and their plotting utilities.
- `data/raw/`: source measurement files. `C2C.xlsx` is retained locally but ignored by Git because of its size.
- `data/datasets/`: torchvision datasets downloaded locally; ignored by Git.
- `results/`: generated figures, tables, configurations, and model checkpoints; ignored by Git.

The repository intentionally excludes datasets, model checkpoints, and generated high-resolution images. Recreate them by running the corresponding scripts.
