# LongLif Liner

A Python research repository containing two related experimental workflows:

1. **Current-characteristic analysis** — extracts peaks from 32 C2C curves, aligns their coordinates, derives target-current values, and models the relationship between target current and measurement variation.
2. **MNIST noise-mapping experiments** — maps neural-network weights through a differential-current codebook and evaluates MNIST accuracy across state-proportional noise coefficients (`alpha`).

It also includes a standalone **sequential continual-learning** experiment based on exclusive fully connected (FC) weight allocation.

> The original code and experimental outputs were retained; this repository has been reorganized for reproducible GitHub use. Dataset files, model checkpoints, and large generated figures are intentionally ignored by Git.

## Repository structure

```text
longLif_liner/
├── data/
│   ├── raw/                         # Source measurement data (C2C.xlsx)
│   └── datasets/                    # Downloaded torchvision datasets (ignored)
├── docs/
│   └── PROJECT_LAYOUT.md
├── results/                         # Generated outputs, checkpoints, figures (ignored)
├── scripts/
│   ├── current_analysis/            # Measurement/current analysis pipeline
│   ├── mnist_noise/                 # MNIST alpha-sweep experiments
│   └── continual_learning/          # Independent continual-learning experiment
├── .gitignore
├── requirements.txt
└── README.md
```

See [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md) for a detailed mapping of folders.

## Installation

Python 3.10+ is recommended. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PyTorch installation can depend on your CUDA version. If the generic installation above is unsuitable, install a matching build from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) first, then install the remaining packages.

## Workflow 1: Current-characteristic analysis

The input file is expected at `data/raw/C2C.xlsx`. Run the following commands from the repository root, in order:

```bash
python scripts/current_analysis/extract_peaks.py
python scripts/current_analysis/align_peaks.py
python scripts/current_analysis/fit_aligned_peaks.py
python scripts/current_analysis/derive_target_current.py
python scripts/current_analysis/fit_alpha_sigma.py
python scripts/current_analysis/analyze_current_distribution.py
```

### Outputs

The scripts write their outputs under `results/current_analysis/`:

- `peaks/`: per-curve peak visualizations, peak tables, and aligned coordinate data.
- `modeling/`: fitted target-current data and global fit results.
- `modeling/alpha_sigma_analysis/`: per-target standard deviation and fitted alpha–sigma relationship.
- `modeling/target_current_distribution_modeling/`: distribution statistics and histograms.

## Workflow 2: MNIST current-noise experiments

The MNIST data is stored in `data/datasets/`. Download it if it is absent:

```bash
python scripts/mnist_noise/download_mnist.py
```

The noise-sweep scripts consume the alpha–sigma table produced by Workflow 1:

```bash
# Coarse alpha sweep; trains a clean MNIST model and evaluates noisy mappings
python scripts/mnist_noise/run_coarse_alpha_scan.py

# Inference-oriented alpha sweep; creates/uses a clean model in its output folder
python scripts/mnist_noise/run_inference_alpha_scan.py

# Fine alpha sweep; expects the clean checkpoint from the inference scan
python scripts/mnist_noise/run_fine_alpha_scan.py
```

Outputs are written under `results/mnist_noise/`:

- `coarse_scan/`
- `inference_scan/`
- `fine_scan/`

Each run records its configuration, alpha/accuracy table, differential-current codebook, figures, and (where applicable) an MNIST checkpoint.

## Continual-learning experiment

The standalone experiment trains a masked MLP on a task sequence:

`MNIST → QMNIST → EMNIST-Digits → Fashion-MNIST → EMNIST-Letters → EMNIST-Balanced`

Datasets are downloaded automatically by the training script when needed. Run:

```bash
python scripts/continual_learning/training/fc.py
```

A timestamped output folder is created under `results/continual_learning/`. Plot a selected run with:

```bash
python scripts/continual_learning/result_processing/plot_fc_accuracy_sequence.py --run-dir results/continual_learning/<run-folder>
python scripts/continual_learning/result_processing/plot_fc_continual_accuracy_full.py --run-dir results/continual_learning/<run-folder>
python scripts/continual_learning/result_processing/plot_fc_cumulative_weight_heatmaps.py --run-dir results/continual_learning/<run-folder>
```

## Reproducibility notes

- The alpha-sweep scripts use fixed random seed `42` by default.
- Most experiment parameters are declared near the top of each script and saved to JSON in the corresponding result directory.
- Generated `.tif` figures are intentionally high resolution and may be large.
- Existing files under `results/` are retained locally but excluded from Git. Re-run the scripts to generate a clean set of outputs after cloning.

## GitHub hygiene

The provided `.gitignore` excludes IDE metadata, Python caches, downloaded datasets, experiment outputs, model checkpoints, archives, and large TIFF figures. Before publishing, review `data/raw/C2C.xlsx` to ensure it may be shared; it is ignored by default because it is an input dataset.

## License

No license was supplied with the original project. Add a `LICENSE` file before distributing the repository publicly.
