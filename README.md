# ALE-MTL: Accumulated Local Effects for Multi-task Learning

ALE-MTL is a Python/PyTorch toolkit for building, training, explaining, and
comparing multi-task learning models. The software focuses on a workflow where
task-specific model behavior is summarized with Accumulated Local Effects (ALE)
profiles and task relationships are estimated with curve similarity, including a
Frechet-based similarity score.

ALEMTL is a Python library for computing interpretable ALE–Fréchet
task similarities and integrating them into deep multi-task learning
workflows.

## Statement of Need

Multi-task learning (MTL) models can share representations across related tasks,
but deciding which tasks should share parameters is often treated as a fixed
design choice. ALE-MTL provides reusable software components for:

- constructing hard-shared and soft-shared MTL architectures;
- computing task-wise ALE profiles on a selected model slice;
- comparing task behavior from ALE curves;
- using similarity-derived task groups during training regularization;
- benchmarking against common MTL baselines.

The package is designed for experiments where researchers need both predictive
models and interpretable task relationship estimates.

## Main Features

- **Composable multitask models**: hard-shared modules, soft-shared modules, and
  per-task model views.
- **Task-first data contract**: batches follow `(n_tasks, batch, features)` and
  outputs follow `(n_tasks, batch, outputs)`.
- **ALE profile computation**: interval construction, perturbation generation,
  local-effect accumulation, centering, standardization, and optional spline
  smoothing.
- **Task similarity**: vectorized discrete Frechet curve comparison and nearest
  task grouping.
- **Training utilities**: per-task loss tracking, similarity-weighted
  regularization, ALE/similarity scheduling, early stopping, and checkpointable
  metrics.
- **Baseline models**: soft sharing, single-task MLP, hard sharing, MMoE,
  Cross-Stitch, PLE, and MTAN-style tabular baselines.
- **Executable examples and notebooks** for reproducible demonstrations.

## Repository Layout

```text
src/alemtl/
├── data/          # multitask datasets and task-first dataloader wrapper
├── models/        # composable MultiTaskModel and baseline MTL models
├── similarity/    # ALE computation and task similarity
└── training/      # loss, trainer, and tracking utilities

examples/          # runnable scripts
notebooks/         # guided workflows
docs/              # Sphinx documentation
scripts/           # Dataset download and preprocessing utilities
tests/             # pytest regression tests
```

## Installation

The project uses a standard `src/` layout. For normal use, install from the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

For development and testing, install the test tools with an editable install:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The test suite should complete without failures.

## Quick Start

Create a synthetic multitask dataset and inspect the task-first batch contract:

```bash
python -m examples.quickstart_multisine
```

Build a small `MultiTaskModel`:

```bash
python -m examples.multitask_model_layout
```

Compute ALE profiles:

```bash
python -m examples.compute_ale_profiles
```

Compute task similarity from ALE curves:

```bash
python -m examples.compute_task_similarity
```

Run a tiny ALE-Frechet training workflow:

```bash
python -m examples.train_alefrechet
```

Compare baseline model forward contracts:

```bash
python -m examples.compare_baselines
```

## Minimal Python Example

```python
import torch
from torch import nn

from alemtl.models import MultiTaskModel
from alemtl.similarity import MultiTaskALE, MultitaskSimilarity

model = MultiTaskModel(
    n_tasks=3,
    modules_layout={
        "encoder": {
            "shared": "hard",
            "module": lambda: nn.Sequential(nn.Linear(2, 8), nn.ReLU()),
        },
        "head": {
            "shared": "soft",
            "module": lambda: nn.Linear(8, 1),
        },
    },
    similarity_layers={"in": "encoder", "out": "head"},
)

batches = [
    (torch.randn(3, 32, 2), torch.randn(3, 32, 1))
    for _ in range(6)
]

ale = MultiTaskALE(
    model=model,
    dataloader=batches,
    n_tasks=3,
    n_features_out=1,
    num_intervals=10,
)
ale.update()
curves = ale(centered=True, cumulative=True, std=1.0)

similarity = MultitaskSimilarity(ale)
similarity.compute()
scores, task_pairs = similarity.tasks_groups()

print(curves.shape)       # (tasks, features, intervals, x_plus_outputs)
print(similarity.scores)
print(task_pairs)
```

## Notebooks

The notebooks are designed as guided, executable companions to the examples:

- `notebooks/01_quickstart_synthetic.ipynb`: synthetic data and task-first
  dataloading.
- `notebooks/02_task_similarity_matrix.ipynb`: ALE profiles and Frechet task
  similarity matrix.
- `notebooks/03_training_with_alefrechet.ipynb`: training with scheduled ALE and
  similarity-derived task groups.

Install the notebook tools and launch them with:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab notebooks/
```

## Documentation

The user guide and API reference are built with Sphinx:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The generated HTML documentation is written to `docs/_build/html`.

## Testing

Run the full regression suite:

```bash
pytest -q
```

The tests cover:

- dataset and dataloader behavior;
- model layout and task-specific parameter utilities;
- ALE interval indexing and accumulation;
- Frechet task similarity;
- loss and metric helpers;
- tracking and trainer workflows.



## License

ALE-MTL is distributed under the BSD 3-Clause License. See `LICENSE.txt` for the
full license text.

## Reproduce the manuscript

From the source repository or the source archive in the supplementary bundle,
install the publication tools and run:

```bash
python -m pip install -e ".[publication]"
python examples/reproduce_revised_table.py --output results/table.tex
python examples/reproduce_nonlinear_figures.py --output results/nonlinear_figures
python examples/custom_similarity.py --checkpoint results/nonlinear_figures/checkpoint.pt
```

The table uses synthetic data and the Wine and Breast Cancer datasets bundled
with scikit-learn; these workflows require no external dataset download.
For a short pipeline check, add `--smoke` to the table command. Smoke results
are not publication results.

See [the reproduction guide](docs/examples.rst) for settings and reference
runtimes, [the installation guide](docs/installation.rst) for the isolated
wheel workflow, and [the tested requirements](requirements/softwarex.txt) for
the reference environment. The manuscript's supplementary bundle
`alemtl-softwarex-submission.zip` contains the verification report and checksums
for that specific snapshot; subsequent source changes need new verification.

## Contributing

For development, keep changes covered by focused tests in `tests/` and prefer
small examples in `examples/` for user-facing workflows. Before opening a
release candidate for SoftwareX, run:

```bash
pytest -q
sphinx-build -b html docs docs/_build/html
python -m examples.compute_task_similarity
python -m examples.train_alefrechet
```

## Citation

When using ALEMTL, cite the archived software release:

Hidalgo, P., Rodriguez, D., and Domínguez-Díaz, A. (2026).
ALEMTL: A PyTorch package for explainable similarity-driven
multi-task learning, version 0.1.0. Zenodo.
https://doi.org/10.5281/zenodo.[VERSION DOI]

The machine-readable citation is available in `CITATION.cff`.