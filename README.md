# ALEMTL: Accumulated Local Effects for Multi-Task Learning

ALEMTL is an open-source Python/PyTorch library for explainable
similarity-driven multi-task learning. It computes task-wise Accumulated
Local Effects (ALE) profiles, derives interpretable task similarities from
their functional behavior, and can use these similarities to guide soft
parameter sharing during training.

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
  regularization, ALE/similarity scheduling, early stopping, checkpointing, and task-wise metric tracking.
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

ALEMTL requires Python >= 3.10.

Install the stable v0.1.0 release directly from GitHub:

```bash
pip install "alemtl @ git+https://github.com/papabloblo/alemtl.git@v0.1.0"
```

For development and testing, install the test tools with an editable install:

```bash
git clone https://github.com/papabloblo/alemtl.git
cd alemtl
python -m pip install -e ".[dev]"
pytest -q
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

## Reproducing the SoftwareX examples

The numerical table and illustrative figures reported in the SoftwareX
article can be regenerated from the repository:

```bash
python -m pip install -e ".[publication]"

python examples/reproduce_revised_table.py \
    --output results/table.tex

python examples/reproduce_nonlinear_figures.py \
    --output results/nonlinear_figures

python examples/custom_similarity.py \
    --checkpoint results/nonlinear_figures/checkpoint.pt
```

The examples use synthetic datasets together with the Wine and Breast
Cancer datasets distributed through scikit-learn and therefore require no
additional dataset downloads.

See `docs/examples.rst` for the complete reproduction protocol and reference
runtimes, and `requirements/softwarex.txt` for the tested SoftwareX
environment.

## Release and reproducibility

The SoftwareX article corresponds to ALEMTL v0.1.0.

- Source release: https://github.com/papabloblo/alemtl/releases/tag/v0.1.0
- Archived release: https://doi.org/<VERSION_DOI>
- Documentation: https://github.com/papabloblo/alemtl/tree/v0.1.0/docs
- Python: >= 3.10
- Tested in CI: Python 3.10, 3.11, 3.12, and 3.13

The supplementary SoftwareX archive contains the source distribution,
wheel, verification logs, generated outputs, and checksums corresponding
to the archived release.

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