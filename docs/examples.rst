Examples
========

The ``examples/`` directory contains executable scripts that use synthetic data
and fixed random seeds where practical. Run commands from the repository root
after installing ALE-MTL.

Quickstart Multisine
--------------------

.. code-block:: bash

   python -m examples.quickstart_multisine

Creates a synthetic multitask regression problem and demonstrates the
task-first dataset and dataloader contract.

Multitask Model Layout
----------------------

.. code-block:: bash

   python -m examples.multitask_model_layout

Builds a small :class:`alemtl.models.MultiTaskModel` with hard-shared and
soft-shared layers.

Compute ALE Profiles
--------------------

.. code-block:: bash

   python -m examples.compute_ale_profiles

Computes ALE profiles for a synthetic multitask model and prints the resulting
curve tensor shapes.

Compute Task Similarity
-----------------------

.. code-block:: bash

   python -m examples.compute_task_similarity

Uses ALE profiles to compute a Frechet-based task similarity matrix and nearest
task groups.

Train ALE-Frechet
-----------------

.. code-block:: bash

   python -m examples.train_alefrechet

Runs a compact training workflow with scheduled ALE and similarity updates.

Compare Baselines
-----------------

.. code-block:: bash

   python -m examples.compare_baselines

Checks the forward-pass contracts for the included baseline multitask models.

Compact Reproducible Comparison Table
-------------------------------------

``examples/reproduce_compact_table.py`` runs an offline, CPU-only illustrative
comparison with fixed configurations. It is a new experiment, not a reduced
reproduction of the original six-dataset Table 3. Install the package from the
repository root, then run::

   python -m pip install -e .
   python -m examples.reproduce_compact_table --out-dir results/compact_table

The defaults run 24 fits: two datasets, four methods, and seeds 11, 22, 33.
Each fit runs 60 epochs, using a width-16 tanh network, Adam at 0.003, batch
size 64, and one CPU thread. No hyperparameter search is performed. Validation
MSE on standardized targets selects the checkpoint; the test split is not used
for selection. Runtime depends on hardware and dependency versions.

Datasets
~~~~~~~~

* Multi-Sine: five synthetic regression tasks, 512 samples each, with uniform
  random inputs and Gaussian noise. This is an interpolation example, not the
  chronological forecasting problem in the original benchmark.
* Diabetes: the 442-observation regression dataset distributed with
  ``sklearn.datasets.load_diabetes(scaled=False)``. Four illustrative tasks are
  defined by the two sex codes and age below/at least 50. Sex is excluded from
  predictors. These cohorts demonstrate multitask regression and are not a
  validated clinical model. Dataset background and original references:
  https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

Within each task, a fixed data seed (1729) produces 60/20/20 train/validation/test
splits. Each split is truncated to the smallest task size using its randomized
row order, avoiding repeated evaluation rows. Exact retained row IDs and counts
are recorded. Predictor and target scaling use only retained training rows.
All methods and model seeds share the same data. Seed variability therefore
measures initialization/training variability, not uncertainty over data splits.

Methods and metrics
~~~~~~~~~~~~~~~~~~~

* Single-task: independent task networks.
* Hard sharing: common encoder and task-specific heads.
* Soft sharing: independent networks with the mean L2 distance over all
  unordered parameter-vector pairs as the sharing penalty.
* ALE--Frechet: the same independent networks, with nearest-task penalties
  weighted by mean feature similarity and averaged over tasks. Fresh ALE and
  similarities are computed from training data every five epochs, using eight
  intervals. The base regularization coefficient is 0.01 for both sharing
  penalties. Symmetric nearest-neighbor pairs can appear twice in the directed
  ALE task average.

The table reports macro-average per-task test RMSE in original target units,
sample standard deviation over seeds, and mean training seconds per seed.
Each task's RMSE is computed from all its test observations. Timing includes
training, validation, scheduled ALE/similarity, and the trainer's final test
pass; it excludes setup, artifact writing, and the separate original-unit
prediction audit. Parameter counts are included in the CSV to expose capacity
differences. This small fixed-budget comparison supports a software workflow
demonstration, not a general ranking of algorithms.

Outputs and audit
~~~~~~~~~~~~~~~~~

The output folder contains ``table.tex`` (tabular fragment), ``table.csv``,
``runs.csv``, ``runs.json``, and ``manifest.json``. Each run includes predictions,
targets, a validation-selected checkpoint, and its training log. Dataset arrays,
scalers, split IDs, source hashes, package versions, Git state, and total runtime
are retained for auditing. Preserve the whole directory with the submission.
The LaTeX fragment can be included inside a manuscript ``table`` environment
with a caption describing the protocol above.

Rebuild both table formats from saved runs and verify their RMSE against saved
predictions, without retraining::

   python -m examples.reproduce_compact_table --summarize-only \
       --out-dir results/compact_table

For an installation check only::

   python -m examples.reproduce_compact_table --smoke \
       --out-dir results/compact_smoke

Smoke mode uses two epochs and one seed. Its LaTeX output is explicitly marked
as unsuitable for publication, and a single seed has no reported standard
deviation. Use a new output directory for each experiment; existing nonempty
folders are never overwritten by training. ``--help`` lists adjustable budgets.
Changing them defines a different experiment and is recorded in the manifest.
Reproducibility of numerical values assumes the recorded environment; runtime
and cross-platform floating-point results may vary.


Revised predictive comparison
-----------------------------

The manuscript's ``Similarity-aware predictive modelling`` subsection uses
``examples/reproduce_revised_table.py``. From the repository root, after
installing ALEMTL and its dependencies, run::

   python examples/reproduce_revised_table.py --output results/table.tex

The script writes only a LaTeX table (requires ``booktabs``). Choose a new output
filename for each run. A short installation check is available with ``--smoke``;
its output is explicitly marked as unsuitable for publication.

This four-method experiment runs 300 validation fits and 200 final fits across
five datasets and five 60/20/20 splits. The methods are independent networks,
hard sharing, all-pairs soft sharing and default ALE--Frechet sharing. Linear
variants have 80 or 240 samples per task before splitting; those labels denote
sample counts, not coefficient sparsity. Wine and Wisconsin Diagnostic Breast
Cancer are adapted to regression as described in the manuscript and script.

Training uses a width-32 tanh layer, Adam learning rate 0.003, 120 epochs and
batch size 64. ALE uses eight intervals, unit-variance curve normalization and
updates every five epochs. Regularized methods select strength from
0/0.01/0.1/1/10 on validation data. Split seeds are 1101--1105; the selection
initialization is 1201 and final initializations are 1301 and 1302. The table
reports mean and sample SD across five split means after averaging the two
final initializations. Features and targets are standardized separately for each
task using training observations only. The strength-to-coefficient mapping is
``strength / (tasks * features)`` for ALE and ``strength / unordered_pairs`` for
conventional soft sharing. The penalty selects weights of explicitly soft-shared
modules (parameter names containing ``weight``), excluding biases.

Validation selection within the final script does not erase earlier exploration
of datasets, architectures and settings using predictive results. This is an
exploratory software example, not an independent confirmatory benchmark.
Overlapping splits are not independent datasets. In Breast radius there are only
two tasks, so nearest-peer selection is forced; the example demonstrates
similarity weighting. Cultivar/diagnosis labels are assumed known when predicting
in the adapted real-data examples.

Tested environment
~~~~~~~~~~~~~~~~~~

The reference calculations used Linux, Python 3.12.14, PyTorch 2.14.0+cpu,
NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1. The earlier result-audit tooling
also used pandas 2.2.3; the table-only script does not import pandas. Two worker
processes each use one CPU thread. Other library versions and platforms may
produce floating-point differences.

Release status
~~~~~~~~~~~~~~

The table was verified using the packaged local submission candidate. No public
archived release identifier has yet been assigned to this reproduction script.
Before submission, archive a release containing this script and record its
actual version DOI in ``CITATION.cff`` and the manuscript software citation.
The existing DOI placeholder is not a valid archival reference.

Dataset and software references
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Pedregosa et al. (2011), `Scikit-learn: Machine Learning in Python
  <https://jmlr.org/papers/v12/pedregosa11a.html>`_.
* Aeberhard and Forina (1992), `Wine <https://doi.org/10.24432/C5PC7J>`_,
  UCI Machine Learning Repository.
* Wolberg et al. (1993), `Breast Cancer Wisconsin (Diagnostic)
  <https://doi.org/10.24432/C5DW2B>`_, UCI Machine Learning Repository.


Interpretation figures on the nonlinear comparison dataset
----------------------------------------------------------

Both manuscript interpretation figures now use ``clustered_nonlinear``, the
same six-task dataset as the table's Nonlinear row. With ALEMTL dependencies
and matplotlib installed, run from the repository root::

   python examples/reproduce_nonlinear_figures.py \
       --output results/nonlinear_figures

The script imports the adjacent ``reproduce_revised_table.py`` to reuse its
exact data preparation and model construction. It makes five validation fits
(seed 1201) over strengths 0/0.01/0.1/1/10, then one final fit (seed 1301), using
split seed 1101. It uses the table's default ALE method, not scale-preserving
ALE. No test data are used for selection or plotting; the trainer's evaluation
loader receives validation data for this diagnostic run.

The reference run selected strength 1.0 and checkpoint epoch 71 (zero-based 70).
Its checkpoint parameters exactly match the corresponding saved table run.
After restoring the best-validation checkpoint, ALE and similarity are
recomputed on training inputs. They are not taken from the last scheduled
training update. All five input features are plotted, with centered,
unit-variance ALE curves. Displayed similarities are feature means; the
library uses feature sums for training. This factor of five leaves nearest-peer
rankings unchanged. Known pairs (1,2), (3,4), (5,6) determine display colours only.

The output directory contains:

* ``nonlinear_ale_profiles.pdf``: all five input-feature ALE profiles.
* ``nonlinear_similarity_structure.pdf``: similarity matrix, pair-score
  histogram and directed nearest-neighbour graph.
* ``figure_manifest.json``: configuration, validation scores, zero-based best
  epoch, peer assignments, package versions and reproduction-script hashes.
* ``checkpoint.pt`` and ``explanations.npz``: fitted weights and plotted data.

Choose an empty output directory. The manuscript PDFs are copies of the PDFs
from this reference run; retain the accompanying manifest and tensors when
archiving the release. The tested environment is the one listed above, with
matplotlib 3.11.2. Other platforms or library versions may change numerical
results. The figures illustrate one fixed split and initialization, not a
statistical claim about pair-recovery accuracy.

The earlier Multi-Sine/Electricity script remains available for separate
experiments; its figures are no longer used in these manuscript subsections.

Customizing the curve-comparison function
-----------------------------------------

After generating the nonlinear figures, run::

   python examples/custom_similarity.py --checkpoint results/nonlinear_figures/checkpoint.pt

This reuses the checkpoint, data split, task-wise scaling, ALE intervals and
unit-variance normalization. It supplies a symmetric nearest-point comparison
through ``MultitaskSimilarity(..., similarity_func=nearest_point_similarity)``.
The callback accepts two tensors shaped ``(batch, intervals, 2)`` and returns
one similarity per pair, with larger values meaning more similar curves.
Both the standardized input coordinate and normalized ALE value contribute.
The score is the exponential of the negative average of the two directed
mean nearest-point distances. It ignores traversal order, unlike Frechet.

The library handles feature aggregation and peer assignment. The script prints
nearest peers for both metrics, any changes, and the five feature scores for
Tasks 1 and 2. This lets a user inspect sensitivity to the comparison rule
without retraining. The scores are not calibrated across metrics; an increase
in their numerical value does not demonstrate improved prediction.

For this checkpoint, both metrics retain the same six nearest-peer assignments. For Tasks 1 and 2, both identify feature 4 as the most similar profile and feature 1 as the least similar. The pairing is unchanged under this alternative comparison rule. Both metrics compare the same normalized curves geometrically, so this is a limited sensitivity check rather than evidence of stability across datasets or seeds.

The checkpoint was trained with similarity-aware sharing. The displayed profiles
and pairings illustrate that fitted workflow; they do not independently validate
task-relatedness estimation.

Task-relative input coordinates and unit-variance ALE curves emphasize shape,
not the magnitude of effects in original units. The toy extension is a usage
demonstration, not evidence for preferring the alternative metric in applications.

Submission-candidate verification
---------------------------------

The original package check reused installed dependencies. A subsequent check on
25 September 2026 created a new Python 3.12.3 virtual environment without access
to system site packages, downloaded the pinned dependencies from PyPI and the
PyTorch CPU index without using the pip cache, and installed the distributed
wheel. ``pip check`` reported no broken requirements and all 72 tests passed.
The tests and examples were extracted from the source distribution and executed
without a development ``src/`` directory or ``PYTHONPATH`` override.
Full execution results are recorded in the accompanying verification report.

The supplementary bundle ``alemtl-softwarex-submission.zip`` contains the source
archive, wheel, scripts (inside the source archive), logs, SHA-256 checksums and
verification reports. The bundle must be attached to the submission. It is a
local artifact, not a published GitHub release. The public release and the three
article-specific example URLs returned HTTP 404 when checked on 25 September
2026; upload the verified snapshot before presenting those URLs as available.

``requirements/softwarex.txt`` records the tested direct dependency versions,
including CPU PyTorch. It is not a complete transitive lock file. The verification
report records the installed dependency versions used for the checks.

On an Intel Core i7-8700 CPU, one full table execution took 193.22 seconds
(two workers, one CPU thread each). The complete six-fit figure workflow took
12.07 seconds (one worker, one CPU thread). Times include interpreter startup
and output generation but exclude installation. These are single-run wall times,
not performance benchmarks. All 20 means and standard deviations matched the
manuscript at reported precision; explanation arrays matched the prior figure
run exactly. All 72 tests passed against the installed package.

The full table command is::

   python examples/reproduce_revised_table.py --workers 2 --output results/table.tex

Use a new output path if a result already exists. ``--smoke`` is a reduced
pipeline check and does not reproduce manuscript values. Neither table command
exports any artifact besides the requested LaTeX table.
