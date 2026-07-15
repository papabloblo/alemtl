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
