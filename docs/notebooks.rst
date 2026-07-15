Notebooks
=========

The notebooks provide guided, executable workflows that complement the example
scripts.

Available Notebooks
-------------------

``notebooks/01_quickstart_synthetic.ipynb``
   Synthetic data generation, task-first tensors, and dataloading.

``notebooks/02_task_similarity_matrix.ipynb``
   ALE computation and Frechet task similarity matrix construction.

``notebooks/03_training_with_alefrechet.ipynb``
   Training with scheduled ALE recomputation and similarity-derived task groups.

Launch
------

Install notebook dependencies and start JupyterLab:

.. code-block:: bash

   pip install -e ".[notebooks]"
   jupyter lab notebooks/

Publication Use
---------------

For a SoftwareX submission, execute notebooks from a clean environment before
release and preserve the commands, version, and random seeds used to regenerate
the figures or tables included in the paper.
