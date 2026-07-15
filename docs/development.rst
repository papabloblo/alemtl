Development
===========

Local Checks
------------

Run the core test suite:

.. code-block:: bash

   pytest -q

Run representative examples:

.. code-block:: bash

   python -m examples.compute_task_similarity
   python -m examples.train_alefrechet

Build documentation:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Code Organization
-----------------

``src/alemtl/data``
   Dataset and dataloader utilities.

``src/alemtl/models``
   Composable multitask model layers and baseline architectures.

``src/alemtl/similarity``
   ALE profile computation and task similarity metrics.

``src/alemtl/training``
   Loss functions, training orchestration, and tracking utilities.

Release Checklist
-----------------

Before a SoftwareX release candidate:

* run the full test suite;
* build the Sphinx documentation;
* execute the example scripts;
* execute notebooks used in the paper;
* update ``CITATION.cff`` with final authors, repository URL, release DOI, and
  article DOI;
* tag the release and archive it on Zenodo.
