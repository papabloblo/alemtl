Reproducibility
===============

ALEMTL provides executable examples, automated tests, pinned reference
dependencies, and reproduction scripts for the workflows reported in the
SoftwareX article.

Software Environment
--------------------

ALEMTL requires Python 3.10 or newer. The package is continuously tested on
Ubuntu with Python 3.10, 3.11, 3.12, and 3.13 through GitHub Actions.

The reference environment used for the numerical SoftwareX examples is recorded
in ``requirements/softwarex.txt``. This file specifies the direct dependency
versions used for the reference calculations, including CPU versions of
PyTorch and torchvision.

Installation and Testing
------------------------

For development and testing, install ALEMTL from the repository root with::

   python -m pip install -e ".[dev]"
   pytest -q

The continuous-integration workflow additionally builds the source distribution
and wheel, checks their package metadata, installs the wheel outside the source
tree, runs the test suite against the installed package, executes representative
reproduction workflows, and builds the Sphinx documentation with warnings
treated as errors.

Reproducing the SoftwareX Examples
----------------------------------

The main numerical comparison reported in the article can be reproduced with::

   python examples/reproduce_revised_table.py 
       --workers 2 
       --output results/table.tex

The nonlinear ALE and task-similarity figures can be generated with::

   python examples/reproduce_nonlinear_figures.py 
       --output results/nonlinear_figures

The alternative curve-comparison example can then be executed with::

   python examples/custom_similarity.py 
       --checkpoint results/nonlinear_figures/checkpoint.pt

Detailed experimental settings, datasets, random seeds, validation procedures,
and reference runtimes are documented in ``docs/examples.rst``.

Software Artifacts
------------------

The repository contains:

* the Python package under ``src/alemtl``;
* the BSD-3-Clause license in ``LICENSE.txt``;
* package and dependency metadata in ``pyproject.toml``;
* machine-readable citation metadata in ``CITATION.cff``;
* executable examples under ``examples``;
* tutorial notebooks under ``notebooks``;
* automated regression tests under ``tests``;
* Sphinx user and API documentation under ``docs``;
* the tested SoftwareX dependency specification in
  ``requirements/softwarex.txt``.

Release and Archival Record
---------------------------

The SoftwareX submission corresponds to ALEMTL version 0.1.0. The source
repository is available at:

https://github.com/papabloblo/alemtl

The versioned GitHub release and permanent archival DOI identify the exact
software snapshot associated with the article. The archival DOI should be used
when citing the software once the v0.1.0 release has been deposited.

Reproducibility Scope
---------------------

The supplied workflows are intended to reproduce the numerical tables and
illustrative figures reported in the SoftwareX article. Exact floating-point
values and execution times may vary across operating systems, hardware, and
dependency versions. The tested environment and experimental configuration are
therefore recorded explicitly to make such differences auditable.