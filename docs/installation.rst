Installation
============

ALE-MTL uses a standard ``src/`` package layout and is intended to be installed
from the repository root during development.

Editable Install
----------------

Create and activate a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip

Install the package with publication tooling:

.. code-block:: bash

   pip install -e ".[publication]"

For documentation work, install the documentation extra:

.. code-block:: bash

   pip install -e ".[docs]"

Validation
----------

Run the test suite from the repository root:

.. code-block:: bash

   pytest -q

The current development tree is expected to pass all tests.

Build the documentation locally with:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

or, if ``make`` is available:

.. code-block:: bash

   make -C docs html

Dependency Groups
-----------------

The package metadata defines several optional extras:

``dev``
   Test and packaging helpers used during routine development.

``notebooks``
   Jupyter dependencies for running the tutorial notebooks.

``docs``
   Sphinx dependencies for building this documentation.

``publication``
   Combined tooling for tests, notebooks, documentation, and release checks.
