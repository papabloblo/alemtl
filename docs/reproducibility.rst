SoftwareX Preparation
=====================

ALE-MTL is being prepared for publication as a SoftwareX software paper. This
page tracks repository-level artifacts that reviewers commonly need.

Current Status
--------------

Implemented artifacts:

* source package under ``src/alemtl``;
* BSD-3-Clause ``LICENSE`` file;
* ``pyproject.toml`` with package metadata and optional extras;
* ``CITATION.cff`` with provisional software citation metadata;
* executable examples;
* tutorial notebooks;
* pytest regression suite;
* Sphinx documentation.

Release-Time Fields
-------------------

Before submission, finalize:

* public repository URL;
* formal author list and affiliations;
* release version;
* Zenodo or equivalent software archive DOI;
* SoftwareX article DOI, once available;
* manuscript metadata table in ``paper/softwarex-osp-template.tex``.

Suggested SoftwareX Paper Structure
-----------------------------------

The paper should make clear:

* the scientific problem addressed by ALE-MTL;
* the software architecture and task-first tensor contract;
* how ALE profiles are computed and reused for task similarity;
* how similarity can affect multitask training;
* which examples and notebooks reproduce the reported workflows;
* how the software can be reused outside the paper experiments.
