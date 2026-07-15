SoftwareX Dataset Preparation
=============================

The SoftwareX illustrative benchmark uses three public real datasets in
addition to the synthetic Multi-Sine and Polynomial generators and the
Electricity preparation routine embedded in the reproduction script. The
real-data workflow is provided by ``scripts/prepare_real_datasets.py`` and is
adapted from the original ALE--Fréchet MTL reproducibility repository.

The script downloads raw data, creates leakage-safe supervised panels, performs
chronological train/validation/test splits independently for every task, fits
predictor standardization statistics on training rows only, and writes CSV
files compatible with
``examples/reproduce_softwarex_illustrative_examples.py``.

Installation
------------

The dataset workflow uses the standard ALEMTL dependencies. For the complete
paper workflow, install the publication extras from the repository root::

   pip install -e ".[publication]"

Output hierarchy
----------------

With the default ``--data-root data`` option, files are written to::

   data/
   ├── raw/<dataset>/
   └── interim/<dataset>/
       ├── <dataset>_full.csv
       ├── <dataset>_train.csv
       ├── <dataset>_val.csv
       ├── <dataset>_test.csv
       ├── <dataset>_meta.csv
       ├── <dataset>_scaler_stats.csv
       ├── <dataset>_preprocessing_manifest.json
       └── by_task/

The preprocessing manifest records the source URL, SHA-256 digest of the local
raw file, generation timestamp, parameters, task count, and split row counts.
This allows a reproduced run to document the exact downloaded input. Raw data
are cached locally and are not included in ALEMTL release archives.

Exchange
--------

Source
   European Central Bank historical euro foreign-exchange reference rates:
   ``https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip``.

Default tasks
   USD, GBP, JPY, CHF, AUD, CAD, and CNY.

Processing
   The currencies are reshaped into a task-indexed daily panel. The generator
   adds day, month, year, month-boundary indicators, log returns, and trailing
   return volatility. It then creates one-step-ahead targets, response lags at
   1, 5, and 21 observations, covariate lags at 1 and 5 observations, and
   rolling response statistics over 5 and 21 observations.

Command::

   python scripts/prepare_real_datasets.py --dataset exchange

A different currency set can be selected with, for example::

   python scripts/prepare_real_datasets.py --dataset exchange \
       --currencies USD,GBP,JPY

The ECB archive is updated over time. Retain the generated preprocessing
manifest and cached raw ZIP for an exact local rerun of a particular download.

METR-LA
-------

Source
   The ``METR-LA.csv`` file in Zenodo record 5146275. The script resolves the
   file through ``https://zenodo.org/api/records/5146275``.

Tasks
   Traffic sensors represented by the columns of the five-minute wide CSV.

Processing
   The wide sensor table is converted to a long task panel. Calendar features
   and past speed differences are added. The forecast horizon is 12 five-minute
   observations (approximately one hour), with response lags at 1, 12, and 288
   observations, covariate lags at 1 and 12 observations, and rolling response
   statistics over 12 and 288 observations. If a compatible
   ``data/raw/metrla/adj_mx.pkl`` is supplied, node degree is added as an
   optional predictor.

Command::

   python scripts/prepare_real_datasets.py --dataset metrla

NN5 Daily
---------

Source
   Monash Forecasting Repository NN5 Daily dataset without missing values,
   archived in Zenodo record 4656117:
   ``https://zenodo.org/records/4656117/files/nn5_daily_dataset_without_missing_values.zip?download=1``.

Tasks
   Daily ATM-withdrawal series contained in the Monash ``.ts``/``.tsf`` file.

Processing
   The generator parses the forecasting file into a long panel, adds daily and
   cyclical calendar variables, and constructs one-day-ahead targets. Response
   lags are 1, 7, 14, and 28 days; covariate lags are 1 and 7 days; rolling
   response statistics use 7- and 28-day windows.

Command::

   python scripts/prepare_real_datasets.py --dataset nn5

Complete preparation and benchmark
----------------------------------

Prepare all three datasets::

   python scripts/prepare_real_datasets.py --dataset all --data-root data

Then run the benchmark using the generated hierarchy::

   python examples/reproduce_softwarex_illustrative_examples.py \
       --section benchmark --data-root data/interim

Alternatively, request data preparation from the benchmark command itself::

   python examples/reproduce_softwarex_illustrative_examples.py \
       --section benchmark --data-root data/interim \
       --prepare-real-datasets --prepare-electricity

``--max-tasks``, ``--benchmark-max-tasks``, and ``--smoke`` are intended only
for diagnostics. They change the task set or training procedure and must be
omitted when reproducing the paper-level experiments.

Licensing and redistribution
----------------------------

ALEMTL distributes the preparation code, not the datasets. Users are
responsible for reviewing and complying with the terms attached to each source.
The adaptation of the original generator retains its MIT notice in
``THIRD_PARTY_NOTICES.md``.
