# Generated research data

This directory is intentionally empty in release archives. Create the real
SoftwareX benchmark datasets with:

```bash
python scripts/prepare_real_datasets.py --dataset all --data-root data
```

The command caches source files under `data/raw/` and writes processed CSV
panels under `data/interim/`. Both directories are ignored by Git because the
raw datasets may be large and remain governed by their source terms.

See `docs/datasets.rst` for sources, transformations, output schemas, and the
complete reproduction command.
