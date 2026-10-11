# Data

The repository intentionally includes both processed CSVs and frozen ESM-2 arrays so that a fresh clone can run external validation without downloading ESM-2.

- `training/`: 7,092 allele-balanced pMHC-II observations and matching frozen ESM-2 arrays.
- `external/`: 309  external/test observations and matching frozen ESM-2 arrays.

Do not reorder either CSV unless the corresponding ESM feature arrays and sample-ID manifest are regenerated. `scripts/verify_repository.py` checks row alignment.
