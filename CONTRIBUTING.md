# Contributing

Thank you for improving this research codebase.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
ruff check .
```

## Style

- Keep core SURE/PURE/conformal routines independent of DeepInv.
- Put DeepInv-specific experiment logic under `src/sscp/experiments/`.
- Avoid absolute paths and machine-specific assumptions.
- Prefer explicit names: use `confidence` for `1-alpha`.
- Do not commit checkpoints, HDF5 datasets, W&B folders, caches, or private paths.

## Pull requests

A good pull request should include:

- a short description of the scientific or engineering change;
- a test or example for any new public function;
- updated docs/configs if the user-facing behaviour changes.
