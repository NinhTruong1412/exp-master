# Prepared Data

Prepared datasets live under `prepared_data/runs/<run_name>/`.

Each run contains the generated parquet files, report, feature config, processing summary, and a local `manifest.json`.

Small tracked manifests are also written to:

- `prepared_data/manifests/<run_name>.json`

Recommended usage:

- create a named run with `python3 scripts/preprocess_experiment_data.py --run-name <run_name>`
- reference that run from a folder under `exp/`
- keep the heavy parquet outputs out of git
