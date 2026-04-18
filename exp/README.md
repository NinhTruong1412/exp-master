# Experiments

Use this directory for small, tracked experiment folders.

Recommended structure:

- `exp/<experiment_name>/config.json`
- `exp/<experiment_name>/notes.md`
- `exp/<experiment_name>/results.json`

Each experiment should record the prepared dataset manifest it depends on, for example:

- `prepared_run`: `bert4rec_v1`
- `prepared_manifest`: `data_prep/manifests/bert4rec_v1.json`
