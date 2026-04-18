# Data Prep

This folder contains everything related to data preparation except the raw source payloads in `data/`.

Contents:

- `preprocess_experiment_data.py`: build named prepared-data runs
- `setup_data_from_drive.py`: download raw data and the legacy processed snapshot from Google Drive
- `data_preparing_plan.md`: planning notes for the preprocessing pipeline
- `manifests/`: small tracked manifests for prepared runs
- `runs/`: untracked prepared-data outputs
- `legacy/processed_final/`: untracked imported snapshot from the shared Drive folder

Typical flow:

1. `python3 data_prep/setup_data_from_drive.py`
2. `python3 data_prep/preprocess_experiment_data.py --run-name <run_name>`
3. Point each experiment in `exp/` at `data_prep/manifests/<run_name>.json`
