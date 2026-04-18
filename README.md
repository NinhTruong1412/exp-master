# Exp

This repository is organized around reproducible data preparation and lightweight experiment tracking.

Large datasets and generated parquet outputs are intentionally not stored in GitHub history. They should be shared through Google Drive.

Project Drive folder:
`https://drive.google.com/drive/folders/1Ebc_dHoB4G9RiltOJJCII6A_LGEcugAU?usp=drive_link`

## What Is Tracked

- `scripts/`
- `exp/` for small experiment configs and notes
- `prepared_data/manifests/` for prepared-dataset manifests
- small setup and usage files

## What Is Not Tracked

- raw dataset payloads under `data/`
- prepared parquet payloads under `prepared_data/runs/`
- large `.parquet` artifacts and intermediate outputs

## Repository Layout

- `data/`: raw inputs and data-preparation planning notes
- `prepared_data/`: prepared dataset runs and tracked manifests
- `exp/`: small experiment folders for model runs
- `scripts/`: bootstrap and preprocessing scripts

## Data Preparation Runs

Use the preprocessing script to create a named prepared dataset run:

- `python3 scripts/preprocess_experiment_data.py --run-name bert4rec_v1`

That writes:

- `prepared_data/runs/bert4rec_v1/`: parquet outputs, report, config, and summary
- `prepared_data/runs/bert4rec_v1/manifest.json`: run-local manifest
- `prepared_data/manifests/bert4rec_v1.json`: tracked manifest for experiment references

This makes it easy for each folder under `exp/` to reference a specific prepared dataset by manifest name.

## Google Drive Workflow

1. Create a Google Drive folder for this project.
2. Upload the contents of `data/` and `processed_final/` there.
3. Share the folder or a read-only link with collaborators.
4. Keep the Drive link updated in the two README files below:
   - `data/README.md`
   - `processed_final/README.md`

## Bootstrap On A New Machine

1. Clone the repository.
2. Install `gdown`:
   - `python3 -m pip install gdown`
3. Download the large data into the repo:
   - `python3 scripts/setup_data_from_drive.py`
4. If `data/` or `processed_final/` already exist locally and you want to replace them:
   - `python3 scripts/setup_data_from_drive.py --force`

The script expects the shared Google Drive folder to contain `data/` and `processed_final/`.
`processed_final/` is treated as a legacy downloaded snapshot; new processing runs should go to `prepared_data/runs/`.

## Recommended Repo Usage

- Keep raw data in `data/`.
- Generate named prepared runs into `prepared_data/runs/`.
- Track only the small manifest files in `prepared_data/manifests/`.
- Keep each model experiment under `exp/<experiment_name>/` and record which prepared run it uses.
