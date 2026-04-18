# Exp

This repository is organized around one consolidated data-prep area plus lightweight experiment tracking.

Large datasets and generated parquet outputs are intentionally not stored in GitHub history. They should be shared through Google Drive.

Project Drive folder:
`https://drive.google.com/drive/folders/1Ebc_dHoB4G9RiltOJJCII6A_LGEcugAU?usp=drive_link`

## What Is Tracked

- `data_prep/` for data-preparation code, docs, and manifests
- `exp/` for small experiment configs and notes
- small setup and usage files

## What Is Not Tracked

- raw dataset payloads under `data/`
- prepared parquet payloads under `data_prep/runs/`
- large `.parquet` artifacts and intermediate outputs

## Repository Layout

- `data/`: raw inputs only
- `data_prep/`: preprocessing scripts, planning notes, manifests, and prepared dataset runs
- `exp/`: small experiment folders for model runs

## Data Preparation Runs

Use the preprocessing script to create a named prepared dataset run:

- `python3 data_prep/preprocess_experiment_data.py --run-name bert4rec_v1`

That writes:

- `data_prep/runs/bert4rec_v1/`: parquet outputs, report, config, and summary
- `data_prep/runs/bert4rec_v1/manifest.json`: run-local manifest
- `data_prep/manifests/bert4rec_v1.json`: tracked manifest for experiment references

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
   - `python3 data_prep/setup_data_from_drive.py`
4. If `data/` or the legacy prepared snapshot already exist locally and you want to replace them:
   - `python3 data_prep/setup_data_from_drive.py --force`

The Drive bootstrap expects the shared folder to contain `data/` and `processed_final/`.
The downloaded `processed_final/` snapshot is placed under `data_prep/legacy/processed_final/`.
New processing runs should go to `data_prep/runs/`.

## Recommended Repo Usage

- Keep raw data in `data/`.
- Keep all preprocessing code and prep documentation in `data_prep/`.
- Generate named prepared runs into `data_prep/runs/`.
- Track only the small manifest files in `data_prep/manifests/`.
- Keep each model experiment under `exp/<experiment_name>/` and record which prepared run it uses.
