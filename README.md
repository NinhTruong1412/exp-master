# Exp

This repository keeps the code, documentation, and lightweight metadata for the experiment workflow.

Large datasets and generated parquet outputs are intentionally not stored in GitHub history. They should be shared through Google Drive.

## What Is Tracked

- `scripts/`
- project docs such as `data_preparing_plan.md`
- small setup and usage files

## What Is Not Tracked

- `data/`
- `processed_final/`
- large `.parquet` artifacts and intermediate outputs

## Google Drive Workflow

1. Create a Google Drive folder for this project.
2. Upload the contents of `data/` and `processed_final/` there.
3. Share the folder or a read-only link with collaborators.
4. Keep the Drive link updated in the two README files below:
   - `data/README.md`
   - `processed_final/README.md`

## Recommended Repo Usage

- Use GitHub for code, docs, and pipeline definitions.
- Use Google Drive for raw and processed data.
- If data changes, update the Drive files and note the change in a commit message or changelog entry.
