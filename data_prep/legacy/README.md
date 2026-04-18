# Legacy Processed Snapshot

The shared Google Drive folder still contains a legacy prepared-data snapshot named `processed_final/`.

When you run `python3 data_prep/setup_data_from_drive.py`, that snapshot is placed here:

- `data_prep/legacy/processed_final/`

Use it only for reference or backward compatibility. New prepared datasets should be generated into:

- `data_prep/runs/<run_name>/`
