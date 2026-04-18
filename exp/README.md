# Experiment: Watch-Time-Aware Sequential Recommendation

End-to-end experiment pipeline for BERT-based sequential recommendation
enriched with watch time and content duration signals.

## Quick Start

```bash
pip install -r requirements.txt

# Create data subset
python scripts/create_subset.py --source-dir ../data_prep/legacy/processed_final --output-dir data --num-users 100000

# Train a model
python scripts/train.py --config configs/variant_a.yaml --seed 42

# Monitor
tensorboard --logdir outputs/tensorboard/

# Run all models
python scripts/train_all.py --seeds 42,123,456

# Compare and test significance
python scripts/compare.py --results-dir outputs/

# Generate report
python scripts/generate_report.py --results-dir outputs/
```

## Structure

- `configs/` — YAML config files for each model
- `src/` — core library (data, models, trainers, evaluation, utils)
- `scripts/` — entry-point scripts
- `outputs/` — checkpoints, logs, TensorBoard (gitignored)
- `data/` — experiment data subset (gitignored)
