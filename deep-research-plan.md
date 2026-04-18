# Deep Research Plan: BERT-Based Sequential Recommendation with Watch Time & Duration

## 1. Problem Statement & Objective

Build and evaluate an end-to-end experiment pipeline for **watch-time-aware sequential recommendation** using BERT-like encoders on proprietary VOD/OTT platform data. The core hypothesis is that enriching the sequential model with duration-normalized engagement signals (watch ratio, duration buckets, time gaps) and multi-task training produces statistically significant ranking improvements over standard BERT4Rec and SASRec baselines.

### Data Reality Check
- **Full dataset**: 569K users, 3.7M events, 2,722 items (movies, series, shows, events, sports)
- **Watch ratio distribution**: Highly skewed — median 0.012, mean 0.158, P95 0.921
- **Sequence lengths**: Median 4, mean 6.5, P95 19 — very sparse histories
- **Experiment subset**: ~100K users sampled from users with seq_len ≥ 5 for meaningful sequence modeling

### Success Criteria
1. Variant A (Multi-Task Duration-Aware BERT) achieves ≥ +3% NDCG@10 lift over vanilla BERT4Rec
2. Improvements are statistically significant (p < 0.05, paired bootstrap test)
3. Duration-stratified metrics show improvement is not purely from long-content bias
4. At least one debiasing variant (B or C) shows measurable improvement on duration-fair metrics

---

## 2. Data Preparation for Experiments

### 2.1 Leverage Existing Pipeline

The preprocessing pipeline in `data_prep/preprocess_experiment_data.py` is complete and produces model-ready outputs in `data_prep/legacy/processed_final/`. The key artifacts are:

| Artifact | Description |
|---|---|
| `interactions_enriched.parquet` | Flat event table with watch_ratio, duration buckets, flags |
| `item_vocab.parquet` | content_id → integer item_id mapping (2,722 items) |
| `user_sequences.parquet` | Per-user ordered sequences with aligned auxiliary features |
| `train.parquet` | Training prefixes (leave-last-two split) |
| `valid.parquet` | Validation: predict second-to-last item |
| `test.parquet` | Test: predict last item |
| `feature_config.json` | Schema and feature usage specification |

### 2.2 Experiment Subset Creation

Create a sampling script (`exp/scripts/create_subset.py`) that:

1. **Loads** `user_sequences.parquet` from `data_prep/legacy/processed_final/`
2. **Filters** users with `seq_len >= 5` (ensures enough context for transformer models)
3. **Stratified samples** ~100K users, stratified by:
   - Sequence length bins (5-7, 8-12, 13-19, 20+)
   - Watch ratio distribution quartiles (to preserve engagement diversity)
4. **Produces** subset parquets: `exp/data/subset_train.parquet`, `subset_valid.parquet`, `subset_test.parquet`
5. **Validates**:
   - No user appears in multiple splits
   - All items in valid/test exist in train
   - Sequence ordering is strictly chronological
   - Prints distribution statistics for verification

### 2.3 Experiment Dataset Class

Implement `exp/src/data/dataset.py` with:

- `SequentialRecDataset(torch.utils.data.Dataset)`: loads subset parquets, handles padding/truncation to max_seq_len=50, returns tensors for item_ids, watch_ratios, watch_buckets, duration_buckets, positions, attention_masks
- `get_dataloaders(config)`: returns train/valid/test DataLoaders with configurable batch_size, num_workers
- **Negative sampling**: full-softmax for BERT4Rec-style (masked item prediction), sampled negatives for SASRec-style (next-item BCE). Both strategies in one flexible dataset
- **Masking logic**: standard BERT4Rec random masking (15-20%) with special handling:
  - Engagement-weighted masking (bias toward high watch-ratio positions) as a toggleable option for Variant A

---

## 3. Model Architecture

### 3.1 Baselines

#### 3.1.1 Popularity Baseline (`exp/src/models/pop.py`)
- Rank items by frequency in training data
- No learning — serves as sanity check
- Duration-stratified variant: rank by frequency within duration bucket

#### 3.1.2 SASRec (`exp/src/models/sasrec.py`)
- Causal (left-to-right) self-attention over item sequences
- Next-item prediction with BCE loss + negative sampling
- **Matched-loss variant**: also train with cross-entropy softmax to ensure fair comparison with BERT4Rec (per Petrov & Macdonald, RecSys 2023)
- Hyperparameters: hidden=128, layers=2, heads=2, dropout=0.2

#### 3.1.3 BERT4Rec (`exp/src/models/bert4rec.py`)
- Bidirectional Transformer with masked item prediction
- Full-softmax cross-entropy loss
- Standard BERT-style random masking at 15%
- Hyperparameters: hidden=128, layers=2, heads=2, dropout=0.2, mask_ratio=0.15

### 3.2 Proposed Variants

#### 3.2.1 Variant A — Multi-Task Duration-Aware BERT4Rec (`exp/src/models/variant_a.py`)

**Token Representation (enriched):**
```
x_t = e(item_id) + e(position) + e(time_gap_bucket) + W_d·φ(log(1+duration)) 
    + e(duration_bucket) + W_r·φ(watch_ratio_clipped) + e(watch_bucket) + e(flags)
```

**Attention Modification:**
```
a_ij = (Q_i · K_j^T) / √h + b_gap(|τ_i - τ_j|) + b_dur(d_j)
```
- Relative time-gap bias: learned embedding per gap bucket added to attention logits
- Duration-conditioned bias: learned per-duration-bucket bias added to attention logits

**Multi-Task Loss:**
```
L = L_mask + λ_r·L_ratio + λ_p·L_pair + λ_c·L_contrastive
```
- `L_mask`: masked item prediction (cross-entropy, same as BERT4Rec)
- `L_ratio`: watch-ratio regression head (Huber loss on masked positions)
- `L_pair`: pairwise BPR loss weighted by debiased watch-ratio difference
- `L_contrastive`: InfoNCE on augmented sequence views (crop + mask)

**Default λ values:** λ_r=0.3, λ_p=0.2, λ_c=0.1

#### 3.2.2 Variant B — Counterfactual Watch BERT (`exp/src/models/variant_b.py`)

**Extends Variant A with:**
- **Duration-stratified quantile targets (D2Q-style)**: for each masked position, instead of raw watch ratio, the regression target is the within-duration-bucket quantile of the user's watch ratio. This removes duration as a confounding shortcut.
- **Adversarial duration discriminator**: a small MLP head that tries to predict the duration bucket from the sequence encoder output. Its gradient is reversed at the encoder (gradient reversal layer), discouraging the representation from encoding pure duration shortcuts.
- **Ordinal classification head**: instead of continuous regression, predicts an ordinal watch-depth class (skip / glance / partial / deep / complete).

**Loss:**
```
L = L_mask + λ_q·L_ordinal + λ_cf·L_d2q + λ_adv·(-L_dur_disc) + λ_c·L_contrastive
```

**Default λ values:** λ_q=0.3, λ_cf=0.2, λ_adv=0.02, λ_c=0.1

#### 3.2.3 Variant C — Calibrated Distributional BERT (`exp/src/models/variant_c.py`)

**Extends Variant A with:**
- **Multi-quantile prediction head**: predicts watch-ratio quantiles at τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9} using pinball loss. Enables uncertainty estimation.
- **Prototype calibration**: K learned prototypes per duration bucket; each masked position's representation is matched to prototypes via soft assignment, with a compactness loss ensuring prototypes represent distinct engagement patterns.
- **Calibration loss**: ECE-style calibration penalty ensuring predicted quantiles match empirical coverage.

**Loss:**
```
L = L_mask + λ_q·L_pinball + λ_cal·L_calibration + λ_c·L_contrastive
```

**Default λ values:** λ_q=0.4, λ_cal=0.1, λ_c=0.1

---

## 4. Training Infrastructure

### 4.1 Hyperparameter Configuration (`exp/configs/`)

Each model gets a YAML config file:

```yaml
# Example: exp/configs/variant_a.yaml
model:
  name: variant_a
  hidden_size: 128
  num_layers: 2
  num_heads: 2
  dropout: 0.2
  max_seq_len: 50
  mask_ratio: 0.15

data:
  subset_dir: exp/data/
  batch_size: 256
  num_workers: 4

training:
  epochs: 100
  lr: 3e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  early_stopping_patience: 10
  early_stopping_metric: ndcg@10
  gradient_accumulation_steps: 1
  fp16: true
  seed: 42

loss_weights:
  lambda_ratio: 0.3
  lambda_pair: 0.2
  lambda_contrastive: 0.1

evaluation:
  metrics: [hr@5, hr@10, hr@20, ndcg@5, ndcg@10, ndcg@20, mrr]
  eval_every_n_epochs: 1
  full_sort: true  # use all items as candidates, not sampled

logging:
  tensorboard_dir: exp/outputs/tensorboard/
  checkpoint_dir: exp/outputs/checkpoints/
  log_every_n_steps: 50
```

### 4.2 GPU Optimization Strategy (24GB)

| Technique | Setting | Rationale |
|---|---|---|
| Mixed precision | FP16 via `torch.cuda.amp` | ~50% memory reduction, ~1.5x speedup |
| Batch size | 256 (with gradient accumulation if needed) | Fits in 24GB with hidden=128 |
| Sequence length | Max 50 (P95 is 19, no data lost) | Minimal padding waste |
| Hidden size | 128 (not 256) | Sufficient for 2.7K item vocab |
| Layers | 2 (not 4) | Diminishing returns on short sequences |
| DataLoader | pin_memory=True, num_workers=4 | Eliminate CPU-GPU transfer bottleneck |
| Gradient checkpointing | Optional, enable if OOM | Trades compute for memory |
| Full-softmax eval | Batched candidate scoring | Avoid OOM during evaluation |

**Estimated training time per model**: ~15-30 min per epoch for 100K users, ~2-5 hours total for 100 epochs with early stopping.

### 4.3 Training Loop (`exp/src/trainers/trainer.py`)

```
Trainer class with:
- Mixed-precision training (AMP scaler)
- Learning rate warmup + cosine decay
- Early stopping on validation NDCG@10
- TensorBoard logging of all losses, metrics, and learning rates
- Checkpoint save/load (best model + latest model)
- Reproducible seeding
- Progress bars (tqdm) with loss display
- Per-epoch evaluation on validation set
- Final evaluation on test set with best checkpoint
```

### 4.4 Real-Time Monitoring (`exp/src/utils/logging_utils.py`)

**TensorBoard Integration:**
- `scalars/train/loss_total`, `loss_mask`, `loss_ratio`, `loss_pair`, `loss_contrastive`
- `scalars/train/lr`, `grad_norm`
- `scalars/valid/hr@5`, `hr@10`, `ndcg@5`, `ndcg@10`, `mrr`
- `scalars/valid/watch_ratio_mae`, `watch_ratio_rmse` (for engagement heads)
- `scalars/valid/duration_stratified_ndcg@10/{bucket}` (per-bucket metrics)
- `hparams/` — hyperparameter logging for comparison across runs
- `text/` — config dump and data statistics

**Console Logging:**
- Per-step: loss components, learning rate, throughput (samples/sec)
- Per-epoch: all metrics, best metric so far, early stopping counter
- End of training: summary table of all models and metrics

**JSON Run Log:**
- `exp/outputs/{model_name}_{seed}/run_log.json`: complete training history, config, and final results
- Machine-readable for automated report generation

---

## 5. Evaluation Protocol

### 5.1 Metrics Suite

| Category | Metrics | Purpose |
|---|---|---|
| Ranking | HR@{5,10,20}, NDCG@{5,10,20}, MRR | Standard next-item accuracy |
| Engagement | Watch-ratio MAE, RMSE (for aux heads) | Engagement prediction quality |
| Duration-fairness | NDCG@10 stratified by duration bucket | Detect long-content bias |
| Watch-time-gain | WTG@10 (duration-debiased ranking) | Bias-corrected ranking quality |

### 5.2 Evaluation Procedure

- **Full-sort evaluation**: rank ALL items for each test user (no sampled metric), batched for GPU efficiency
- **Consistent protocol**: all models evaluated with identical code path
- **Duration-stratified**: break down every metric by content_duration_bucket (<5m, 5-15m, 15-30m, 30-60m, 60-120m, 120m+)
- **User-activity-stratified**: break down by user sequence length quartile

### 5.3 Statistical Significance Testing (`exp/src/utils/significance.py`)

1. **Paired bootstrap test** (primary): resample per-user metric vectors 10,000 times, compute p-value for each model pair
2. **Wilcoxon signed-rank test** (secondary): non-parametric paired test on per-user NDCG@10
3. **95% confidence intervals**: reported for all main metrics
4. **Multiple comparison correction**: Bonferroni correction when comparing all model pairs
5. **Multi-seed runs**: each model trained with 3 seeds (42, 123, 456) — report mean ± std across seeds

### 5.4 Ablation Design

Starting from vanilla BERT4Rec, add components incrementally:

| Ablation ID | What's Added | Purpose |
|---|---|---|
| A0 | BERT4Rec (baseline) | Floor |
| A1 | + duration/watch-ratio input features | Value of enriched tokens |
| A2 | + time-gap attention bias | Value of temporal bias |
| A3 | + duration-conditioned attention bias | Value of duration bias |
| A4 | + watch-ratio regression head (L_ratio) | Value of engagement MTL |
| A5 | + pairwise ranking loss (L_pair) | Value of ranking objective |
| A6 | + contrastive loss (L_contrastive) | Value of contrastive regularization |
| A7 | Full Variant A (= A6) | Complete model |
| B1 | + D2Q duration-stratified targets | Value of debiased labels |
| B2 | + adversarial duration discriminator | Value of representation debiasing |
| B3 | Full Variant B | Complete debiasing model |
| C1 | + multi-quantile head | Value of distributional prediction |
| C2 | + prototype calibration | Value of calibration |
| C3 | Full Variant C | Complete calibration model |

---

## 6. Code Structure

```
exp/
├── configs/
│   ├── pop.yaml
│   ├── sasrec.yaml
│   ├── bert4rec.yaml
│   ├── variant_a.yaml
│   ├── variant_b.yaml
│   └── variant_c.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # SequentialRecDataset, masking, negative sampling
│   │   └── sampler.py           # Stratified user sampling, augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base model with common interface
│   │   ├── pop.py               # Popularity baseline
│   │   ├── sasrec.py            # SASRec baseline
│   │   ├── bert4rec.py          # BERT4Rec baseline
│   │   ├── variant_a.py         # Multi-Task Duration-Aware BERT4Rec
│   │   ├── variant_b.py         # Counterfactual Watch BERT
│   │   ├── variant_c.py         # Calibrated Distributional BERT
│   │   └── modules/
│   │       ├── __init__.py
│   │       ├── embeddings.py    # Item, position, time-gap, duration, watch embeddings
│   │       ├── attention.py     # Standard + duration-biased multi-head attention
│   │       ├── heads.py         # Regression, ordinal, quantile, adversarial heads
│   │       └── losses.py        # All loss functions (mask CE, Huber, BPR, InfoNCE, pinball, GRL)
│   ├── trainers/
│   │   ├── __init__.py
│   │   └── trainer.py           # Unified training loop for all models
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # HR, NDCG, MRR, MAE, RMSE, WTG
│   │   ├── evaluator.py         # Full-sort evaluation, stratified metrics
│   │   └── significance.py      # Bootstrap, Wilcoxon, confidence intervals
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # YAML config loader/validator
│       ├── logging_utils.py     # TensorBoard writer, console logger, JSON logger
│       ├── seed.py              # Reproducible seeding (torch, numpy, python)
│       └── device.py            # GPU detection, memory monitoring
├── scripts/
│   ├── create_subset.py         # Sample 100K users from processed data
│   ├── train.py                 # Entry: train a single model from config
│   ├── train_all.py             # Entry: train all models sequentially
│   ├── evaluate.py              # Entry: evaluate a trained model
│   ├── ablation.py              # Entry: run ablation study
│   ├── compare.py               # Entry: cross-model comparison + significance tests
│   └── generate_report.py       # Entry: produce final markdown report
├── outputs/                     # .gitignored — checkpoints, logs, results
│   └── tensorboard/
├── requirements.txt
└── README.md
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Infrastructure + Data)
1. Set up `exp/` directory structure, requirements.txt, README
2. Implement config system (`utils/config.py`, `utils/seed.py`, `utils/device.py`)
3. Implement data loading and subsetting (`scripts/create_subset.py`, `data/dataset.py`)
4. Implement logging infrastructure (`utils/logging_utils.py`, TensorBoard setup)
5. Implement evaluation metrics (`evaluation/metrics.py`, `evaluation/evaluator.py`)
6. Implement base training loop (`trainers/trainer.py`)
7. **Validation**: verify data loading, check shapes, run dummy training step

### Phase 2: Baselines
8. Implement Popularity baseline (`models/pop.py`)
9. Implement SASRec (`models/sasrec.py`)
10. Implement BERT4Rec (`models/bert4rec.py`)
11. Train and evaluate all baselines (3 seeds each)
12. **Validation**: check that BERT4Rec ≥ SASRec ≥ Pop on NDCG@10 (expected ordering)

### Phase 3: Variant A (Primary Proposed Model)
13. Implement enriched embeddings (`modules/embeddings.py`)
14. Implement duration-biased attention (`modules/attention.py`)
15. Implement auxiliary heads: regression, pairwise, contrastive (`modules/heads.py`, `modules/losses.py`)
16. Implement Variant A model (`models/variant_a.py`)
17. Train and evaluate Variant A (3 seeds)
18. Run ablation study A0→A7
19. **Validation**: check Variant A > BERT4Rec, inspect duration-stratified metrics

### Phase 4: Variants B & C
20. Implement D2Q duration-stratified targets and gradient reversal layer
21. Implement Variant B model (`models/variant_b.py`)
22. Implement multi-quantile head and prototype calibration
23. Implement Variant C model (`models/variant_c.py`)
24. Train and evaluate Variants B & C (3 seeds each)
25. **Validation**: check B improves on duration-stratified metrics, C improves calibration

### Phase 5: Analysis & Report
26. Run statistical significance tests across all model pairs
27. Generate duration-stratified comparison plots
28. Run full ablation comparison
29. Generate final report (`scripts/generate_report.py`)
30. Write conclusions and recommendations

---

## 8. Experimental Pitfalls to Guard Against

| Pitfall | Mitigation |
|---|---|
| **Objective mismatch** (SASRec BCE vs BERT4Rec CE) | Train SASRec with both BCE and matched CE; compare fairly |
| **Label leakage** (future watch data in features) | Feature checklist: never use target item's watch_time as input; use only training-set item stats |
| **Duration overfitting** (model learns "longer = more watch time") | Duration-stratified metrics + Variant B adversarial debiasing |
| **Random seed variance** | 3 seeds per model, report mean ± std, paired significance tests |
| **Evaluation protocol differences** | ALL models use identical full-sort evaluation code |
| **Low watch-ratio floor** (median 0.012) | Keep zero/short-watch events; test multiple target transformations |
| **Short sequences** (median 4) | Sample users with seq_len ≥ 5; test short vs. long user cohorts separately |
| **Overfitting on small vocab** (2,722 items) | Dropout, weight decay, early stopping, contrastive regularization |

---

## 9. Expected Outputs

### 9.1 Artifacts

| Output | Path | Description |
|---|---|---|
| Trained models | `exp/outputs/{model}_{seed}/best_model.pt` | Best checkpoint per run |
| Training logs | `exp/outputs/{model}_{seed}/run_log.json` | Full training history |
| TensorBoard | `exp/outputs/tensorboard/{model}_{seed}/` | Real-time monitoring |
| Comparison table | `exp/outputs/comparison_results.json` | Cross-model metrics |
| Significance tests | `exp/outputs/significance_results.json` | P-values, CIs |
| Final report | `exp/outputs/EXPERIMENT_REPORT.md` | Human-readable results |

### 9.2 Report Structure

```
1. Executive Summary
2. Dataset Description (subset statistics)
3. Model Descriptions (architecture + hyperparameters)
4. Main Results Table (all models × all metrics with CIs)
5. Duration-Stratified Analysis (per-bucket NDCG breakdown)
6. Ablation Study (waterfall from BERT4Rec → Variant A)
7. Variant B: Debiasing Analysis
8. Variant C: Calibration Analysis
9. Statistical Significance Summary
10. Training Efficiency Analysis (time, memory, convergence)
11. Conclusions & Recommendations
12. Appendix: Full hyperparameters, per-seed results
```

---

## 10. Dependencies

```
# exp/requirements.txt
torch>=2.0
numpy
pandas
pyarrow
pyyaml
tensorboard
tqdm
scipy            # significance tests
scikit-learn     # stratified sampling, some metrics
matplotlib       # report plots
seaborn          # report plots
```

---

## 11. Estimated Timeline (Sequential Execution)

| Phase | Tasks | Estimated GPU Time |
|---|---|---|
| Phase 1: Foundation | Infra + data + evaluation | No GPU needed |
| Phase 2: Baselines | Pop + SASRec + BERT4Rec (3 seeds each) | ~6-12 hours |
| Phase 3: Variant A | Model + training + ablation (3+7 runs) | ~12-20 hours |
| Phase 4: Variants B & C | Two models (3 seeds each) | ~8-16 hours |
| Phase 5: Analysis | Significance + plots + report | Minimal GPU |
| **Total** | | **~26-48 hours GPU** |

---

## 12. Quick-Start Commands

```bash
# 1. Install dependencies
pip install -r exp/requirements.txt

# 2. Create data subset
python exp/scripts/create_subset.py \
  --source-dir data_prep/legacy/processed_final \
  --output-dir exp/data \
  --num-users 100000 \
  --min-seq-len 5

# 3. Train a single model
python exp/scripts/train.py --config exp/configs/variant_a.yaml --seed 42

# 4. Train all models
python exp/scripts/train_all.py --seeds 42,123,456

# 5. Monitor training
tensorboard --logdir exp/outputs/tensorboard/

# 6. Run comparison and significance tests
python exp/scripts/compare.py --results-dir exp/outputs/

# 7. Generate final report
python exp/scripts/generate_report.py --results-dir exp/outputs/ --output exp/outputs/EXPERIMENT_REPORT.md
```
