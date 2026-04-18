 # Interaction + Content Enrichment Dataset for BERT4Rec Experiments

  ## Summary

  Build a reproducible preprocessing pipeline that converts the current raw inputs into a clean, model-ready
  dataset centered on watch_time and content duration, while enriching each event with useful content metadata
  from content_profile.csv and explicitly excluding catalog performance metrics from the final modeling features.

  The pipeline will read:

  - data/raw_data/*.parquet: per-user sequential interactions stored as ordered contents arrays
  - data/content_profile.csv: full content catalog, but only metadata fields will be retained for enrichment

  The pipeline will produce:

  - A cleaned flat interaction table with one row per user-content event
  - A model-ready sequence dataset for BERT4Rec-style experiments
  - A processed content metadata table aligned to the interaction universe
  - A Markdown report describing raw data status, cleaning decisions, feature construction, and final dataset
    quality

  ## Key Changes

  ### 1. Raw-to-flat interaction normalization

  Create a preprocessing script that:

  - Loads all parquet shards and explodes each user’s contents array into atomic events
  - Preserves the existing within-user order and also sorts by min_wt_timestamp as a safety step
  - Normalizes types for user_id, content_id, min_wt_timestamp, watch_time, and runtime
  - Renames fields into a stable schema for downstream use:
      - user_id
      - content_id
      - event_ts
      - watch_time_sec
      - runtime_sec_raw

  ### 2. Content metadata cleaning and selection

  Create a catalog-prep step that:

  - Loads data/content_profile.csv
  - Keeps only metadata useful for representation learning and enrichment:
      - content_id
      - content_title
      - runtime_mins
      - content_type
      - release_year
      - country
      - provider
      - content_genre
      - is_simulcast
      - geo_check
      - short_description
      - long_description
  - Drops performance/aggregate behavior columns from the modeling output:
      - all views_*
      - weekend_views
      - unique_viewers_30d
      - returning_viewers_30d
      - completed_viewers
      - completion_rate
      - avg_watchtime
      - avg_watch_ratio
      - end_date_simulcast
  - Standardizes missing text/categorical values and trims obvious text noise
  - Adds runtime_sec_meta = runtime_mins * 60

  ### 3. Enrichment and cleaning rules

  Join interactions to cleaned content metadata by content_id and apply moderate filtering:

  - Drop events with missing content_id, missing user_id, missing timestamp, or failed content join
  - Drop rows with non-positive runtime
  - Keep zero/very-short watch events as requested by the moderate policy, but flag them
  - Keep rows where watch_time > runtime, but cap the derived normalized ratio and retain an anomaly flag rather
    than silently deleting unless the overrun is extreme
  - Deduplicate exact duplicate events on (user_id, content_id, event_ts, watch_time_sec)
  - Keep only users with enough history for sequential modeling:
      - default threshold: at least 3 valid events after cleaning
  - Keep only items with minimum interaction support:
      - default threshold: at least 5 valid events after cleaning

  Derived features to add per event:

  - runtime_sec_final: prefer runtime_sec_raw; fall back to runtime_sec_meta if raw is missing/invalid
  - watch_ratio = watch_time_sec / runtime_sec_final
  - watch_ratio_clipped clipped to [0, 1]
  - watch_bucket using behaviorally useful bins such as 0, (0,0.1], (0.1,0.3], (0.3,0.7], (0.7,1.0], >1
  - is_zero_watch
  - is_short_watch
  - is_over_runtime
  - content_duration_bucket from runtime_sec_final
  - recency_day or equivalent date field derived from timestamp for analysis

  ### 4. Model-ready sequence artifacts

  Produce BERT4Rec-oriented outputs:

  - processed/interactions_enriched.parquet: flat cleaned event table
  - processed/content_metadata_clean.parquet: cleaned content metadata table
  - processed/item_vocab.parquet or CSV: integer item mapping from content_id
  - processed/user_sequences.parquet: per-user ordered item/history representation
  - processed/train.parquet, processed/valid.parquet, processed/test.parquet: chronological splits at the user
    level using leave-last-two or equivalent BERT4Rec-ready split
  - processed/feature_config.json or YAML: explicit description of sequence-side auxiliary signals to use in
    experiments

  Sequence-side fields should include:

  - item token sequence
  - aligned watch_ratio_clipped sequence
  - aligned watch_bucket sequence
  - aligned content_duration_bucket sequence
  - optional aligned categorical metadata tokens for content_type, provider, country

  ### 5. Data analysis and documentation

  Generate a Markdown document, e.g. processed/DATA_PROCESSING_REPORT.md, that includes:

  - Raw data overview
  - Observed schema and nested interaction structure
  - Counts before processing:
      - users
      - exploded events
      - unique contents in raw interactions
      - content join coverage
      - missingness by key field
      - watch-time and runtime anomaly rates
  - Processing decisions and rationale
  - Counts after processing:
      - retained users
      - retained events
      - retained items
      - distribution of sequence lengths
      - distribution of watch ratio
      - coverage of metadata fields
  - Final schema for each artifact
  - Notes on how the processed outputs can be fed into BERT4Rec or feature-enriched sequential baselines

  ## Public Interfaces / Output Schemas

  ### interactions_enriched.parquet

  Expected columns:

  - user_id
  - content_id
  - event_ts
  - watch_time_sec
  - runtime_sec_raw
  - runtime_sec_meta
  - runtime_sec_final
  - watch_ratio
  - watch_ratio_clipped
  - watch_bucket
  - is_zero_watch
  - is_short_watch
  - is_over_runtime
  - content_type
  - release_year
  - country
  - provider
  - content_genre
  - is_simulcast
  - geo_check
  - content_duration_bucket

  ### content_metadata_clean.parquet

  Expected columns:

  - content_id
  - runtime_sec_meta
  - release_year
  - country
  - provider
  - content_genre
  - is_simulcast
  - geo_check
  - short_description
  - long_description

  ### user_sequences.parquet

  Expected per-user fields:

  - user_id
  - item_sequence
  - event_ts_sequence
  - watch_ratio_sequence
  - watch_bucket_sequence
  - duration_bucket_sequence
  - seq_len

  ## Test Plan

  Run non-destructive validation checks after preprocessing:

  - Assert all output schemas and required columns exist
  - Assert no nulls in key fields of cleaned interactions: user_id, content_id, event_ts, runtime_sec_final
  - Assert per-user sequences are strictly non-decreasing by timestamp
  - Assert watch_ratio_clipped is always within [0, 1]
  - Assert every sequence-side aligned feature has the same length as item_sequence
  - Assert train/valid/test splits are leakage-safe and chronologically ordered
  - Report dropped-row counts by reason:
      - missing join
      - missing timestamp
      - invalid runtime
      - duplicate event
      - user/item support filtering

  ## Assumptions and Defaults

  - Primary deliverable is model-ready output, not just EDA
  - Filtering policy is moderate
  - Raw interaction order is treated as valid but still re-sorted by timestamp for safety
  - Catalog performance metrics will be excluded from final modeling features even if retained briefly for raw-
    data diagnostics
  - runtime in raw interactions is treated as the main event-time duration source when valid; catalog duration is
    a fallback/reference source
  - Default support thresholds are min_user_events = 3 and min_item_events = 5
  - Output directory will be processed/ under the repo root
  - Implementation will be script-based and reproducible, not notebook-only