# ETL Pipeline pour prédiction de ventes

Cette pipeline ETL (Extraire, Transformer, Loader) transforme le dataset initial en dataset prêt pour être entraîné.

## Pipeline Overview

```
Raw Data (artifacts/data/raw/)
    ↓
[1. LOAD] → Load CSV data
    ↓
[2. TRANSFORM] → Aggregate by City + Date, Cap outliers
    ↓
[3. SPLIT] → Temporal train/test split (80/20)
    ↓
[4. FEATURES] → Create temporal, lag, rolling features
    ↓
Train/Test Features (artifacts/data/processed/)
```
