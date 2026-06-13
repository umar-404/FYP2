# Genomic Driver/Passenger Mutation Prediction Pipeline

A two-stage machine learning pipeline for identifying cancer driver mutations from COSMIC genomic data using biochemical feature engineering and ensemble classifiers.

---

## Overview

This pipeline consists of two scripts that work in sequence:

1. **`genomic1_dataprep.py`** — Data preparation and feature engineering
2. **`genomic1.py`** — Model training, evaluation, and gene danger scoring

The goal is to classify mutations as **driver** (oncogenic, recurring) or **passenger** (neutral, random) based on amino acid biochemical properties and positional hotspot frequency.

---

## Files

| File | Role | Input | Output |
|---|---|---|---|
| `genomic1_dataprep.py` | Data prep | `cosmic.csv` | `advanced_genomic_data.csv` |
| `genomic1.py` | ML training & scoring | `advanced_genomic_data.csv` | `ui_top_10_chart.json`, `ui_top_100_table.json` |

---

## Stage 1 — Data Preparation (`genomic1_dataprep.py`)

### What it does

**1. Parses amino acid mutations**
Extracts three components from COSMIC's `AA Mutation` field (e.g., `p.R175H`):
- `Ref` — reference amino acid (e.g., `R`)
- `Pos` — position in the protein sequence (e.g., `175`)
- `Alt` — alternate/mutant amino acid (e.g., `H`)

Rows with ambiguous mutations (`p.?`) are filtered out before parsing.

**2. Biochemical feature engineering**
Each mutation is enriched with two chemical difference scores derived from a built-in amino acid property map:

| Feature | Description |
|---|---|
| `Vol_Diff` | Absolute difference in side-chain volume (Å³) between Ref and Alt |
| `Hydro_Diff` | Absolute difference in Kyte-Doolittle hydropathy index between Ref and Alt |

These features give the model biochemical context — large volume or polarity shifts often indicate functionally disruptive mutations.

**3. Hotspot-based target labeling**
A `Mut_ID` is constructed as `GeneName_Position`. If a specific gene-position combination appears **more than 10 times** across the dataset, it is labeled a **driver** (Target = 1); otherwise it is a **passenger** (Target = 0).

This reflects the cancer genomics principle that recurrently mutated positions are likely under positive selection.

### Output columns added

| Column | Type | Description |
|---|---|---|
| `Ref` | str | Reference amino acid |
| `Pos` | int | Mutation position |
| `Alt` | str | Alternate amino acid |
| `Vol_Diff` | float | Side-chain volume difference |
| `Hydro_Diff` | float | Hydropathy index difference |
| `Mut_ID` | str | Gene + position identifier |
| `Target` | int | 1 = Driver, 0 = Passenger |

---

## Stage 2 — Model Training & Scoring (`genomic1.py`)

### Feature set

The following five features are used for classification:

| Feature | Description |
|---|---|
| `Ref_Enc` | Label-encoded reference amino acid |
| `Pos` | Mutation position in protein |
| `Alt_Enc` | Label-encoded alternate amino acid |
| `Vol_Diff` | Side-chain volume difference |
| `Hydro_Diff` | Hydropathy index difference |

### Train/test split strategy

A **gene-aware group split** (`GroupShuffleSplit`) is used so that **no gene appears in both training and test sets**. This prevents data leakage — a model should not be evaluated on a gene it was trained on, since hotspot patterns are gene-specific.

- Test size: 20%
- Split boundary: `Gene Name`

### Class balancing

The raw dataset is heavily imbalanced (far more passengers than drivers). To address this, only the **training set** is rebalanced:

- All driver mutations are kept
- Passenger mutations are downsampled to **1.5× the number of drivers**
- The test set is left untouched (reflects real-world distribution)

This avoids inflated accuracy from the model simply predicting the majority class.

### Models evaluated

Five classifiers are trained and benchmarked:

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Bagging ensemble, 50 trees |
| XGBoost | Gradient boosting (histogram method) |
| LightGBM | Gradient boosting, optimized for speed |
| CatBoost | Gradient boosting with categorical handling |

Each model is evaluated on **Accuracy** and **F1 Score** (weighted toward the minority driver class), along with training time.

### Gene danger scoring

After evaluation, **XGBoost** is used to score all mutations in the full dataset. Each mutation receives a `Danger_Score` (probability of being a driver, from `predict_proba`). Scores are then **averaged per gene** to produce a ranked gene-level danger index.

### Outputs

| File | Contents |
|---|---|
| `ui_top_10_chart.json` | Top 10 genes by mean danger score (for chart rendering) |
| `ui_top_100_table.json` | Top 100 genes by mean danger score (for table rendering) |

Both JSON files follow the format: `[{"Gene Name": "...", "Danger_Score": ...}, ...]`

---

## How to Run

### Prerequisites

```bash
pip install pandas scikit-learn xgboost lightgbm catboost tqdm
```

### Step 1 — Prepare data

Place your `cosmic.csv` in the working directory, then run:

```bash
python genomic1_dataprep.py
```

This generates `advanced_genomic_data.csv`.

### Step 2 — Train and score

```bash
python genomic1.py
```

This outputs the evaluation table to the terminal and saves the two JSON ranking files.

---

## Input Data Requirements

`cosmic.csv` must contain at minimum:

| Column | Description |
|---|---|
| `AA Mutation` | HGVS protein-level mutation string (e.g., `p.R175H`) |
| `Gene Name` | HGNC gene symbol |

---

## Design Notes

- The hotspot threshold of **>10 occurrences** is a deliberate choice to define recurrence. Lowering it increases driver labels (more recall, less precision); raising it tightens the definition.
- **Group-based splitting** is essential here. Without it, mutations from the same gene would appear in both train and test, and the model would trivially learn gene identity rather than biochemical signal.
- The **1.5× passenger-to-driver ratio** in training is a soft balance — aggressive enough to prevent the model from ignoring drivers, but not so extreme as to introduce excessive noise.
- XGBoost is selected for final scoring due to its combination of accuracy and calibrated probability outputs via `predict_proba`.
