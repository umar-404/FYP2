import pandas as pd
import numpy as np

# 1. Load the merged dataset
df = pd.read_csv('FINAL_COLON_CANCER_DATASET.csv')

# 2. FILTER & TARGET CREATION
# We remove patients with 'Unknown' stages because the model can't learn from them.
valid_stages = ['Stage I', 'Stage II', 'Stage III', 'Stage IV', 
                'Stage IIIA', 'Stage IIIB', 'Stage IIIC', 'Stage IVA', 'Stage IVB', 'Stage IVC']
df = df[df['diagnoses.ajcc_pathologic_stage'].isin(valid_stages)].copy()

# Create binary target: 1 = Advanced (III/IV), 0 = Early (I/II)
df['target'] = df['diagnoses.ajcc_pathologic_stage'].apply(
    lambda x: 1 if 'III' in str(x) or 'IV' in str(x) else 0
)

# 3. FEATURE SELECTION (Dropping IDs and "Leaky" columns)
# We drop specific columns that tell the model the answer directly (like Stage or Tumor-M/N values)
cols_to_drop = [
    'cases.case_id', 
    'diagnoses.ajcc_pathologic_stage', 
    'diagnoses.ajcc_clinical_stage',
    'diagnoses.ajcc_pathologic_t', 
    'diagnoses.ajcc_pathologic_n', 
    'diagnoses.ajcc_pathologic_m',
    'diagnoses.ajcc_staging_system_edition'
]
df = df.drop(columns=cols_to_drop)

# 4. HANDLE MULTI-LABEL TREATMENTS
# This turns "Chemo; Surgery" into separate columns for each treatment type.
treatments = df['treatments.treatment_type'].str.get_dummies(sep='; ')
# Prefix them so we know they are treatments
treatments = treatments.add_prefix('Treatment_')
df = df.drop(columns=['treatments.treatment_type'])

# 5. ENCODE ALL OTHER CATEGORICAL DATA
# This converts Gender, Race, Smoking Status, etc., into numeric One-Hot columns.
df_numeric = pd.get_dummies(df)

# 6. COMBINE AND SAVE
# Merge the treatment columns back with the rest of the numeric data
final_ml_ready = pd.concat([df_numeric, treatments], axis=1)

# Ensure the 'target' column is at the very end
cols = [c for c in final_ml_ready.columns if c != 'target'] + ['target']
final_ml_ready = final_ml_ready[cols]

# 7. Final Clean-up: Convert any leftover booleans (True/False) to 1/0
final_ml_ready = final_ml_ready.astype(float)

# Save the file
final_ml_ready.to_csv('ML_READY_DATA.csv', index=False)

print("--- ML READY DATA GENERATED ---")
print(f"Total Rows (Patients): {len(final_ml_ready)}")
print(f"Total Features (Columns): {len(final_ml_ready.columns)}")
print("The file 'ML_READY_DATA.csv' is now 100% numeric and ready for any algorithm.")