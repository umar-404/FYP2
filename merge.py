import pandas as pd
import numpy as np

# 1. Load the files
# Assuming clinical was saved as CSV in the previous step
df_clinical = pd.read_csv('organized_clinical_data_v2.csv')
df_exposure = pd.read_excel('exposure.xlsx')
df_family = pd.read_excel('family_history.xlsx')

ID_COL = 'cases.case_id'

# 2. Pre-processing Exposure and Family History
# We clean the '-- symbol and other placeholders found in your snippets
placeholders = ["'--", "--", "0", 0, "Unknown", "Not Reported", "not reported"]

def clean_and_flatten(df, name):
    """
    Ensures exposure and family files are also one-row-per-patient
    before we merge them to the main file.
    """
    # Replace placeholders with NaN
    df = df.replace(placeholders, np.nan)
    
    # Group by ID and take the 'first' valid value found for each column
    # (If there are multiple family members, this takes the primary record)
    df_clean = df.groupby(ID_COL).first().reset_index()
    return df_clean

print("Cleaning Exposure and Family History files...")
df_exposure_clean = clean_and_flatten(df_exposure, "Exposure")
df_family_clean = clean_and_flatten(df_family, "Family")

# 3. THE MASTER MERGE
# We start with clinical because every patient MUST have clinical data.
print("Merging datasets...")
final_df = pd.merge(df_clinical, df_exposure_clean, on=ID_COL, how='left')
final_df = pd.merge(final_df, df_family_clean, on=ID_COL, how='left')

# 4. Final Data Polish
# After merging, if a patient didn't have a family history record, 
# Pandas will put a NaN. We fill those with 'Unknown'.
final_df = final_df.fillna("Unknown")

# 5. Summary and Save
print("\n--- FINAL DATASET SUMMARY ---")
print(f"Total Patients: {len(final_df)}")
print(f"Total Features: {len(final_df.columns)}")

# Check for the specific columns you shared
print("\nSample of combined data:")
print(final_df[[ID_COL, 'exposures.tobacco_smoking_status', 'family_histories.relative_with_cancer_history']].head())

final_df.to_csv('FINAL_COLON_CANCER_DATASET.csv', index=False)
print("\nSUCCESS: 'FINAL_COLON_CANCER_DATASET.csv' is ready for Machine Learning!")