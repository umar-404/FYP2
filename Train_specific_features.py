import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# TRAIN CLINICAL MODEL
# Trains a Random Forest on 32 clean clinical features only.
# No leakage columns. Purpose-built for new patient intake.
# Run this once. Saves 3 files:
#   clinical_model.pkl
#   clinical_scaler.pkl
#   clinical_features.pkl
# ============================================================

FEATURES = [
    'demographic.age_at_index',
    'diagnoses.days_to_last_follow_up',
    'demographic.gender_male',
    'demographic.gender_female',
    'diagnoses.tumor_grade_G1',
    'diagnoses.tumor_grade_G2',
    'diagnoses.tumor_grade_G3',
    'diagnoses.tumor_grade_G4',
    'diagnoses.tumor_grade_Not Reported',
    'diagnoses.metastasis_at_diagnosis_Metastasis, NOS',
    'diagnoses.metastasis_at_diagnosis_No Metastasis',
    'diagnoses.metastasis_at_diagnosis_Unknown',
    'diagnoses.residual_disease_R0',
    'diagnoses.residual_disease_R1',
    'diagnoses.residual_disease_R2',
    'diagnoses.residual_disease_RX',
    'diagnoses.classification_of_tumor_primary',
    'diagnoses.classification_of_tumor_recurrence',
    'diagnoses.classification_of_tumor_metastasis',
    'diagnoses.classification_of_tumor_Synchronous primary',
    'diagnoses.classification_of_tumor_not reported',
    'diagnoses.prior_malignancy_yes',
    'diagnoses.prior_malignancy_no',
    'diagnoses.prior_malignancy_unknown',
    'family_histories.relative_with_cancer_history_yes',
    'family_histories.relative_with_cancer_history_no',
    'family_histories.relative_with_cancer_history_Unknown',
    'exposures.tobacco_smoking_status_Lifelong Non-Smoker',
    'exposures.tobacco_smoking_status_Current Smoker',
    'exposures.tobacco_smoking_status_Current Reformed Smoker for < or = 15 yrs',
    'exposures.tobacco_smoking_status_Current Reformed Smoker for > 15 yrs',
    'exposures.tobacco_smoking_status_Unknown',
]

print("📂 Loading dataset...")
df = pd.read_csv('ML_READY_DATA.csv')

X = df[FEATURES].fillna(0)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Balance classes via oversampling
X_train_df = X_train.copy()
X_train_df['target'] = y_train.values
majority          = X_train_df[X_train_df['target'] == 1.0]
minority          = X_train_df[X_train_df['target'] == 0.0]
minority_upsampled = resample(
    minority, replace=True,
    n_samples=len(majority),
    random_state=42
)
balanced     = pd.concat([majority, minority_upsampled])
X_train_bal  = balanced.drop('target', axis=1)
y_train_bal  = balanced['target']

# Scale
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train_bal)
X_test_scaled   = scaler.transform(X_test)

# Train
print("🚀 Training clinical model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_split=20,
    random_state=42
)
model.fit(X_train_scaled, y_train_bal)

# Evaluate
preds = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]

print("\n" + "=" * 40)
print("  CLINICAL MODEL PERFORMANCE")
print("=" * 40)
print(f"  Accuracy  : {accuracy_score(y_test, preds):.4f}")
print(f"  F1 Score  : {f1_score(y_test, preds):.4f}")
print(f"  ROC AUC   : {roc_auc_score(y_test, probs):.4f}")
print("=" * 40)

# Save
joblib.dump(model,   'clinical_model.pkl')
joblib.dump(scaler,  'clinical_scaler.pkl')
joblib.dump(FEATURES,'clinical_features.pkl')

print("\n✅ Saved:")
print("   clinical_model.pkl")
print("   clinical_scaler.pkl")
print("   clinical_features.pkl")
print("\nRun new_patient.py to start patient intake.\n")