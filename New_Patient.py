import pandas as pd
import numpy as np
import joblib
import os
from severity_engine import SeverityEngine

# ============================================================
# NEW PATIENT INTAKE SYSTEM
#
# Flow:
#   1. Ask questions in plain English
#   2. Convert answers to ML format (0s and 1s)
#   3. Calculate cancer probability using clinical model
#   4. Calculate severity using SeverityEngine
#   5. Display result
#   6. Save patient to new_patients.csv
# ============================================================

SAVE_FILE     = 'new_patients.csv'
MODEL_PATH    = 'clinical_model.pkl'
SCALER_PATH   = 'clinical_scaler.pkl'
FEATURES_PATH = 'clinical_features.pkl'


# ============================================================
# LOAD MODEL, SCALER, FEATURE LIST
# ============================================================

def load_model():
    try:
        model    = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        print("✅ Clinical model loaded successfully.")
        return model, scaler, features
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("   Please run train_clinical_model.py first.")
        exit()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ask(question, valid_options):
    """Present numbered options and return chosen answer."""
    while True:
        print(f"\n  {question}")
        for i, opt in enumerate(valid_options, 1):
            print(f"    {i}. {opt}")
        answer = input("  Enter number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(valid_options):
            return valid_options[int(answer) - 1]
        print("  ❌ Invalid. Please enter a valid number.")


def ask_number(question, min_val, max_val):
    """Ask for a numeric value within a range."""
    while True:
        print(f"\n  {question} ({min_val}–{max_val}):")
        answer = input("  Your answer: ").strip()
        try:
            val = float(answer)
            if min_val <= val <= max_val:
                return val
            print(f"  ❌ Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  ❌ Please enter a valid number.")


# ============================================================
# STEP 1 — COLLECT PLAIN ENGLISH INPUT
# ============================================================

def collect_patient_info():

    print("\n" + "=" * 60)
    print("       NEW PATIENT INTAKE — COLON CANCER ASSESSMENT")
    print("=" * 60)
    print("  Answer each question by entering the option number.")
    print("=" * 60)

    info = {}

    info['age'] = ask_number(
        "Patient age",
        min_val=0, max_val=120
    )

    info['gender'] = ask(
        "Patient gender:",
        ["Male", "Female"]
    )

    info['tumor_grade'] = ask(
        "Tumor grade (how abnormal the cancer cells look):",
        [
            "G1 — Well differentiated (cells look most normal)",
            "G2 — Moderately differentiated",
            "G3 — Poorly differentiated (cells look abnormal)",
            "G4 — Undifferentiated (cells look most abnormal)",
            "Not Reported / Unknown",
        ]
    )

    info['metastasis'] = ask(
        "Was metastasis (cancer spread) present at diagnosis?",
        [
            "Yes — Metastasis present",
            "No — No metastasis",
            "Unknown",
        ]
    )

    info['residual_disease'] = ask(
        "Residual disease after surgery (tumor left behind):",
        [
            "R0 — No residual tumor (clean margins)",
            "R1 — Microscopic residual tumor",
            "R2 — Macroscopic residual tumor (visible)",
            "Unknown / Not done",
        ]
    )

    info['tumor_classification'] = ask(
        "How is the tumor classified?",
        [
            "Primary — First occurrence",
            "Recurrence — Cancer came back",
            "Metastasis — Spread from another site",
            "Synchronous primary — Two primaries at same time",
            "Not Reported",
        ]
    )

    info['prior_malignancy'] = ask(
        "Does the patient have a history of prior cancer?",
        ["Yes", "No", "Unknown"]
    )

    info['family_history'] = ask(
        "Does any close family member have a history of cancer?",
        ["Yes", "No", "Unknown"]
    )

    info['smoking'] = ask(
        "Patient smoking status:",
        [
            "Lifelong Non-Smoker",
            "Current Smoker",
            "Reformed Smoker (quit ≤ 15 years ago)",
            "Reformed Smoker (quit > 15 years ago)",
            "Unknown / Not Documented",
        ]
    )

    info['follow_up_days'] = ask_number(
        "Days to last follow-up (365 = 1 year, 730 = 2 years)",
        min_val=0, max_val=10000
    )

    return info


# ============================================================
# STEP 2 — CONVERT ENGLISH ANSWERS → ML FORMAT (0s and 1s)
# ============================================================

def convert_to_ml_format(info):
    """
    Maps plain English answers to exact binary column names
    used in ML_READY_DATA.csv and the clinical model.
    All columns start at 0. Only the chosen option becomes 1.
    """

    row = {
        # Continuous
        'demographic.age_at_index':                                                     0.0,
        'diagnoses.days_to_last_follow_up':                                             0.0,

        # Gender
        'demographic.gender_male':                                                      0,
        'demographic.gender_female':                                                    0,

        # Tumor grade
        'diagnoses.tumor_grade_G1':                                                     0,
        'diagnoses.tumor_grade_G2':                                                     0,
        'diagnoses.tumor_grade_G3':                                                     0,
        'diagnoses.tumor_grade_G4':                                                     0,
        'diagnoses.tumor_grade_Not Reported':                                           0,

        # Metastasis
        'diagnoses.metastasis_at_diagnosis_Metastasis, NOS':                            0,
        'diagnoses.metastasis_at_diagnosis_No Metastasis':                              0,
        'diagnoses.metastasis_at_diagnosis_Unknown':                                    0,

        # Residual disease
        'diagnoses.residual_disease_R0':                                                0,
        'diagnoses.residual_disease_R1':                                                0,
        'diagnoses.residual_disease_R2':                                                0,
        'diagnoses.residual_disease_RX':                                                0,

        # Tumor classification
        'diagnoses.classification_of_tumor_primary':                                    0,
        'diagnoses.classification_of_tumor_recurrence':                                 0,
        'diagnoses.classification_of_tumor_metastasis':                                 0,
        'diagnoses.classification_of_tumor_Synchronous primary':                        0,
        'diagnoses.classification_of_tumor_not reported':                               0,

        # Prior malignancy
        'diagnoses.prior_malignancy_yes':                                               0,
        'diagnoses.prior_malignancy_no':                                                0,
        'diagnoses.prior_malignancy_unknown':                                           0,

        # Family history
        'family_histories.relative_with_cancer_history_yes':                            0,
        'family_histories.relative_with_cancer_history_no':                             0,
        'family_histories.relative_with_cancer_history_Unknown':                        0,

        # Smoking
        'exposures.tobacco_smoking_status_Lifelong Non-Smoker':                         0,
        'exposures.tobacco_smoking_status_Current Smoker':                              0,
        'exposures.tobacco_smoking_status_Current Reformed Smoker for < or = 15 yrs':   0,
        'exposures.tobacco_smoking_status_Current Reformed Smoker for > 15 yrs':        0,
        'exposures.tobacco_smoking_status_Unknown':                                     0,
    }

    # Fill continuous
    row['demographic.age_at_index']         = info['age']
    row['diagnoses.days_to_last_follow_up'] = info['follow_up_days']

    # Gender
    if info['gender'] == 'Male':
        row['demographic.gender_male'] = 1
    else:
        row['demographic.gender_female'] = 1

    # Tumor grade
    grade_map = {
        "G1 — Well differentiated (cells look most normal)":    'diagnoses.tumor_grade_G1',
        "G2 — Moderately differentiated":                       'diagnoses.tumor_grade_G2',
        "G3 — Poorly differentiated (cells look abnormal)":     'diagnoses.tumor_grade_G3',
        "G4 — Undifferentiated (cells look most abnormal)":     'diagnoses.tumor_grade_G4',
        "Not Reported / Unknown":                               'diagnoses.tumor_grade_Not Reported',
    }
    row[grade_map[info['tumor_grade']]] = 1

    # Metastasis
    metastasis_map = {
        "Yes — Metastasis present": 'diagnoses.metastasis_at_diagnosis_Metastasis, NOS',
        "No — No metastasis":       'diagnoses.metastasis_at_diagnosis_No Metastasis',
        "Unknown":                  'diagnoses.metastasis_at_diagnosis_Unknown',
    }
    row[metastasis_map[info['metastasis']]] = 1

    # Residual disease
    residual_map = {
        "R0 — No residual tumor (clean margins)":    'diagnoses.residual_disease_R0',
        "R1 — Microscopic residual tumor":           'diagnoses.residual_disease_R1',
        "R2 — Macroscopic residual tumor (visible)": 'diagnoses.residual_disease_R2',
        "Unknown / Not done":                        'diagnoses.residual_disease_RX',
    }
    row[residual_map[info['residual_disease']]] = 1

    # Tumor classification
    classification_map = {
        "Primary — First occurrence":                       'diagnoses.classification_of_tumor_primary',
        "Recurrence — Cancer came back":                    'diagnoses.classification_of_tumor_recurrence',
        "Metastasis — Spread from another site":            'diagnoses.classification_of_tumor_metastasis',
        "Synchronous primary — Two primaries at same time": 'diagnoses.classification_of_tumor_Synchronous primary',
        "Not Reported":                                     'diagnoses.classification_of_tumor_not reported',
    }
    row[classification_map[info['tumor_classification']]] = 1

    # Prior malignancy
    prior_map = {
        "Yes":     'diagnoses.prior_malignancy_yes',
        "No":      'diagnoses.prior_malignancy_no',
        "Unknown": 'diagnoses.prior_malignancy_unknown',
    }
    row[prior_map[info['prior_malignancy']]] = 1

    # Family history
    family_map = {
        "Yes":     'family_histories.relative_with_cancer_history_yes',
        "No":      'family_histories.relative_with_cancer_history_no',
        "Unknown": 'family_histories.relative_with_cancer_history_Unknown',
    }
    row[family_map[info['family_history']]] = 1

    # Smoking
    smoking_map = {
        "Lifelong Non-Smoker":                   'exposures.tobacco_smoking_status_Lifelong Non-Smoker',
        "Current Smoker":                        'exposures.tobacco_smoking_status_Current Smoker',
        "Reformed Smoker (quit ≤ 15 years ago)": 'exposures.tobacco_smoking_status_Current Reformed Smoker for < or = 15 yrs',
        "Reformed Smoker (quit > 15 years ago)": 'exposures.tobacco_smoking_status_Current Reformed Smoker for > 15 yrs',
        "Unknown / Not Documented":              'exposures.tobacco_smoking_status_Unknown',
    }
    row[smoking_map[info['smoking']]] = 1

    return row


# ============================================================
# STEP 3 — CALCULATE PROBABILITY USING CLINICAL MODEL
# ============================================================

def calculate_probability(ml_row, model, scaler, features):
    """
    Converts the ML row into a DataFrame matching the
    exact feature order the model was trained on,
    scales it, then returns the cancer probability.
    """
    df_input = pd.DataFrame([ml_row])[features]
    df_scaled = scaler.transform(df_input)
    probability = model.predict_proba(df_scaled)[0][1]
    return float(probability)


# ============================================================
# STEP 4 — CALCULATE SEVERITY
# ============================================================

def calculate_severity(ml_row, probability):

    engine     = SeverityEngine()
    row_series = pd.Series(ml_row)

    base_level, final_level, upgrades, downgrades = \
        engine.calculate_severity(row_series, probability)

    treatment = engine.recommend_treatment(final_level)

    if upgrades:
        adjustment = f"Upgraded from {base_level} → {final_level} due to: {', '.join(upgrades)}"
    elif downgrades:
        adjustment = f"Downgraded from {base_level} → {final_level} due to: {', '.join(downgrades)}"
    else:
        adjustment = f"No adjustment — stays at {base_level}"

    return {
        "Cancer_Probability":       f"{probability * 100:.2f}%",
        "Base_Level":               base_level,
        "Severity_Level":           final_level,
        "Adjustment":               adjustment,
        "Treatment_Recommendation": treatment,
    }


# ============================================================
# STEP 5 — SAVE PATIENT TO CSV
# ============================================================

def save_patient(info, severity_result):

    record = {
        "age":                      info['age'],
        "gender":                   info['gender'],
        "tumor_grade":              info['tumor_grade'].split(' — ')[0],
        "metastasis":               info['metastasis'],
        "residual_disease":         info['residual_disease'].split(' — ')[0],
        "tumor_classification":     info['tumor_classification'].split(' — ')[0],
        "prior_malignancy":         info['prior_malignancy'],
        "family_history":           info['family_history'],
        "smoking_status":           info['smoking'],
        "follow_up_days":           info['follow_up_days'],
        "cancer_probability":       severity_result['Cancer_Probability'],
        "base_severity_level":      severity_result['Base_Level'],
        "final_severity_level":     severity_result['Severity_Level'],
        "adjustment_reason":        severity_result['Adjustment'],
        "treatment_recommendation": severity_result['Treatment_Recommendation'],
    }

    df_new = pd.DataFrame([record])

    if os.path.exists(SAVE_FILE):
        df_existing = pd.read_csv(SAVE_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(SAVE_FILE, index=False)

    return len(df_combined)


# ============================================================
# MAIN
# ============================================================

def main():

    # Load model once at startup
    model, scaler, features = load_model()

    while True:

        # Step 1 — Collect English input
        info = collect_patient_info()

        # Step 2 — Convert to ML format
        ml_row = convert_to_ml_format(info)

        # Step 3 — Calculate probability automatically
        probability = calculate_probability(ml_row, model, scaler, features)

        # Step 4 — Calculate severity
        severity_result = calculate_severity(ml_row, probability)

        # Step 5 — Display result
        print("\n" + "=" * 60)
        print("              PATIENT ASSESSMENT RESULT")
        print("=" * 60)
        print(f"  Cancer Probability   : {severity_result['Cancer_Probability']}")
        print(f"  Base Severity Level  : {severity_result['Base_Level']}")
        print(f"  Final Severity Level : {severity_result['Severity_Level']}")
        print(f"  Adjustment           : {severity_result['Adjustment']}")
        print(f"  Treatment            : {severity_result['Treatment_Recommendation']}")
        print("=" * 60)

        # Step 6 — Save
        total = save_patient(info, severity_result)
        print(f"\n  ✅ Patient saved to {SAVE_FILE}")
        print(f"  📋 Total patients on record: {total}")

        # Continue?
        again = input("\n  Add another patient? (yes / no): ").strip().lower()
        if again != 'yes':
            print("\n  Session ended. All patients saved.\n")
            break


if __name__ == "__main__":
    main()