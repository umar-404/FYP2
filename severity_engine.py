# Logic for determining severity levels based on predicted probabilities and clinical features. 
# This engine calculates a base severity level from the probability of cancer, 
# then applies clinical adjustments based on key features to arrive at a final severity level. 
# It also provides treatment recommendations based on the final severity.

import pandas as pd


class SeverityEngine:

    def __init__(self):

        # =====================================================
        # PROBABILITY THRESHOLDS → BASE SEVERITY LEVEL
        # No arbitrary multipliers. Probability directly
        # determines the starting level.
        # =====================================================

        self.PROB_THRESHOLDS = {
            "Mild":     (0.00, 0.30),
            "Moderate": (0.30, 0.55),
            "High":     (0.55, 0.75),
            "Critical": (0.75, 1.00),
        }

        # Severity levels ordered lowest to highest
        self.LEVELS = ["Mild", "Moderate", "High", "Critical"]


    # =====================================================
    # STEP 1 — BASE LEVEL FROM PROBABILITY
    # =====================================================

    def get_base_level(self, probability):

        for level, (low, high) in self.PROB_THRESHOLDS.items():
            if low <= probability < high:
                return level

        return "Critical"  # probability == 1.0 edge case


    # =====================================================
    # STEP 2 — CLINICAL ADJUSTMENT (+1 or -1 level)
    # Each feature either upgrades or downgrades the
    # base level by one step. Maximum one step total.
    # =====================================================

    def get_clinical_adjustment(self, row):

        upgrade_reasons   = []
        downgrade_reasons = []


        # -------------------------------------------------
        # UPGRADE SIGNALS (bad clinical signs)
        # -------------------------------------------------

        # Metastasis at diagnosis — strongest upgrade signal
        if row.get('diagnoses.metastasis_at_diagnosis_Metastasis, NOS', 0) == 1:
            upgrade_reasons.append("Metastasis at diagnosis")

        # High tumor grade
        if row.get('diagnoses.tumor_grade_G4', 0) == 1:
            upgrade_reasons.append("Grade G4 (Undifferentiated)")
        elif row.get('diagnoses.tumor_grade_G3', 0) == 1:
            upgrade_reasons.append("Grade G3 (Poorly differentiated)")

        # Residual disease after surgery
        if row.get('diagnoses.residual_disease_R2', 0) == 1:
            upgrade_reasons.append("Residual disease R2 (Macroscopic)")
        elif row.get('diagnoses.residual_disease_R1', 0) == 1:
            upgrade_reasons.append("Residual disease R1 (Microscopic)")

        # Tumor classification
        if row.get('diagnoses.classification_of_tumor_metastasis', 0) == 1:
            upgrade_reasons.append("Tumor classified as metastasis")
        elif row.get('diagnoses.classification_of_tumor_recurrence', 0) == 1:
            upgrade_reasons.append("Tumor classified as recurrence")

        # Prior malignancy
        if row.get('diagnoses.prior_malignancy_yes', 0) == 1:
            upgrade_reasons.append("Prior malignancy history")

        # Short follow-up (< 1 year)
        followup = row.get('diagnoses.days_to_last_follow_up', None)
        if followup is not None and pd.notnull(followup) and followup < 365:
            upgrade_reasons.append("Short follow-up (< 1 year)")

        # Old age
        age = row.get('demographic.age_at_index', None)
        if age is not None and pd.notnull(age) and age > 70:
            upgrade_reasons.append(f"Age > 70 ({int(age)})")


        # -------------------------------------------------
        # DOWNGRADE SIGNALS (good clinical signs)
        # -------------------------------------------------

        # Clean surgical margins
        if row.get('diagnoses.residual_disease_R0', 0) == 1:
            downgrade_reasons.append("Clean margins R0")

        # No metastasis confirmed
        if row.get('diagnoses.metastasis_at_diagnosis_No Metastasis', 0) == 1:
            downgrade_reasons.append("No metastasis at diagnosis")

        # Low tumor grade
        if row.get('diagnoses.tumor_grade_G1', 0) == 1:
            downgrade_reasons.append("Grade G1 (Well differentiated)")

        # No family history
        if row.get('family_histories.relative_with_cancer_history_no', 0) == 1:
            downgrade_reasons.append("No family history of cancer")

        # Non-smoker
        if row.get('exposures.tobacco_smoking_status_Lifelong Non-Smoker', 0) == 1:
            downgrade_reasons.append("Lifelong non-smoker")

        # Long follow-up (> 2 years — good prognosis sign)
        if followup is not None and pd.notnull(followup) and followup >= 730:
            downgrade_reasons.append("Long follow-up (> 2 years)")


        # -------------------------------------------------
        # DECISION — upgrade takes priority over downgrade
        # Only move ONE step in either direction
        # -------------------------------------------------

        if upgrade_reasons:
            return +1, upgrade_reasons, []

        elif downgrade_reasons:
            return -1, [], downgrade_reasons

        else:
            return 0, [], []


    # =====================================================
    # STEP 3 — APPLY ADJUSTMENT TO BASE LEVEL
    # =====================================================

    def apply_adjustment(self, base_level, adjustment):

        current_index = self.LEVELS.index(base_level)
        new_index     = current_index + adjustment

        # Clamp within valid range
        new_index = max(0, min(new_index, len(self.LEVELS) - 1))

        return self.LEVELS[new_index]


    # =====================================================
    # CALCULATE SEVERITY — MAIN METHOD
    # =====================================================

    def calculate_severity(self, row, probability):

        # Step 1: Base level from probability
        base_level = self.get_base_level(probability)

        # Step 2: Clinical adjustment
        adjustment, upgrade_reasons, downgrade_reasons = \
            self.get_clinical_adjustment(row)

        # Step 3: Final level
        final_level = self.apply_adjustment(base_level, adjustment)

        return base_level, final_level, upgrade_reasons, downgrade_reasons


    # =====================================================
    # TREATMENT RECOMMENDATION
    # =====================================================

    def recommend_treatment(self, severity_level):

        recommendations = {
            "Mild": (
                "Routine monitoring, "
                "screening, lifestyle management"
            ),
            "Moderate": (
                "Chemotherapy consultation, "
                "clinical observation, periodic imaging"
            ),
            "High": (
                "Aggressive chemotherapy, "
                "radiation therapy, possible surgery"
            ),
            "Critical": (
                "Immediate oncology intervention, "
                "targeted therapy, intensive treatment planning"
            ),
        }

        return recommendations.get(severity_level, "No recommendation available")


    # =====================================================
    # SINGLE PATIENT SEVERITY
    # =====================================================

    def predict_patient_severity(
        self,
        X_data,
        prediction_probabilities,
        patient_index
    ):

        row  = X_data.iloc[patient_index]
        prob = float(prediction_probabilities[patient_index])

        base_level, final_level, upgrades, downgrades = \
            self.calculate_severity(row, prob)

        treatment = self.recommend_treatment(final_level)

        # Build adjustment explanation
        if upgrades:
            adjustment_note = f"Upgraded from {base_level} due to: {', '.join(upgrades)}"
        elif downgrades:
            adjustment_note = f"Downgraded from {base_level} due to: {', '.join(downgrades)}"
        else:
            adjustment_note = f"No adjustment — level stays at {base_level}"

        return {
            "Patient_Index":            int(patient_index),
            "Cancer_Probability":       f"{prob * 100:.2f}%",
            "Base_Level":               base_level,
            "Severity_Level":           final_level,
            "Adjustment":               adjustment_note,
            "Treatment_Recommendation": treatment,
        }


    # =====================================================
    # COMPLETE DATASET PROCESSING
    # =====================================================

    def process_dataset(self, X_data, prediction_probabilities):

        all_results = []

        for i in range(len(X_data)):
            patient_result = self.predict_patient_severity(
                X_data,
                prediction_probabilities,
                i
            )
            all_results.append(patient_result)

        df = pd.DataFrame(all_results)

        df.to_csv('severity_results_all_patients.csv', index=False)
        print(f"✅ Severity calculated for {len(df)} patients.")
        print(f"📁 Saved to: severity_results_all_patients.csv")
        print("\n📊 Severity Distribution:")
        print(df['Severity_Level'].value_counts())

        return df