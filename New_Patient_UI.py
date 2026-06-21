"""
New_Patient_UI.py — Professional GUI for Colon Cancer Patient Assessment
=========================================================================
A modern, two-panel GUI for entering patient clinical data and viewing
cancer severity assessment results directly on screen.

Uses CustomTkinter for a polished, clinical-grade appearance.
"""

import customtkinter as ctk
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ── Import existing ML pipeline components ──────────────────────────────────
from severity_engine import SeverityEngine

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SAVE_FILE     = 'new_patients.csv'
MODEL_PATH    = 'clinical_model.pkl'
SCALER_PATH   = 'clinical_scaler.pkl'
FEATURES_PATH = 'clinical_features.pkl'

# ── Clinical colour palette ────────────────────────────────────────────────
PRIMARY_BLUE   = "#1A6B8A"
ACCENT_TEAL    = "#2E9CAD"
LIGHT_BG       = "#EEF2F7"
CARD_BG        = "#FFFFFF"
RESULT_BG      = "#1E2A38"
TEXT_DARK      = "#1A1A2E"
TEXT_MEDIUM    = "#4A5568"
TEXT_LIGHT     = "#FFFFFF"
SUCCESS_GREEN  = "#38A169"
WARNING_YELLOW = "#D69E2E"
ALERT_ORANGE   = "#DD6B20"
CRITICAL_RED   = "#E53E3E"
INPUT_BG       = "#F7F9FC"
INPUT_BORDER   = "#CBD5E0"

# ── Severity → colour mapping ────────────────────────────────────────────────
SEVERITY_COLORS = {
    "Mild":     SUCCESS_GREEN,
    "Moderate": WARNING_YELLOW,
    "High":     ALERT_ORANGE,
    "Critical": CRITICAL_RED,
}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING (reused from New_Patient.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_model():
    """Load clinical model, scaler, and feature list from pickle files."""
    try:
        model    = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, scaler, features
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("   Please run train_clinical_model.py first.")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# ML FORMAT CONVERSION (reused from New_Patient.py)
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_ml_format(info):
    """
    Maps plain-English answers to exact binary column names
    used in the clinical model. All columns start at 0.
    Only the chosen option becomes 1.
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

    row['demographic.age_at_index']         = info['age']
    row['diagnoses.days_to_last_follow_up'] = info['follow_up_days']

    if info['gender'] == 'Male':
        row['demographic.gender_male'] = 1
    else:
        row['demographic.gender_female'] = 1

    grade_map = {
        "G1": 'diagnoses.tumor_grade_G1',
        "G2": 'diagnoses.tumor_grade_G2',
        "G3": 'diagnoses.tumor_grade_G3',
        "G4": 'diagnoses.tumor_grade_G4',
        "Not Reported": 'diagnoses.tumor_grade_Not Reported',
    }
    row[grade_map[info['tumor_grade']]] = 1

    metastasis_map = {
        "Yes":    'diagnoses.metastasis_at_diagnosis_Metastasis, NOS',
        "No":     'diagnoses.metastasis_at_diagnosis_No Metastasis',
        "Unknown": 'diagnoses.metastasis_at_diagnosis_Unknown',
    }
    row[metastasis_map[info['metastasis']]] = 1

    residual_map = {
        "R0":     'diagnoses.residual_disease_R0',
        "R1":     'diagnoses.residual_disease_R1',
        "R2":     'diagnoses.residual_disease_R2',
        "Unknown": 'diagnoses.residual_disease_RX',
    }
    row[residual_map[info['residual_disease']]] = 1

    classification_map = {
        "Primary":            'diagnoses.classification_of_tumor_primary',
        "Recurrence":         'diagnoses.classification_of_tumor_recurrence',
        "Metastasis":         'diagnoses.classification_of_tumor_metastasis',
        "Synchronous primary": 'diagnoses.classification_of_tumor_Synchronous primary',
        "Not Reported":      'diagnoses.classification_of_tumor_not reported',
    }
    row[classification_map[info['tumor_classification']]] = 1

    prior_map = {
        "Yes":     'diagnoses.prior_malignancy_yes',
        "No":      'diagnoses.prior_malignancy_no',
        "Unknown": 'diagnoses.prior_malignancy_unknown',
    }
    row[prior_map[info['prior_malignancy']]] = 1

    family_map = {
        "Yes":     'family_histories.relative_with_cancer_history_yes',
        "No":      'family_histories.relative_with_cancer_history_no',
        "Unknown": 'family_histories.relative_with_cancer_history_Unknown',
    }
    row[family_map[info['family_history']]] = 1

    smoking_map = {
        "Lifelong Non-Smoker":                   'exposures.tobacco_smoking_status_Lifelong Non-Smoker',
        "Current Smoker":                        'exposures.tobacco_smoking_status_Current Smoker',
        "Reformed Smoker (≤ 15 years ago)":     'exposures.tobacco_smoking_status_Current Reformed Smoker for < or = 15 yrs',
        "Reformed Smoker (> 15 years ago)":     'exposures.tobacco_smoking_status_Current Reformed Smoker for > 15 yrs',
        "Unknown":                               'exposures.tobacco_smoking_status_Unknown',
    }
    row[smoking_map[info['smoking']]] = 1

    return row


# ═══════════════════════════════════════════════════════════════════════════
# ASSESSMENT LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def calculate_probability(ml_row, model, scaler, features):
    df_input = pd.DataFrame([ml_row])[features]
    df_scaled = scaler.transform(df_input)
    probability = model.predict_proba(df_scaled)[0][1]
    return float(probability)


def calculate_severity(ml_row, probability):
    engine     = SeverityEngine()
    row_series = pd.Series(ml_row)
    base_level, final_level, upgrades, downgrades = \
        engine.calculate_severity(row_series, probability)

    print(f"[DEBUG] calculate_severity: base={base_level}, final={final_level}, upgrades={upgrades}, downgrades={downgrades}")

    # Ensure final_level is valid
    if final_level not in ["Mild", "Moderate", "High", "Critical"]:
        final_level = "Moderate"  # Safe default

    treatment = engine.recommend_treatment(final_level)
    print(f"[DEBUG] treatment returned: '{treatment}'")

    if upgrades:
        adjustment = f"Upgraded {base_level} → {final_level}: {', '.join(upgrades)}"
    elif downgrades:
        adjustment = f"Downgraded {base_level} → {final_level}: {', '.join(downgrades)}"
    else:
        adjustment = f"No adjustment — stays at {base_level}"

    result_dict =  {
        "Cancer_Probability":       f"{probability * 100:.2f}%",
        "Base_Level":               base_level,
        "Severity_Level":           final_level,
        "Adjustment":               adjustment,
        "Treatment_Recommendation": treatment if treatment else "Consult oncology team",
    }
    print(f"[DEBUG] result dict: {result_dict}")
    return result_dict


def save_patient(info, severity_result):
    record = {
        "age":                      info['age'],
        "gender":                   info['gender'],
        "tumor_grade":              info['tumor_grade'],
        "metastasis":               info['metastasis'],
        "residual_disease":         info['residual_disease'],
        "tumor_classification":     info['tumor_classification'],
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


# ═══════════════════════════════════════════════════════════════════════════
# GUI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class PatientAssessmentUI(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("Colon Cancer Severity Assessment — Patient Intake")
        self.geometry("1500x900")
        self.minsize(1300, 800)
        ctk.set_appearance_mode("light")

        self.model, self.scaler, self.features = load_model()

        self._build_layout()
        self._build_header()
        self._build_form_panel()
        self._build_results_panel()
        self._build_footer()

        self._current_result = None

    # ─────────────────────────────────────────────────────────────────────
    # LAYOUT
    # ─────────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=5)
        self.grid_columnconfigure(1, weight=4)

    # ─────────────────────────────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────────────────────────────

    def _build_header(self):
        header = ctk.CTkFrame(self, height=68, fg_color=PRIMARY_BLUE, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)

        # Left: icon + title
        left_frame = ctk.CTkFrame(header, fg_color="transparent")
        left_frame.pack(side="left", padx=28, pady=0)
        left_frame.place(rely=0.5, anchor="w")

        icon_lbl = ctk.CTkLabel(
            left_frame, text="🩺",
            font=ctk.CTkFont(size=26),
            text_color=TEXT_LIGHT,
        )
        icon_lbl.pack(side="left", padx=(0, 10))

        title_lbl = ctk.CTkLabel(
            left_frame,
            text="Colon Cancer Severity Assessment",
            font=ctk.CTkFont(size=19, weight="bold"),
            text_color=TEXT_LIGHT,
        )
        title_lbl.pack(side="left")

        sub_lbl = ctk.CTkLabel(
            left_frame,
            text="Patient Intake — Clinical Decision Support",
            font=ctk.CTkFont(size=10),
            text_color="#B8E0F0",
        )
        sub_lbl.pack(side="left", padx=16)

        # Right: badge
        badge = ctk.CTkFrame(header, fg_color=ACCENT_TEAL, corner_radius=14)
        badge.pack(side="right", padx=24, pady=0)
        badge.place(rely=0.5, anchor="e")

        ctk.CTkLabel(
            badge,
            text="  CDSS  ",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_LIGHT,
        ).pack(padx=8, pady=4)

    # ─────────────────────────────────────────────────────────────────────
    # FORM PANEL (left)
    # ─────────────────────────────────────────────────────────────────────

    def _build_form_panel(self):
        # Outer shell
        self.form_frame = ctk.CTkFrame(self, fg_color=LIGHT_BG, corner_radius=0)
        self.form_frame.grid(row=1, column=0, sticky="nsew", padx=(16, 4), pady=16)
        self.form_frame.grid_rowconfigure(0, weight=0)
        self.form_frame.grid_rowconfigure(1, weight=1)

        # Panel header bar
        form_header = ctk.CTkFrame(self.form_frame, fg_color=CARD_BG, height=52, corner_radius=0)
        form_header.grid(row=0, column=0, sticky="ew")
        form_header.grid_propagate(False)

        ctk.CTkLabel(
            form_header,
            text="  📋  Patient Information",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=PRIMARY_BLUE,
        ).pack(side="left", padx=20)
        ctk.CTkLabel(
            form_header,
            text="Enter clinical data below",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MEDIUM,
        ).pack(side="right", padx=20)

        # White card containing the form
        self.form_card = ctk.CTkFrame(self.form_frame, fg_color=CARD_BG, corner_radius=10)
        self.form_card.grid(row=1, column=0, sticky="nsew", padx=14, pady=(10, 14))
        self.form_card.grid_columnconfigure(0, weight=1)

        # Scrollable form content
        self.form_inner = ctk.CTkScrollableFrame(
            self.form_card,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=CARD_BG,
            scrollbar_button_hover_color=INPUT_BORDER,
        )
        self.form_inner.pack(fill="both", expand=True, padx=20, pady=(12, 4))
        self.form_inner.grid_columnconfigure(0, weight=0)   # label column
        self.form_inner.grid_columnconfigure(1, weight=1)   # input column

        # ── StringVars ─────────────────────────────────────────────────
        self._age_var        = ctk.StringVar()
        self._gender_var     = ctk.StringVar(value="Male")
        self._grade_var      = ctk.StringVar(value="G1")
        self._metastasis_var = ctk.StringVar(value="No")
        self._residual_var   = ctk.StringVar(value="R0")
        self._class_var      = ctk.StringVar(value="Primary")
        self._prior_var      = ctk.StringVar(value="No")
        self._family_var     = ctk.StringVar(value="No")
        self._smoking_var    = ctk.StringVar(value="Lifelong Non-Smoker")
        self._followup_var   = ctk.StringVar()

        # ── Field rows ────────────────────────────────────────────────
        ROW = 0
        FIELD_W   = 210   # minimum width for the label column
        INPUT_W   = 380   # minimum width for the input column

        def field_row(r, label_text, hint, widget):
            """Create a label+hint row on top, widget below spanning 2 cols."""
            # Label
            lbl = ctk.CTkLabel(
                self.form_inner, text=label_text,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=TEXT_DARK, anchor="w",
            )
            lbl.grid(row=r, column=0, columnspan=2, sticky="w", pady=(14, 0), padx=(0, 8))

            # Hint (if any)
            if hint:
                hint_lbl = ctk.CTkLabel(
                    self.form_inner, text=hint,
                    font=ctk.CTkFont(size=10),
                    text_color=TEXT_MEDIUM, anchor="w",
                )
                hint_lbl.grid(row=r, column=1, sticky="w", pady=(14, 0), padx=(0, 8))

        def input_row(r, widget, pady=(2, 10)):
            """Place a widget spanning both columns, centered."""
            widget.grid(row=r, column=0, columnspan=2, sticky="w", pady=pady, padx=(0, 8))

        # ── Age ────────────────────────────────────────────────────────
        field_row(ROW, "Age (years)", "Patient age (0–120)", ROW)
        ROW += 1
        age_entry = ctk.CTkEntry(
            self.form_inner, textvariable=self._age_var,
            placeholder_text="Enter patient age (e.g. 55)", width=INPUT_W, height=44,
            font=ctk.CTkFont(size=14),
            fg_color="#E8EEF4",
        )
        input_row(ROW, age_entry)
        ROW += 1

        # ── Gender ────────────────────────────────────────────────────
        field_row(ROW, "Gender", None, ROW)
        ROW += 1
        self._add_combo(ROW, self._gender_var, ["Male", "Female"], INPUT_W)
        ROW += 1

        # ── Tumor Grade ───────────────────────────────────────────────
        field_row(ROW, "Tumor Grade", "Cell differentiation level", ROW)
        ROW += 1
        grade_combo = ctk.CTkOptionMenu(
            self.form_inner, variable=self._grade_var,
            values=["G1", "G2", "G3", "G4", "Not Reported"],
            width=INPUT_W, height=44,
            font=ctk.CTkFont(size=14),
            dropdown_font=ctk.CTkFont(size=13),
            fg_color="#2E9CAD", button_color="#1A6B8A", button_hover_color="#145670",
            text_color=TEXT_LIGHT,
        )
        input_row(ROW, grade_combo, (2, 10))
        ROW += 1

        # ── Metastasis ────────────────────────────────────────────────
        field_row(ROW, "Metastasis", "Cancer spread at diagnosis", ROW)
        ROW += 1
        meta_combo = ctk.CTkOptionMenu(
            self.form_inner, variable=self._metastasis_var,
            values=["Yes", "No", "Unknown"],
            width=INPUT_W, height=44,
            font=ctk.CTkFont(size=14),
            dropdown_font=ctk.CTkFont(size=13),
            fg_color="#2E9CAD", button_color="#1A6B8A", button_hover_color="#145670",
            text_color=TEXT_LIGHT,
        )
        input_row(ROW, meta_combo, (2, 10))
        ROW += 1

        # ── Residual Disease ───────────────────────────────────────────
        field_row(ROW, "Residual Disease", "Tumor remaining after surgery", ROW)
        ROW += 1
        res_combo = ctk.CTkOptionMenu(
            self.form_inner, variable=self._residual_var,
            values=["R0", "R1", "R2", "Unknown"],
            width=INPUT_W, height=44,
            font=ctk.CTkFont(size=14),
            dropdown_font=ctk.CTkFont(size=13),
            fg_color="#2E9CAD", button_color="#1A6B8A", button_hover_color="#145670",
            text_color=TEXT_LIGHT,
        )
        input_row(ROW, res_combo, (2, 10))
        ROW += 1

        # ── Tumor Classification ──────────────────────────────────────
        field_row(ROW, "Tumor Classification", "Type of tumor occurrence", ROW)
        ROW += 1
        class_combo = ctk.CTkOptionMenu(
            self.form_inner, variable=self._class_var,
            values=["Primary", "Recurrence", "Metastasis", "Synchronous primary", "Not Reported"],
            width=INPUT_W, height=44,
            font=ctk.CTkFont(size=14),
            dropdown_font=ctk.CTkFont(size=13),
            fg_color="#2E9CAD", button_color="#1A6B8A", button_hover_color="#145670",
            text_color=TEXT_LIGHT,
        )
        input_row(ROW, class_combo, (2, 10))
        ROW += 1

        # ── Prior Malignancy ───────────────────────────────────────────
        field_row(ROW, "Prior Malignancy", "History of previous cancer", ROW)
        ROW += 1
        prior_combo = ctk.CTkOptionMenu(
            self.form_inner, variable=self._prior_var,
            values=["Yes", "No", "Unknown"],
            width=INPUT_W, height=44,
            font=ctk.CTkFont(size=14),
            dropdown_font=ctk.CTkFont(size=13),
            fg_color="#2E9CAD", button_color="#1A6B8A", button_hover_color="#145670",
            text_color=TEXT_LIGHT,
        )
        input_row(ROW, prior_combo, (2, 10))
        ROW += 1

        # ── Family History ──────────────────────────────────────────────
        field_row(ROW, "Family History", "Close relative with cancer history", ROW)
        ROW += 1
        fam_combo = ctk.CTkOptionMenu(
            self.form_inner, variable=self._family_var,
            values=["Yes", "No", "Unknown"],
            width=INPUT_W, height=44,
            font=ctk.CTkFont(size=14),
            dropdown_font=ctk.CTkFont(size=13),
            fg_color="#2E9CAD", button_color="#1A6B8A", button_hover_color="#145670",
            text_color=TEXT_LIGHT,
        )
        input_row(ROW, fam_combo, (2, 10))
        ROW += 1

        # ── Smoking Status ──────────────────────────────────────────────
        field_row(ROW, "Smoking Status", "Patient's smoking history", ROW)
        ROW += 1
        smoke_combo = ctk.CTkOptionMenu(
            self.form_inner, variable=self._smoking_var,
            values=[
                "Lifelong Non-Smoker",
                "Current Smoker",
                "Reformed Smoker (≤ 15 years ago)",
                "Reformed Smoker (> 15 years ago)",
                "Unknown",
            ],
            width=INPUT_W, height=44,
            font=ctk.CTkFont(size=14),
            dropdown_font=ctk.CTkFont(size=13),
            fg_color="#2E9CAD", button_color="#1A6B8A", button_hover_color="#145670",
            text_color=TEXT_LIGHT,
        )
        input_row(ROW, smoke_combo, (2, 10))
        ROW += 1

        # ── Follow-up Days ──────────────────────────────────────────────
        field_row(ROW, "Follow-up Days", "Days since last follow-up (365 = 1 year)", ROW)
        ROW += 1
        followup_entry = ctk.CTkEntry(
            self.form_inner, textvariable=self._followup_var,
            placeholder_text="Enter days since last follow-up (e.g. 365)", width=INPUT_W, height=44,
            font=ctk.CTkFont(size=14),
            fg_color="#E8EEF4",
        )
        input_row(ROW, followup_entry, (2, 10))
        ROW += 1

        # ── Buttons (placed outside scrollable area, in form_card) ─────
        self._build_form_buttons()

    def _add_combo(self, row, var, values, width):
        combo = ctk.CTkOptionMenu(
            self.form_inner, variable=var, values=values,
            width=width, height=44,
            font=ctk.CTkFont(size=14),
            dropdown_font=ctk.CTkFont(size=13),
            fg_color="#2E9CAD", button_color="#1A6B8A", button_hover_color="#145670",
            text_color=TEXT_LIGHT,
        )
        combo.grid(row=row, column=0, columnspan=2, sticky="w", pady=(2, 10), padx=(0, 8))

    def _build_form_buttons(self):
        btn_frame = ctk.CTkFrame(self.form_card, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(8, 8), padx=20)

        self._assess_btn = ctk.CTkButton(
            btn_frame,
            text="🔍  Assess Patient",
            command=self._on_assess,
            width=210, height=46,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=PRIMARY_BLUE,
            hover_color=ACCENT_TEAL,
            text_color=TEXT_LIGHT,
            corner_radius=8,
        )
        self._assess_btn.pack(side="left", padx=(0, 12))

        self._clear_btn = ctk.CTkButton(
            btn_frame,
            text="↺  Clear Form",
            command=self._on_clear,
            width=160, height=46,
            font=ctk.CTkFont(size=13),
            fg_color="#E2E8F0",
            hover_color="#CBD5E0",
            text_color=TEXT_DARK,
            corner_radius=8,
        )
        self._clear_btn.pack(side="left", padx=(0, 12))

        self._save_btn = ctk.CTkButton(
            btn_frame,
            text="💾  Save Patient",
            command=self._on_save,
            width=160, height=46,
            font=ctk.CTkFont(size=13),
            fg_color=SUCCESS_GREEN,
            hover_color="#2F8A5A",
            text_color=TEXT_LIGHT,
            corner_radius=8,
            state="disabled",
        )
        self._save_btn.pack(side="left")

    # ─────────────────────────────────────────────────────────────────────
    # RESULTS PANEL (right)
    # ─────────────────────────────────────────────────────────────────────

    def _build_results_panel(self):
        self.results_frame = ctk.CTkFrame(self, fg_color=RESULT_BG, corner_radius=12)
        self.results_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 16), pady=16)
        self.results_frame.grid_rowconfigure(0, weight=0)
        self.results_frame.grid_rowconfigure(1, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        # Title bar
        title_bar = ctk.CTkFrame(self.results_frame, fg_color="transparent", corner_radius=0)
        title_bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        ctk.CTkLabel(
            title_bar,
            text="📋  Assessment Results",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT_LIGHT,
        ).pack(side="left")

        self._severity_badge = ctk.CTkLabel(
            title_bar,
            text="",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_LIGHT,
            padx=10, pady=3,
            corner_radius=8,
        )
        self._severity_badge.pack(side="right")

        # Content scroll area
        self.results_content = ctk.CTkFrame(
            self.results_frame, fg_color="transparent", corner_radius=8
        )
        self.results_content.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.results_content.grid_columnconfigure(0, weight=1)

        # Placeholder
        self._placeholder_label = ctk.CTkLabel(
            self.results_content,
            text="⬅  Enter patient data and click\n   \"Assess Patient\" to see results",
            font=ctk.CTkFont(size=13),
            text_color="#7A8FA8",
            justify="center",
        )
        self._placeholder_label.pack(pady=80, fill="both", expand=True)

    def _show_results(self, result):
        for w in self.results_content.winfo_children():
            w.destroy()

        sev_color = SEVERITY_COLORS.get(result["Severity_Level"], TEXT_MEDIUM)

        # Update badge
        self._severity_badge.configure(
            text=f"  {result['Severity_Level'].upper()}  ",
            fg_color=sev_color,
        )

        # ── Probability card ──────────────────────────────────────────
        prob_card = ctk.CTkFrame(self.results_content, fg_color="#253548", corner_radius=10)
        prob_card.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            prob_card, text="Cancer Probability",
            font=ctk.CTkFont(size=12), text_color="#7A8FA8",
        ).pack(anchor="w", padx=18, pady=(14, 2))

        ctk.CTkLabel(
            prob_card, text=result["Cancer_Probability"],
            font=ctk.CTkFont(size=44, weight="bold"),
            text_color=TEXT_LIGHT,
        ).pack(anchor="w", padx=18, pady=(0, 14))

        # ── Severity card ────────────────────────────────────────────
        sev_card = ctk.CTkFrame(self.results_content, fg_color="#253548", corner_radius=10)
        sev_card.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            sev_card, text="Severity Level",
            font=ctk.CTkFont(size=12), text_color="#7A8FA8",
        ).pack(anchor="w", padx=18, pady=(14, 2))

        ctk.CTkLabel(
            sev_card, text=result["Severity_Level"],
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=sev_color,
        ).pack(anchor="w", padx=18, pady=(0, 4))

        ctk.CTkLabel(
            sev_card, text=f"Base: {result['Base_Level']}",
            font=ctk.CTkFont(size=12), text_color="#7A8FA8",
        ).pack(anchor="w", padx=18, pady=(0, 14))

        # ── Adjustment card ───────────────────────────────────────────
        adj_card = ctk.CTkFrame(self.results_content, fg_color="#253548", corner_radius=10)
        adj_card.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            adj_card, text="📋 Clinical Adjustment",
            font=ctk.CTkFont(size=12), text_color="#7A8FA8",
        ).pack(anchor="w", padx=18, pady=(14, 4))

        ctk.CTkLabel(
            adj_card, text=result["Adjustment"],
            font=ctk.CTkFont(size=13), text_color="#A8C4D4",
            wraplength=340, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 14))

        # ── Treatment card ─────────────────────────────────────────────
        treat_card = ctk.CTkFrame(self.results_content, fg_color="#2A3A4E", corner_radius=10)
        treat_card.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            treat_card, text="💊 Treatment Recommendation",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#FFFFFF",
        ).pack(anchor="w", padx=18, pady=(14, 8))

        # Treatment text - direct label with bright color
        treat_value = str(result.get("Treatment_Recommendation", "Consult oncology team"))
        if treat_value.strip() == "":
            treat_value = "Consult oncology team"

        ctk.CTkLabel(
            treat_card, text=treat_value,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#00FF88",
            wraplength=360, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 14))

        # ── Status ─────────────────────────────────────────────────────
        self._status_lbl = ctk.CTkLabel(
            self.results_content, text="",
            font=ctk.CTkFont(size=11), text_color=SUCCESS_GREEN,
        )
        self._status_lbl.pack(pady=(10, 0))

    # ─────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────

    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=34, fg_color="#DDE3EA", corner_radius=0)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        footer.grid_propagate(False)

        ctk.CTkLabel(
            footer,
            text="Colon Cancer Severity Assessment System  |  Clinical Decision Support",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MEDIUM,
        ).pack(side="left", padx=20)
        ctk.CTkLabel(
            footer,
            text="v1.0",
            font=ctk.CTkFont(size=9),
            text_color=TEXT_MEDIUM,
        ).pack(side="right", padx=20)

    # ─────────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────────────

    def _on_assess(self):
        try:
            age = float(self._age_var.get())
            if not (0 <= age <= 120):
                self._show_error("Age must be between 0 and 120.")
                return
        except ValueError:
            self._show_error("Please enter a valid age (a number).")
            return

        try:
            followup = float(self._followup_var.get())
            if not (0 <= followup <= 10000):
                self._show_error("Follow-up days must be between 0 and 10000.")
                return
        except ValueError:
            self._show_error("Please enter a valid follow-up days (a number).")
            return

        info = {
            "age":                  age,
            "gender":               self._gender_var.get(),
            "tumor_grade":          self._grade_var.get(),
            "metastasis":           self._metastasis_var.get(),
            "residual_disease":     self._residual_var.get(),
            "tumor_classification": self._class_var.get(),
            "prior_malignancy":     self._prior_var.get(),
            "family_history":       self._family_var.get(),
            "smoking":              self._smoking_var.get(),
            "follow_up_days":       followup,
        }

        try:
            ml_row      = convert_to_ml_format(info)
            probability = calculate_probability(ml_row, self.model, self.scaler, self.features)
            result      = calculate_severity(ml_row, probability)

            print(f"[DEBUG] _on_assess result keys: {result.keys()}")
            print(f"[DEBUG] Treatment_Recommendation in result: {'Treatment_Recommendation' in result}")
            print(f"[DEBUG] Full result: {result}")

            self._current_result = (info, result)
            self._show_results(result)
            self._save_btn.configure(state="normal")
        except Exception as e:
            import traceback
            error_msg = f"Error during assessment:\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            self._show_error(error_msg)

    def _on_clear(self):
        self._age_var.set("")
        self._gender_var.set("Male")
        self._grade_var.set("G1")
        self._metastasis_var.set("No")
        self._residual_var.set("R0")
        self._class_var.set("Primary")
        self._prior_var.set("No")
        self._family_var.set("No")
        self._smoking_var.set("Lifelong Non-Smoker")
        self._followup_var.set("")

        self._current_result = None
        self._save_btn.configure(state="disabled")
        self._severity_badge.configure(text="", fg_color="transparent")

        for w in self.results_content.winfo_children():
            w.destroy()
        self._placeholder_label = ctk.CTkLabel(
            self.results_content,
            text="⬅  Enter patient data and click\n   \"Assess Patient\" to see results",
            font=ctk.CTkFont(size=13),
            text_color="#7A8FA8",
            justify="center",
        )
        self._placeholder_label.pack(pady=80, fill="both", expand=True)

    def _on_save(self):
        if not self._current_result:
            return
        info, result = self._current_result
        total = save_patient(info, result)
        self._status_lbl.configure(text=f"✅ Patient saved — {total} total record(s)")

    def _show_error(self, message):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Input Error")
        dialog.geometry("420x150")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        ctk.CTkLabel(
            dialog, text="⚠  " + message,
            font=ctk.CTkFont(size=13),
            wraplength=360, justify="center",
        ).pack(pady=28, padx=24)

        ctk.CTkButton(
            dialog, text="OK",
            command=dialog.destroy,
            width=110, height=38,
            font=ctk.CTkFont(size=12),
            fg_color=PRIMARY_BLUE, text_color=TEXT_LIGHT,
            corner_radius=6,
        ).pack(pady=(0, 20))


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = PatientAssessmentUI()
    app.mainloop()
