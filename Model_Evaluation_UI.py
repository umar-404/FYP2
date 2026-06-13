"""
Model_Evaluation_UI.py — Clinical Performance Evaluation & Patient Triage Panel
==============================================================================
Provides an integrated diagnostic dashboard visualizing regularized baselines
performance matrices, overfitting verification plots, feature importance,
and dynamic patient triage records using your custom SeverityEngine.
"""

import customtkinter as ctk
import tkinter as tki
import pandas as pd
import numpy as np
import os
import sys
import joblib

# ── Dummy class so pickle can resolve FullInferencePipeline from trained_rf_model.pkl ──
class FullInferencePipeline:
    def __init__(self, selector, scaler, model):
        self.selector = selector
        self.scaler = scaler
        self.model = model
    def predict_proba(self, X_raw):
        X_sel = self.selector.transform(X_raw)
        X_sca = self.scaler.transform(X_sel)
        return self.model.predict_proba(X_sca)
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATHS = {
    "cm": os.path.join(SCRIPT_DIR, '01_Confusion_Matrices.png'),
    "gap": os.path.join(SCRIPT_DIR, '02_Accuracy_Gap_Line.png'),
    "pr": os.path.join(SCRIPT_DIR, '03_Precision_Recall_Scatter.png'),
    "feat": os.path.join(SCRIPT_DIR, '04_Feature_Importance.png')
}

# ── Clinical UI Palette ────────────────────────────────────────────────────
PRIMARY_BLUE   = "#1A6B8A"
ACCENT_TEAL    = "#2E9CAD"
LIGHT_BG       = "#EEF2F7"
CARD_BG        = "#FFFFFF"
TEXT_DARK      = "#1A1A2E"
TEXT_MEDIUM    = "#4A5568"
TEXT_LIGHT     = "#FFFFFF"
BAR_BG         = "#E2E8F0"

CRITICAL_RED   = "#E53E3E"
SUCCESS_GREEN  = "#38A169"

# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION INTERFACE CLASS
# ═══════════════════════════════════════════════════════════════════════════

class ModelEvaluationUI(ctk.CTk):
    """Main dashboard environment for baseline machine learning models."""

    def __init__(self):
        super().__init__()

        self.title("Clinical Diagnostics Hub — Model Evaluation UI Panel")
        self.geometry("1340x880")
        self.minsize(1200, 780)
        ctk.set_appearance_mode("light")

        # Fixed placeholder list mirroring the heavily regularized metrics table
        self.metrics_data = [
            {"Model": "Decision_Tree", "Train_Acc": 0.7924, "Test_Acc": 0.8839, "Precision": 0.9861, "Recall": 0.8554, "F1_Score": 0.9161},
            {"Model": "SVM_Soft", "Train_Acc": 0.8281, "Test_Acc": 0.8750, "Precision": 0.9726, "Recall": 0.8554, "F1_Score": 0.9103},
            {"Model": "Logistic_Reg", "Train_Acc": 0.8348, "Test_Acc": 0.8571, "Precision": 0.9589, "Recall": 0.8434, "F1_Score": 0.8974},
            {"Model": "XGBoost_Reg", "Train_Acc": 0.7746, "Test_Acc": 0.8393, "Precision": 0.9851, "Recall": 0.7952, "F1_Score": 0.8800},
            {"Model": "Random_Forest", "Train_Acc": 0.8125, "Test_Acc": 0.8304, "Precision": 0.9571, "Recall": 0.8072, "F1_Score": 0.8758}
        ]
        self.df_metrics = pd.DataFrame(self.metrics_data)

        self._build_layout()
        self._build_header()
        self._build_left_sidebar_panel()
        self._build_right_content_tabs()
        self._build_footer()

    def _build_layout(self):
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Primary workspace
        self.grid_rowconfigure(2, weight=0)  # Footer
        self.grid_columnconfigure(0, weight=0)  # Left Data Panel
        self.grid_columnconfigure(1, weight=1)  # Right Chart Panel

    def _build_header(self):
        header = ctk.CTkFrame(self, height=64, fg_color=PRIMARY_BLUE, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)

        title_lbl = ctk.CTkLabel(
            header, text="📊  Clinical Machine Learning Diagnostic Engine — Model Evaluation UI",
            font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT_LIGHT
        )
        title_lbl.pack(side="left", padx=24)

    def _build_left_sidebar_panel(self):
        """Builds static metrics summary metrics rows and triage submission fields."""
        panel = ctk.CTkFrame(self, width=420, fg_color=LIGHT_BG, corner_radius=0)
        panel.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        panel.grid_propagate(False)

        ctk.CTkLabel(
            panel, text="Consolidated Performance Table",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_DARK
        ).pack(anchor="w", padx=20, pady=(20, 10))

        # Core Metrics Display Table Panel Card
        table_card = ctk.CTkScrollableFrame(panel, fg_color=CARD_BG, corner_radius=8, height=300)
        table_card.pack(fill="x", padx=16, pady=4)
        table_card.grid_columnconfigure((0, 1, 2), weight=1)

        # Header Columns
        headers = ["Model Name", "Test Acc", "F1 Score"]
        for ci, h_text in enumerate(headers):
            ctk.CTkLabel(
                table_card, text=h_text, font=ctk.CTkFont(size=11, weight="bold"),
                text_color=PRIMARY_BLUE, anchor="w" if ci == 0 else "e"
            ).grid(row=0, column=ci, sticky="ew", padx=8, pady=6)

        # Populate structured diagnostic rows (Sorted by F1 Score)
        for idx, row in self.df_metrics.iterrows():
            bg_color = "#F8FAFC" if idx % 2 == 0 else CARD_BG
            row_frame = ctk.CTkFrame(table_card, fg_color=bg_color, corner_radius=4)
            row_frame.grid(row=idx+1, column=0, columnspan=3, sticky="ew", pady=2)
            row_frame.grid_columnconfigure(0, weight=2)
            row_frame.grid_columnconfigure((1, 2), weight=1)

            ctk.CTkLabel(row_frame, text=row["Model"], font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_DARK, anchor="w").grid(row=0, column=0, sticky="ew", padx=8, pady=4)
            ctk.CTkLabel(row_frame, text=f"{row['Test_Acc']:.2%}", font=ctk.CTkFont(size=11), text_color=TEXT_MEDIUM, anchor="e").grid(row=0, column=1, sticky="ew", padx=8, pady=4)
            ctk.CTkLabel(row_frame, text=f"{row['F1_Score']:.2%}", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_TEAL, anchor="e").grid(row=0, column=2, sticky="ew", padx=8, pady=4)

        # Real-time Patient Evaluation Box
        self._build_patient_triage_module(panel)

    def _build_patient_triage_module(self, parent):
        triage_card = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=8)
        triage_card.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            triage_card, text="Quick Patient Severity Triage Engine",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=PRIMARY_BLUE
        ).pack(anchor="w", padx=16, pady=(12, 4))

        input_frame = ctk.CTkFrame(triage_card, fg_color="transparent")
        input_frame.pack(fill="x", padx=16, pady=8)

        ctk.CTkLabel(input_frame, text="Patient Target Index:", font=ctk.CTkFont(size=12), text_color=TEXT_DARK).pack(side="left")
        
        self.patient_id_entry = ctk.CTkEntry(input_frame, width=100, height=28)
        self.patient_id_entry.insert(0, "501")
        self.patient_id_entry.pack(side="left", padx=12)

        triage_btn = ctk.CTkButton(
            triage_card, text="Run Evaluation Profile", fg_color=ACCENT_TEAL,
            hover_color=PRIMARY_BLUE, height=32, font=ctk.CTkFont(size=12, weight="bold"),
            command=self._execute_live_triage
        )
        triage_btn.pack(fill="x", padx=16, pady=(4, 12))

        # Log Output Box
        self.output_box = ctk.CTkTextbox(triage_card, height=180, font=ctk.CTkFont(family="monospace", size=11), fg_color="#F8FAFC", border_width=1, border_color=BAR_BG)
        self.output_box.pack(fill="x", padx=16, pady=(0, 16))
        self.output_box.insert("0.0", "System idle. Enter a patient index to compute live severity engine variables...")

    def _build_right_content_tabs(self):
        """Creates tabview wrappers for tracking exported figures."""
        self.tab_view = ctk.CTkTabview(self, fg_color=LIGHT_BG, text_color=PRIMARY_BLUE, segmented_button_selected_color=PRIMARY_BLUE)
        self.tab_view.grid(row=1, column=1, sticky="nsew", padx=16, pady=12)

        self.tab_view.add("Confusion Matrices")
        self.tab_view.add("Generalization Check")
        self.tab_view.add("Precision-Recall Tradeoff")
        self.tab_view.add("Predictive Feature Importance")

        self._render_image_in_tab("Confusion Matrices", IMAGE_PATHS["cm"], (740, 480))
        self._render_image_in_tab("Generalization Check", IMAGE_PATHS["gap"], (740, 440))
        self._render_image_in_tab("Precision-Recall Tradeoff", IMAGE_PATHS["pr"], (700, 480))
        self._render_image_in_tab("Predictive Feature Importance", IMAGE_PATHS["feat"], (740, 480))

    def _render_image_in_tab(self, tab_name, image_path, size):
        target_tab = self.tab_view.tab(tab_name)
        
        if os.path.exists(image_path):
            try:
                pil_img = Image.open(image_path)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=size)
                
                img_lbl = ctk.CTkLabel(target_tab, image=ctk_img, text="")
                img_lbl.pack(expand=True, fill="both", padx=10, pady=10)
            except Exception as e:
                ctk.CTkLabel(target_tab, text=f"Error displaying image matrix layer: {e}").pack(pady=20)
        else:
            ctk.CTkLabel(
                target_tab, text=f"⚠️ Resource missing:\n{os.path.basename(image_path)}\n\nPlease execute your training script first to write visual plots.",
                font=ctk.CTkFont(size=12), text_color=TEXT_MEDIUM
            ).pack(expand=True)

    def _execute_live_triage(self):
        """Loads real serialized models and runs evaluation via your SeverityEngine."""
        from severity_engine import SeverityEngine

        self.output_box.delete("0.0", "end")
        patient_str = self.patient_id_entry.get().strip()

        if not patient_str.isdigit():
            self.output_box.insert("0.0", "❌ Error: Please enter a valid numerical patient index.")
            return

        patient_idx = int(patient_str)

        # Attempt tracking data loads
        try:
            data = pd.read_csv('ML_READY_DATA.csv')
            X = data.drop(columns=['target'])
            
            with open('trained_rf_model.pkl', 'rb') as f:
                rf_model = joblib.load(f)
                
        except FileNotFoundError as e:
            self.output_box.insert(
                "0.0", 
                f"❌ Pipeline Execution Aborted:\nMissing file: {os.path.basename(e.filename)}\n\n"
                f"Please ensure you run your training script to generate the tracking pickle models first!"
            )
            return

        # Verification boundary checking
        if patient_idx < 0 or patient_idx >= len(X):
            self.output_box.insert(
                "0.0", 
                f"❌ Index Out of Bounds!\n\n"
                f"The loaded dataset contains patient records from index 0 to {len(X)-1}.\n\n"
                f"Index {patient_idx} does not exist."
            )
            return

        try:
            # Generate actual mathematical risks probability scores vector maps
            y_prob_full = rf_model.predict_proba(X)[:, 1]
            
            # Spin up your exact clinical verification engine module file logic
            severity_engine = SeverityEngine()
            result = severity_engine.predict_patient_severity(
                X,
                y_prob_full,
                patient_index=patient_idx
            )
            
            self.output_box.insert("0.0", str(result))
            
        except Exception as e:
            self.output_box.insert("0.0", f"❌ Inference Engine Core Exception:\n{str(e)}")

    def _build_footer(self):
        footer = ctk.CTkFrame(self, height=30, fg_color="#DDE3EA", corner_radius=0)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        
        ctk.CTkLabel(
            footer, text="Clinical Classification Pipeline Dashboard Environment | Model Evaluation UI Module",
            font=ctk.CTkFont(size=10), text_color=TEXT_MEDIUM
        ).pack(side="left", padx=20, pady=4)

if __name__ == "__main__":
    app = ModelEvaluationUI()
    app.mainloop()