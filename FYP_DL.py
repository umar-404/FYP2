"""
deep_learning_pipeline.py — Professional Deep Learning Clinical Core Engine
==============================================================================
Implements an isolated Multi-Layer Perceptron (MLP) Neural Network for high-
dimensional colon cancer screening risk prediction.

Features: Dynamic Feature Selection, Loss Convergence Mapping, Custom Tabular
          Sample Weights, and Granular Terminal Benchmarking Matrix Views.
"""

import os
import sys
import warnings
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_sample_weight

# ═══════════════════════════════════════════════════════════════════════════
# 1. SETUP ENGINE & ENVIRONMENTAL STANDARDS
# ═══════════════════════════════════════════════════════════════════════════
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")  # Clean darkgrid standard for neural convergence charts

print("="*80)
print("🧬 INITIALIZING CLINICAL DEEP LEARNING ANALYSIS ENGINE")
print("="*80)

# ── Load Live Dataset ──────────────────────────────────────────────────────
try:
    data = pd.read_csv('ML_READY_DATA.csv') 
    X = data.drop(columns=['target'])
    y = data['target']
    # print(f"✅ Source Matrix Verified: {X.shape[0]} patient rows, {X.shape[1]} original features.")
except FileNotFoundError:
    print("❌ Fatal Exception: 'ML_READY_DATA.csv' not detected in current working root.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION & DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
# Stratified partition guarantees test data mirrors clinical class imbalances
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("⚡ Running ANOVA Feature Filter (Dropping noise variables)...")
selector = SelectKBest(score_func=f_classif, k=25)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Normalization layer configuration
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Compute inverse frequency mapping weights to neutralize imbalance issues
sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
print("📊 Imbalance Correction Vector applied cleanly to training batches.")

# ═══════════════════════════════════════════════════════════════════════════
# 3. CONSTRUCT DEEP NEURAL NETWORK MATRIX (MLP ARCHITECTURE)
# ═══════════════════════════════════════════════════════════════════════════
print("\n🚀 Building Deep Feedforward Multi-Layer Perceptron Network...")

# Architecture: 25 Optimized Inputs -> Hidden 1 (64) -> Hidden 2 (32) -> Output (1)
mlp_network = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',            # Rectified Linear Unit for handling non-linear variations
    solver='adam',                # Stochastic gradient-based optimizer
    alpha=0.08,                   # Enhanced L2 regularization parameter to block overfitting
    batch_size=32,                # Iteration batch slice size
    learning_rate_init=0.001,     # Initial learning stride parameters
    max_iter=1500,                # High ceiling loop bounds
    early_stopping=True,          # Auto-terminates if internal validation plateaus
    validation_fraction=0.15,     # 15% isolated evaluation checkpoint frame
    random_state=42
)

# Execute neural learning sequence loop
mlp_network.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
print("🏁 Network convergence finalized.")

# ═══════════════════════════════════════════════════════════════════════════
# 4. GRANULAR TERMINAL PERFORMANCE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════
train_preds = mlp_network.predict(X_train_scaled)
test_preds = mlp_network.predict(X_test_scaled)
test_probs = mlp_network.predict_proba(X_test_scaled)[:, 1]

print("\n" + "═"*80)
print("DEEP LEARNING MODEL PERFORMANCE REPORT (TERMINAL OUTPUT)")
print("═"*80)
print(f"• Network Train Accuracy : {accuracy_score(y_train, train_preds):.2%}")
print(f"• Network Test Accuracy  : {accuracy_score(y_test, test_preds):.2%}")
print(f"• F1-Score Assessment    : {f1_score(y_test, test_preds, zero_division=0):.2%}")
print(f"• Precision Index Value  : {precision_score(y_test, test_preds, zero_division=0):.2%}")
print(f"• Recall Sensitivity Rate: {recall_score(y_test, test_preds, zero_division=0):.2%}")
print("═"*80)

print("\n📋 SCALED CLASSIFICATION MATRIX ANALYSIS:")
print(classification_report(y_test, test_preds, target_names=["Class 0 (Low Risk)", "Class 1 (High Risk)"]))

# Print terminal ASCII confusion representation layout
cm = confusion_matrix(y_test, test_preds)
print("📊 RAW CONFUSION MATRIX ARRAY VALUES:")
print(f"   [True Negatives: {cm[0][0]:03d}]   [False Positives: {cm[0][1]:03d}]")
print(f"   [False Negatives: {cm[1][0]:03d}]   [True Positives:  {cm[1][1]:03d}]\n")

# ═══════════════════════════════════════════════════════════════════════════
# 5. RESEARCH VISUALIZATION EXPORTS
# ═══════════════════════════════════════════════════════════════════════════
print("📈 Processing advanced training visualizations plots...")

# Chart A: Neural Convergence Loss Curve Graph
plt.figure(figsize=(10, 5))
plt.plot(mlp_network.loss_curve_, color='#1A6B8A', linewidth=2.5, label='Training Loss Index')
if mlp_network.early_stopping:
    plt.plot(mlp_network.validation_scores_, color='#DD6B20', linewidth=2, linestyle='--', label='Validation Accuracy Vector')
plt.title("Deep Learning Pipeline Convergence: Loss Value Profile Across Epochs", fontdict={'weight': 'bold', 'size': 12})
plt.xlabel("Training Iterations (Epoch Count)")
plt.ylabel("Loss / Score Value Margin")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('05_Deep_Learning_Loss_Curve.png', dpi=300)
plt.close()

# Chart B: Deep Learning Dedicated Confusion Matrix Heatmap
plt.figure(figsize=(7, 5.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['Predicted Low', 'Predicted High'],
            yticklabels=['Actual Low', 'Actual High'],
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title("Deep Learning MLP Confusion Matrix", fontdict={'weight': 'bold', 'size': 12}, pad=12)
plt.tight_layout()
plt.savefig('06_Deep_Learning_Confusion_Matrix.png', dpi=300)
plt.close()

print("📁 Visualization artifacts successfully exported to PNG records:")
print("   ↳ '05_Deep_Learning_Loss_Curve.png'")
print("   ↳ '06_Deep_Learning_Confusion_Matrix.png'")
print("="*80)


# Place this at the absolute bottom of deep_learning_pipeline.py to save the assets:
class DeepInferencePipeline:
    def __init__(self, selector, scaler, model):
        self.selector = selector
        self.scaler = scaler
        self.model = model
    def predict_proba(self, X_raw):
        X_sel = self.selector.transform(X_raw)
        X_sca = self.scaler.transform(X_sel)
        return self.model.predict_proba(X_sca)

pipeline_wrapper = DeepInferencePipeline(selector, scaler, mlp_network)

with open('trained_mlp_model.pkl', 'wb') as f:
    pickle.dump(pipeline_wrapper, f)
print("✅ Deep Learning UI Assets exported successfully.")