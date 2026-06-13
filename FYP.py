import pandas as pd
import numpy as np
import warnings
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from severity_engine import SeverityEngine

# 1. SETUP & DATA LOADING
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

try:
    data = pd.read_csv('ML_READY_DATA.csv') 
    X = data.drop(columns=['target'])
    y = data['target']
    print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
except FileNotFoundError:
    print("❌ Error: 'ML_READY_DATA.csv' not found.")
    exit()

# Calculate scale_pos_weight for XGBoost imbalance control
num_neg = np.sum(y == 0)
num_pos = np.sum(y == 1)
imbalance_ratio = num_neg / num_pos if num_pos > 0 else 1.0

# 2. PREPROCESSING (SPLIT & SCALE ONLY - SMOTE REMOVED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (Essential for Gradient Descent models like Logistic Reg & SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. DEFINE HEAVILY REGULARIZED MODELS
models = {
    # Lowered C to increase L2 penalty strengths
    "Logistic_Reg": LogisticRegression(penalty='l2', C=0.01, class_weight='balanced', max_iter=2000),
    
    # Standardized shallow bounds to minimize split memorization
    "Decision_Tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=15, class_weight='balanced', random_state=42),
    
    # Restricted depths and enabled sample bootstrapping features
    "Random_Forest": RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_split=25, min_samples_leaf=10, class_weight='balanced', random_state=42),
    
    # Highly regularized soft-margin linear vector spaces
    "SVM_Soft": SVC(kernel='linear', C=0.005, class_weight='balanced', probability=True, random_state=42),
    
    # Added subsample and robust lambda penalties to crush ensemble variance
    "XGBoost_Reg": XGBClassifier(max_depth=3, reg_lambda=50, reg_alpha=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=imbalance_ratio, eval_metric='logloss', random_state=42)
}

# 4. TRAINING LOOP & METRICS COLLECTION
results = []
probs_dict = {}

# Setup Confusion Matrix Grid
fig_cm, axes_cm = plt.subplots(2, 3, figsize=(18, 10))
axes_cm = axes_cm.flatten()

print("🚀 Training heavily regularized models...")

for i, (name, model) in enumerate(models.items()):
    # Train on standard SCALED training features
    model.fit(X_train_scaled, y_train)
    
    # Run predictions across paths
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Compile performance tracking structures
    metrics = {
        "Model": name,
        "Train_Acc": accuracy_score(y_train, train_preds),
        "Test_Acc": accuracy_score(y_test, test_preds),
        "Precision": precision_score(y_test, test_preds, zero_division=0),
        "Recall": recall_score(y_test, test_preds, zero_division=0),
        "F1_Score": f1_score(y_test, test_preds, zero_division=0)
    }
    results.append(metrics)
    probs_dict[name] = test_probs

    # Plot CM
    cm = confusion_matrix(y_test, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes_cm[i], cmap='Blues', cbar=False)
    axes_cm[i].set_title(f"CM: {name}")

# Save Confusion Matrices
fig_cm.delaxes(axes_cm[5])
plt.tight_layout()
plt.savefig('01_Confusion_Matrices.png')
plt.close()

# 5. VISUALIZATIONS
df_res = pd.DataFrame(results)

# A. Accuracy Gap (Line Chart)
plt.figure(figsize=(12, 6))
plt.plot(df_res['Model'], df_res['Train_Acc'], marker='o', label='Train (Original)', color='blue')
plt.plot(df_res['Model'], df_res['Test_Acc'], marker='s', label='Test (Original)', linestyle='--', color='red')
plt.title("Generalization Check: Train vs Test Accuracy")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('02_Accuracy_Gap_Line.png')
plt.close()

# B. Precision-Recall Tradeoff (Scatter Plot)
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_res, x='Precision', y='Recall', hue='Model', s=200)
plt.title("Clinical Utility: Precision vs Recall")
plt.axhline(0.9, color='green', linestyle=':', label='90% Recall Target')
plt.legend()
plt.savefig('03_Precision_Recall_Scatter.png')
plt.close()

# C. Feature Importance (Random Forest)
rf_model = models["Random_Forest"]
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title("Top 15 Predictive Clinical Features")
plt.savefig('04_Feature_Importance.png')
plt.close()

# 6. FINAL OUTPUT
pd.options.display.float_format = '{:.2%}'.format
print("\n" + "="*80)
print("FINAL CONSOLIDATED PERFORMANCE TABLE")
print("="*80)
print(df_res.sort_values(by="F1_Score", ascending=False))
print("="*80)
print("\n📁 All visualizations saved to PNG files in current directory.")

# 7. EXPORT PIPELINE ARTIFACTS FOR CORE UI
y_prob_full = models["Random_Forest"].predict_proba(scaler.transform(X))[:, 1]

severity_engine = SeverityEngine()
result = severity_engine.predict_patient_severity(X, y_prob_full, patient_index=501)
print(result)

print("\n📦 Exporting clinical pipeline serialized components...")
with open('trained_rf_model.pkl', 'wb') as f:
    pickle.dump(models["Random_Forest"], f)

with open('dataset_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Core pipeline artifacts saved successfully.")