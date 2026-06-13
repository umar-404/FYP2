import pandas as pd
import numpy as np
import warnings
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
from imblearn.over_sampling import SMOTE

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

# 2. PREPROCESSING (SPLIT, SCALE, BALANCE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (Essential for Gradient Descent models like Logistic Reg & SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class Balancing (SMOTE) - Only apply to training data
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print(f"📊 Class Distribution after SMOTE: {np.bincount(y_train_bal)}")

# 3. DEFINE REGULARIZED MODELS
models = {
    "Logistic_Reg": LogisticRegression(penalty='l2', C=0.1, max_iter=2000),
    "Decision_Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=20, random_state=42),
    "SVM_Soft": SVC(kernel='linear', C=0.01, probability=True, random_state=42),
    "XGBoost_Reg": XGBClassifier(reg_lambda=15, reg_alpha=2, learning_rate=0.05, random_state=42)
}

# 4. TRAINING LOOP & METRICS COLLECTION
results = []
probs_dict = {}

# Setup Confusion Matrix Grid
fig_cm, axes_cm = plt.subplots(2, 3, figsize=(18, 10))
axes_cm = axes_cm.flatten()

print("🚀 Training models...")

for i, (name, model) in enumerate(models.items()):
    # Train on BALANCED data
    model.fit(X_train_bal, y_train_bal)
    
    # Predict on ORIGINAL test data
    train_preds = model.predict(X_train_bal)
    test_preds = model.predict(X_test_scaled)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    metrics = {
        "Model": name,
        "Train_Acc": accuracy_score(y_train_bal, train_preds),
        "Test_Acc": accuracy_score(y_test, test_preds),
        "Precision": precision_score(y_test, test_preds),
        "Recall": recall_score(y_test, test_preds),
        "F1_Score": f1_score(y_test, test_preds)
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
plt.plot(df_res['Model'], df_res['Train_Acc'], marker='o', label='Train (Balanced)', color='blue')
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
print(df_results := df_res.sort_values(by="F1_Score", ascending=False))
print("="*80)
print("\n📁 All visualizations saved to PNG files in current directory.")

y_prob_full = models["Random_Forest"].predict_proba(scaler.transform(X))[:, 1]

severity_engine = SeverityEngine()

result = severity_engine.predict_patient_severity(
    X,
    y_prob_full,
    patient_index=501  # ← change this number to any patient
)

print(result)