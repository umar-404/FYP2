import pandas as pd
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegressionopycop
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit

# 1. LOAD DATA
print("📂 Loading data...")
df = pd.read_csv("advanced_genomic_data.csv")

# 2. ENCODE CATEGORICAL FEATURES
le = LabelEncoder()
df['Ref_Enc'] = le.fit_transform(df['Ref'].astype(str))
df['Alt_Enc'] = le.fit_transform(df['Alt'].astype(str))

# 3. FEATURES, TARGET, AND GROUPS
features = ['Ref_Enc', 'Pos', 'Alt_Enc', 'Vol_Diff', 'Hydro_Diff']
X = df[features]
y = df['Target']
groups = df['Gene Name']  # Using full Gene Name as the separating boundary

# Split so that the same Gene NEVER appears in both Train and Test sets
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train_raw, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train_raw, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ⚖️ FIX OVERFITTING: Balance ONLY the Training data to eliminate baseline cheating
print("⚖️ Balancing training dataset classes to break baseline accuracy inflation...")
train_df_raw = X_train_raw.copy()
train_df_raw['Target'] = y_train_raw

drivers_train = train_df_raw[train_df_raw['Target'] == 1]
passengers_train = train_df_raw[train_df_raw['Target'] == 0]

# Sample passengers to be 1.5 times the drivers in the training pool
sampled_passengers_train = passengers_train.sample(n=int(len(drivers_train) * 1.5), random_state=42)

# Recombine into our finalized training arrays
balanced_train_df = pd.concat([drivers_train, sampled_passengers_train]).sample(frac=1, random_state=42)
X_train = balanced_train_df[features]
y_train = balanced_train_df['Target']

# 4. TRAIN MODELS FOR EVALUATION TABLE
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1),
    "XGBoost": XGBClassifier(tree_method='hist'),
    "LightGBM": LGBMClassifier(n_estimators=100, verbose=-1),
    "CatBoost": CatBoostClassifier(iterations=100, silent=True)
}

results = []
print(f"\n{'='*75}")
print(f"{'TERMINAL EVALUATION: DRIVER VS PASSENGER PREDICTION':^75}")
print(f"{'='*75}")

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": f"{acc:.4f}",
        "F1 Score": f"{f1:.4f}",
        "Time (s)": f"{time.time()-start:.2f}"
    })

print(pd.DataFrame(results).to_string(index=False))
print("="*75)

# 5. GENERATE ORIGINAL RANKINGS 
print("\n Calculating AI Danger Scores for all genes...")

# Predict the probability using XGBoost (matching your original format)
df['Danger_Score'] = models["XGBoost"].predict_proba(X)[:, 1]

# Reverting back to your exact grouping logic (Mean score + original Gene Names)
rankings = df.groupby('Gene Name')['Danger_Score'].mean().sort_values(ascending=False).reset_index()

# 6. OUTPUT AND SAVE (Top 10 vs Top 100)
print("\n TOP-10 AI GENE RANKING :")
print(rankings.head(10))

# Export JSON files with original column structures intact
top_10 = rankings.head(10)
top_10.to_json("ui_top_10_chart.json", orient='records')

top_100 = rankings.head(100)
top_100.to_json("ui_top_100_table.json", orient='records')

print("\n✅ SUCCESS!")
print("- Terminal format restored to match your original configuration perfectly.")
print("- 'ui_top_10_chart.json' and 'ui_top_100_table.json' saved successfully.")