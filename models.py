import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the ML Ready dataset
data = pd.read_csv('ML_READY_DATA.csv')

# 2. Separate Features (X) and Target (y)
X = data.drop(columns=['target'])
y = data['target']

# 3. Split the data (Same split for all models for a fair test)
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the 3 Models
log_reg = LogisticRegression(max_iter=1000)
dec_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. Train and Predict
# Model A: Logistic Regression
log_reg.fit(X_train, y_train)
log_acc = accuracy_score(y_test, log_reg.predict(X_test))

# Model B: Decision Tree
dec_tree.fit(X_train, y_train)
tree_acc = accuracy_score(y_test, dec_tree.predict(X_test))

# Model C: Random Forest
rand_forest.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rand_forest.predict(X_test))

# 6. Compare the Accuracies
print("--- ACCURACY COMPARISON ---")
print(f"Logistic Regression: {log_acc:.2%}")
print(f"Decision Tree:       {tree_acc:.2%}")
print(f"Random Forest:       {rf_acc:.2%}")

# 7. Find the winner
results = {
    "Logistic Regression": log_acc,
    "Decision Tree": tree_acc,
    "Random Forest": rf_acc
}
best_model = max(results, key=results.get)
print(f"\nWinner for your FYP: {best_model} ({results[best_model]:.2%})")