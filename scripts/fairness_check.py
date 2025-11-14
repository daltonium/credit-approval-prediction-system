import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score
# Optional: from fairlearn.metrics import demographic_parity_ratio

# 1. LOAD DATA & MODEL
X_test = pd.read_csv('X_test_for_eval.csv')
y_test = pd.read_csv('y_test_for_eval.csv').squeeze()
best_model = joblib.load('XGBoost_model.joblib')  # or your preferred model

# 2. CHOOSE SENSITIVE FEATURE (e.g., 'A1' for gender)
sensitive_feature = 'A1'  # Change to your available group feature
if sensitive_feature not in X_test.columns:
    print(f"Column {sensitive_feature} not found in test data! Can't run fairness check.")
    exit()

# 3. MAKE PREDICTIONS
y_pred = best_model.predict(X_test)
X_test[sensitive_feature] = X_test[sensitive_feature].astype(str)

groups = X_test[sensitive_feature].unique()
print("\n=== Fairness Metrics by Group ({}): ===".format(sensitive_feature))

# 4. SELECTION RATE AND METRICS PER GROUP
group_rates = {}
for group in groups:
    mask = X_test[sensitive_feature] == group
    selection_rate = (y_pred[mask] == 1).mean()
    group_precision = precision_score(y_test[mask], y_pred[mask])
    group_recall = recall_score(y_test[mask], y_pred[mask])
    group_rates[group] = selection_rate
    print(f"Group {group}: Selection Rate={selection_rate:.3f}, Precision={group_precision:.3f}, Recall={group_recall:.3f}")

# 5. DEMOGRAPHIC PARITY RATIO
min_rate = min(group_rates.values())
max_rate = max(group_rates.values())
dem_parity_ratio = min_rate / max_rate if max_rate > 0 else 0
print(f"\nDemographic Parity Ratio (min/max selection rate): {dem_parity_ratio:.3f}")
if dem_parity_ratio < 0.8:
    print("Warning: Potential disparate impact detected (ratio < 0.8)")
else:
    print("Demographic parity is acceptable (ratio >= 0.8)")

# Advanced: If using fairlearn
# from fairlearn.metrics import demographic_parity_ratio
# ratio = demographic_parity_ratio(y_true=y_test, y_pred=y_pred, sensitive_features=X_test[sensitive_feature])
# print('Fairlearn Demographic Parity Ratio:', ratio)
