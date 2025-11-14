import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score

X_test = pd.read_csv('../splits/X_test_for_eval.csv')
y_test = pd.read_csv('../splits/y_test_for_eval.csv').squeeze()
best_model = joblib.load('../models/XGBoost_model.joblib')  # or your preferred model

sensitive_feature = 'A1'  # Change to available feature
if sensitive_feature not in X_test.columns:
    print(f"Column {sensitive_feature} not found in test data! Can't run fairness check.")
    exit()

y_pred = best_model.predict(X_test)
X_test[sensitive_feature] = X_test[sensitive_feature].astype(str)
groups = X_test[sensitive_feature].unique()

print(f"\n=== Fairness Metrics by Group ({sensitive_feature}): ===")
group_rates = {}
for group in groups:
    mask = X_test[sensitive_feature] == group
    selection_rate = (y_pred[mask] == 1).mean()
    group_precision = precision_score(y_test[mask], y_pred[mask])
    group_recall = recall_score(y_test[mask], y_pred[mask])
    group_rates[group] = selection_rate
    print(f"Group {group}: Selection Rate={selection_rate:.3f}, Precision={group_precision:.3f}, Recall={group_recall:.3f}")
min_rate, max_rate = min(group_rates.values()), max(group_rates.values())
dem_parity_ratio = min_rate / max_rate if max_rate > 0 else 0
print(f"\nDemographic Parity Ratio (min/max selection rate): {dem_parity_ratio:.3f}")
if dem_parity_ratio < 0.8:
    print("Warning: Potential disparate impact detected (ratio < 0.8)")
else:
    print("Demographic parity is acceptable (ratio >= 0.8)")
