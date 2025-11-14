import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# 1. LOAD CLEANED DATA
df = pd.read_csv('credit_approval_cleaned.csv')
print('Loaded cleaned data:', df.shape)

# 2. SPLIT FEATURES & TARGET (Assume last column is target)
target_col = df.columns[-1]
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# 3. ENCODE TARGET (if needed)
# UCI original has +/-, others map these to 1/0; change if needed
if y.dtype == object or set(y.unique()) == {'+', '-'}:
    y = y.map({'+':1, '-':0})
else:
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    joblib.dump(le_y, 'target_label_encoder.joblib')

# 4. IDENTIFY CATEGORICAL & NUMERICAL (based on dtype)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# 5. ENCODE CATEGORICAL FEATURES (Label Encoding; save encoders)
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    joblib.dump(le, f'label_encoder_{col}.joblib')

# 6. ENGINEER NEW FEATURES (optional, e.g., debt_to_income)
# Example: debt_to_income = A8 / A2  if both columns exist and numeric
if set(['A8','A2']).issubset(X.columns):
    # Add small value to denominator to prevent div/0
    X['debt_to_income'] = (X['A8'].astype(float) / (X['A2'].astype(float) + 1e-5)).round(4)
    numerical_cols.append('debt_to_income')

# 7. SCALE NUMERICAL FEATURES
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
joblib.dump(scaler, 'feature_scaler.joblib')

# 8. SAVE PROCESSED DATASETS
X.to_csv('credit_features_processed.csv', index=False)
pd.Series(y).to_csv('credit_labels_processed.csv', index=False, header=['target'])
print('Saved processed features and labels for modeling.')
