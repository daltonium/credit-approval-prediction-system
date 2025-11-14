import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

CLEAN_PATH = 'data/credit_approval_cleaned.csv'
df = pd.read_csv(CLEAN_PATH, header=None)
print('Loaded cleaned data:', df.shape)

TARGET_COL = df.columns[-1]
X = df.drop(columns=[TARGET_COL]).copy()
y = df[TARGET_COL].copy()

if y.dtype == object or set(y.unique()) == {'+', '-'}:
    y = y.map({'+': 1, '-': 0})
else:
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    joblib.dump(le_y, 'models/target_label_encoder.joblib')

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    joblib.dump(le, f'models/label_encoder_{col}.joblib')

# Optional: add debt_to_income if columns 1 and 7 exist (A2 and A8 in UCI dataset)
if 1 in X.columns and 7 in X.columns:
    X['debt_to_income'] = (X[7].astype(float) / (X[1].astype(float) + 1e-5)).round(4)
    numerical_cols.append('debt_to_income')

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
joblib.dump(scaler, 'models/feature_scaler.joblib')

X.to_csv('data/credit_features_processed.csv', index=False)
pd.Series(y).to_csv('data/credit_labels_processed.csv', index=False, header=['target'])
print('Processed features/labels saved.')
