import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Load cleaned data
CLEAN_PATH = '../data/credit_approval_cleaned.csv'
df = pd.read_csv(CLEAN_PATH, header=None)
print('Loaded cleaned data:', df.shape)

# Target assumed last column
TARGET_COL = df.columns[-1]
X = df.drop(columns=[TARGET_COL]).copy()
y = df[TARGET_COL].copy()

# Encode target
if y.dtype == object or set(y.unique()) == {'+', '-'}:
    y = y.map({'+': 1, '-': 0})
else:
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    joblib.dump(le_y, '../models/target_label_encoder.joblib')

# Identify categorical & numerical
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Label encode categoricals and save encoders
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    joblib.dump(le, f'../models/label_encoder_{col}.joblib')

# Feature engineering: add debt_to_income if A8/A2 exist
if set([8, 1]).issubset(X.columns):    # UCI: A8==col 8, A2==col 1
    X['debt_to_income'] = (X[8].astype(float) / (X[1].astype(float) + 1e-5)).round(4)
    numerical_cols.append('debt_to_income')

# Scale numericals and save scaler
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
joblib.dump(scaler, '../models/feature_scaler.joblib')

X.to_csv('../data/credit_features_processed.csv', index=False)
pd.Series(y).to_csv('../data/credit_labels_processed.csv', index=False, header=['target'])
print('Processed features/labels saved to ../data/')
