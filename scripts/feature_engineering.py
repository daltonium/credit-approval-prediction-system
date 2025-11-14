import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load cleaned data
CLEAN_PATH = 'data/credit_approval_cleaned.csv'
df = pd.read_csv(CLEAN_PATH, header=None)
print('Loaded cleaned data:', df.shape)

# Target assumed last column
TARGET_COL = df.columns[-1]
X = df.drop(columns=[TARGET_COL]).copy()
y = df[TARGET_COL].copy()

# --- FAIRNESS MITIGATION: Remove sensitive feature ---
# Column 0 ('b') showed disparate impact in fairness check
SENSITIVE_FEATURES = [0]  # Add more if needed
print(f"Removing sensitive feature(s) to improve fairness: {SENSITIVE_FEATURES}")
X = X.drop(columns=SENSITIVE_FEATURES, errors='ignore')
print(f"Features after removal: {X.shape[1]} (was {df.shape[1]-1})")

# Encode target
if y.dtype == object or set(y.unique()) == {'+', '-'}:
    y = y.map({'+': 1, '-': 0})
else:
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    joblib.dump(le_y, 'models/target_label_encoder.joblib')

# Identify categorical & numerical
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Label encode categoricals and save encoders
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    joblib.dump(le, f'models/label_encoder_{col}.joblib')

# Optional: Feature engineering - add debt_to_income if columns 1 and 7 exist
# Note: After dropping column 0, indices shift! Column 1 is now at index 0, column 7 is now at 6
# Be careful with column references after dropping
if 0 in X.columns and 6 in X.columns:  # Adjusted indices
    try:
        X['debt_to_income'] = (X[6].astype(float) / (X[0].astype(float) + 1e-5)).round(4)
        numerical_cols.append('debt_to_income')
        print("Added engineered feature: debt_to_income")
    except:
        print("Could not create debt_to_income feature")

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
joblib.dump(scaler, 'models/feature_scaler.joblib')

# Save processed datasets
X.to_csv('data/credit_features_processed.csv', index=False)
pd.Series(y).to_csv('data/credit_labels_processed.csv', index=False, header=['target'])
print('Processed features/labels saved (sensitive features removed).')
print('Now retrain your models with: python build_models.py')
