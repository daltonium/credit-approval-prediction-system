import pandas as pd
import joblib
import os

# Load trained model and scaler
best_model = joblib.load('models/XGBoost_model.joblib')
scaler = joblib.load('models/feature_scaler.joblib')

# Get actual feature names from a saved processed file
X_sample = pd.read_csv('data/credit_features_processed.csv', nrows=1)
input_features = X_sample.columns.tolist()

print("Model expects these features:", input_features)

# Identify which encoders exist
available_encoders = {}
for col in input_features:
    encoder_path = f'models/label_encoder_{col}.joblib'
    if os.path.exists(encoder_path):
        available_encoders[col] = joblib.load(encoder_path)

categorical_cols = list(available_encoders.keys())
numerical_cols = [col for col in input_features if col not in categorical_cols]

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

def get_user_input():
    user_data = {}
    print("\n=== Credit Approval Prediction ===")
    print("Please enter applicant details:\n")
    
    for feat in input_features:
        if feat in categorical_cols:
            valid_values = list(available_encoders[feat].classes_)
            val = input(f"{feat} (valid: {valid_values}): ")
            try:
                val = available_encoders[feat].transform([val])[0]
            except:
                print(f"Invalid input for {feat}. Using default value.")
                val = available_encoders[feat].transform([valid_values[0]])[0]
        else:
            val = input(f"{feat} (number): ")
            try:
                val = float(val)
            except:
                print(f"Invalid number for {feat}. Using 0.")
                val = 0.0
        user_data[feat] = val
    
    return user_data

# Get user input
user_data = get_user_input()
user_df = pd.DataFrame([user_data])

# Scale numerical columns only
if numerical_cols:
    user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])

# Predict
prediction = best_model.predict(user_df)[0]
proba = best_model.predict_proba(user_df)[0]

print("\n" + "="*50)
if prediction == 1:
    print("✅ RESULT: CREDIT APPROVED")
    print(f"Confidence: {proba[1]*100:.1f}%")
else:
    print("❌ RESULT: CREDIT DENIED")
    print(f"Confidence: {proba[0]*100:.1f}%")
print("="*50)
