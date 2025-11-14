import pandas as pd
import joblib
import numpy as np

# 1. LOAD TRAINED MODEL, SCALER, ENCODERS
best_model = joblib.load('XGBoost_model.joblib')  # or your preferred/best model
scaler = joblib.load('feature_scaler.joblib')

# List input features in order used in training (adjust as needed)
input_features = [
    'A9', 'A11', 'A8', 'A3', 'A15', 'A14', 'A2', 'debt_to_income', 'A6', 'A10'
]
# List categorical features (and which need LabelEncoder)
categorical_cols = ['A9', 'A6', 'A10']  # adjust based on your project
encoders = {col: joblib.load(f'label_encoder_{col}.joblib') for col in categorical_cols}

# List numerical features (for scaling)
numerical_cols = ['A11', 'A8', 'A3', 'A15', 'A14', 'A2', 'debt_to_income']  # adjust as appropriate

def get_user_input():
    user_data = []
    print("Please enter applicant details:")
    for feat in input_features:
        val = input(f"{feat}: ")
        if feat in categorical_cols:
            try:
                val = encoders[feat].transform([val])[0]
            except Exception as e:
                print(f"Invalid input for {feat}. Acceptable: {list(encoders[feat].classes_)}")
                return None
        else:
            try:
                val = float(val)
            except:
                print(f"{feat} must be a number.")
                return None
        user_data.append(val)
    return user_data

user_data = get_user_input()
if user_data is not None:
    user_df = pd.DataFrame([user_data], columns=input_features)
    # Scale numerical columns only
    user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])
    # Predict
    prediction = best_model.predict(user_df)
    if prediction[0] == 1:
        print("Result: APPROVED")
    else:
        print("Result: DENIED")
