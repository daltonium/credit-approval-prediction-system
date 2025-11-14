import pandas as pd
import joblib
import numpy as np

best_model = joblib.load('../models/XGBoost_model.joblib')  # or your selected model
scaler = joblib.load('../models/feature_scaler.joblib')

input_features = [
    'A9', 'A11', 'A8', 'A3', 'A15', 'A14', 'A2', 'debt_to_income', 'A6', 'A10'
]
categorical_cols = ['A9', 'A6', 'A10']  # Adjust as needed
encoders = {col: joblib.load(f'../models/label_encoder_{col}.joblib') for col in categorical_cols}
numerical_cols = ['A11', 'A8', 'A3', 'A15', 'A14', 'A2', 'debt_to_income']  # Adjust as appropriate

def get_user_input():
    user_data = []
    print("Please enter applicant details:")
    for feat in input_features:
        val = input(f"{feat}: ")
        if feat in categorical_cols:
            try:
                val = encoders[feat].transform([val])[0]
            except Exception:
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
    user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])
    prediction = best_model.predict(user_df)
    print('Result: APPROVED' if prediction[0] == 1 else 'Result: DENIED')
