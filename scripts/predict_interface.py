import pandas as pd
import joblib

best_model = joblib.load('models/XGBoost_model.joblib')
scaler = joblib.load('models/feature_scaler.joblib')

input_features = list(range(15))  # Adjust to your actual feature count
categorical_cols = [0, 3, 4, 5, 6, 8, 9, 11, 12]  # Adjust based on your encoding
encoders = {col: joblib.load(f'models/label_encoder_{col}.joblib') for col in categorical_cols}
numerical_cols = [i for i in input_features if i not in categorical_cols]

def get_user_input():
    user_data = []
    print("Please enter applicant details:")
    for feat in input_features:
        val = input(f"Feature {feat}: ")
        if feat in categorical_cols:
            try:
                val = encoders[feat].transform([val])[0]
            except:
                print(f"Invalid input for feature {feat}.")
                return None
        else:
            try:
                val = float(val)
            except:
                print(f"Feature {feat} must be a number.")
                return None
        user_data.append(val)
    return user_data

user_data = get_user_input()
if user_data is not None:
    user_df = pd.DataFrame([user_data], columns=input_features)
    user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])
    prediction = best_model.predict(user_df)
    print('Result: APPROVED' if prediction == 1 else 'Result: DENIED')
