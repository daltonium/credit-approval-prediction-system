import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Load trained model and scaler
best_model = joblib.load('models/XGBoost_model.joblib')
scaler = joblib.load('models/feature_scaler.joblib')

# Load available encoders
available_encoders = {}
for file in os.listdir('models'):
    if file.startswith('label_encoder_') and file.endswith('.joblib'):
        col = file.replace('label_encoder_', '').replace('.joblib', '')
        available_encoders[col] = joblib.load(f'models/{file}')

# Feature mapping with meaningful descriptions
FEATURE_MAP = {
    '1': {'name': 'Age', 'type': 'numerical', 'description': 'Age in years (e.g., 25, 30, 45)'},
    '2': {'name': 'Years Employed', 'type': 'numerical', 'description': 'Years at current job (e.g., 0.5, 2, 10)'},
    '3': {'name': 'Marital Status', 'type': 'categorical', 'description': 'u=unmarried, y=married, l=living together'},
    '4': {'name': 'Bank Customer Type', 'type': 'categorical', 'description': 'g=standard, gg=premium, p=basic'},
    '5': {
        'name': 'Education Level', 
        'type': 'categorical', 
        'description': 'Choose your highest education:',
        'mappings': {
            'aa': 'High School',
            'c': 'Associate Degree',
            'cc': 'Bachelor\'s Degree',
            'd': 'Some College',
            'e': 'Master\'s Degree',
            'ff': 'Professional Degree',
            'i': 'Vocational Training',
            'j': 'Doctorate',
            'k': 'Technical Certification',
            'm': 'Trade School',
            'q': 'Less than High School',
            'r': 'GED',
            'w': 'Graduate Certificate',
            'x': 'Other'
        }
    },
    '6': {
        'name': 'Occupation Type', 
        'type': 'categorical', 
        'description': 'Select your occupation category:',
        'mappings': {
            'bb': 'Professional/Technical',
            'dd': 'Management/Executive',
            'ff': 'Sales',
            'h': 'Administrative/Clerical',
            'j': 'Service Industry',
            'n': 'Construction/Trades',
            'o': 'Healthcare',
            'v': 'Education/Teaching',
            'z': 'Other/Retired'
        }
    },
    '7': {'name': 'Years at Current Address', 'type': 'numerical', 'description': 'Years (e.g., 0, 2, 5, 10)'},
    '8': {'name': 'Employment Status', 'type': 'categorical', 'description': 't=employed, f=unemployed'},
    '9': {'name': 'Credit History', 'type': 'categorical', 'description': 't=good, f=poor'},
    '10': {'name': 'Driver License', 'type': 'numerical', 'description': '1=yes, 0=no'},
    '11': {'name': 'Citizenship', 'type': 'categorical', 'description': 't=citizen, f=non-citizen'},
    '12': {'name': 'Credit Type Requested', 'type': 'categorical', 'description': 'g=standard, p=platinum, s=secured'},
    '13': {'name': 'Monthly Income', 'type': 'numerical', 'description': 'Monthly income in dollars (e.g., 2000, 3500, 5000)'},
    '14': {'name': 'Total Debt', 'type': 'numerical', 'description': 'Total debt in dollars (e.g., 0, 500, 2000)'}
}

def get_user_friendly_input():
    """Get input from user with clear descriptions"""
    user_data = {}
    print("\n" + "="*60)
    print("        CREDIT CARD APPROVAL PREDICTION SYSTEM")
    print("="*60)
    print("\nPlease provide the following information:\n")
    
    # Get sample data to know column order
    X_sample = pd.read_csv('data/credit_features_processed.csv', nrows=1)
    input_features = X_sample.columns.tolist()
    
    for feat in input_features:
        if feat not in FEATURE_MAP:
            user_data[feat] = 0
            continue
            
        feature_info = FEATURE_MAP[feat]
        print(f"📋 {feature_info['name']}")
        print(f"   {feature_info['description']}")
        
        if feature_info['type'] == 'categorical' and feat in available_encoders:
            valid_values = list(available_encoders[feat].classes_)
            
            # Check if we have meaningful mappings
            if 'mappings' in feature_info:
                mappings = feature_info['mappings']
                print("\n   Options:")
                for code, label in mappings.items():
                    if code in valid_values:
                        print(f"      {code} = {label}")
                print()
            else:
                print(f"   Valid options: {', '.join(map(str, valid_values))}")
            
            val = input(f"   Enter code: ").strip()
            
            try:
                val = available_encoders[feat].transform([val])[0]
            except:
                default = valid_values[0]
                print(f"   ⚠️  Invalid input! Using default: {default}")
                val = available_encoders[feat].transform([default])[0]
        else:
            val = input(f"   Enter value: ").strip()
            try:
                val = float(val)
            except:
                print(f"   ⚠️  Invalid number! Using 0")
                val = 0.0
        
        user_data[feat] = val
        print()
    
    return user_data, input_features

# Get user input
print("\n" + "🏦" * 30)
user_data, input_features = get_user_friendly_input()

# Create DataFrame
user_df = pd.DataFrame([user_data])

# Identify numerical columns for scaling
numerical_cols = [col for col in input_features if col not in available_encoders.keys()]

# Scale numerical features
if numerical_cols:
    user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])

# Make prediction
prediction = best_model.predict(user_df)[0]
proba = best_model.predict_proba(user_df)[0]

# Display result
print("="*60)
print("                   PREDICTION RESULT")
print("="*60)

if prediction == 1:
    print("\n✅ ✅ ✅  CREDIT CARD APPROVED  ✅ ✅ ✅\n")
    print(f"   Model Confidence: {proba[1]*100:.2f}%")
    print("\n   Congratulations! You are eligible for a credit card.")
else:
    print("\n❌ ❌ ❌  CREDIT CARD DENIED  ❌ ❌ ❌\n")
    print(f"   Model Confidence: {proba[0]*100:.2f}%")
    print("\n   Unfortunately, your application was not approved.")
    print("   Possible reasons:")
    print("   - Insufficient income or employment history")
    print("   - Poor credit history")
    print("   - High debt-to-income ratio")

print("\n" + "="*60 + "\n")
