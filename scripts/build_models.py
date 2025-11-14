import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. LOAD PROCESSED FEATURES & LABELS
X = pd.read_csv('credit_features_processed.csv')
y = pd.read_csv('credit_labels_processed.csv').squeeze()  # Get as Series
print('Features shape:', X.shape, 'Labels shape:', y.shape)

# 2. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print('Train:', X_train.shape, 'Test:', X_test.shape)

# 3. TRAIN MULTIPLE MODELS
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
trained_models = {}

for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train.values.ravel())
    joblib.dump(model, f'{name}_model.joblib')
    trained_models[name] = model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name} Test Accuracy: {acc:.4f}')

# 4. RANDOM FOREST HYPERPARAMETER TUNING (EXAMPLE)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params,
                    scoring='accuracy', cv=5, n_jobs=-1)
grid.fit(X_train, y_train.values.ravel())
print('Best Params (Random Forest):', grid.best_params_)
print('Best Score:', grid.best_score_)
# Save tuned model
joblib.dump(grid.best_estimator_, 'RandomForest_gridsearch_best_model.joblib')

# 5. SAVE TEST SPLIT FOR EVALUATION
X_test.to_csv('X_test_for_eval.csv', index=False)
y_test.to_csv('y_test_for_eval.csv', index=False, header=True)

print('\nAll models trained and saved. Ready for evaluation!')
