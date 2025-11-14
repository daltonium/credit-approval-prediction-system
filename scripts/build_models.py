import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X = pd.read_csv('data/credit_features_processed.csv')
y = pd.read_csv('data/credit_labels_processed.csv').squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train.values.ravel())
    joblib.dump(model, f'models/{name}_model.joblib')
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f'{name} Test Accuracy: {acc:.4f}')

rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, scoring='accuracy', cv=5, n_jobs=-1)
grid.fit(X_train, y_train.values.ravel())
print('Best Params (Random Forest):', grid.best_params_)
joblib.dump(grid.best_estimator_, 'models/RandomForest_gridsearch_best_model.joblib')

X_test.to_csv('splits/X_test_for_eval.csv', index=False)
y_test.to_csv('splits/y_test_for_eval.csv', index=False, header=True)
print('All models trained and saved.')
