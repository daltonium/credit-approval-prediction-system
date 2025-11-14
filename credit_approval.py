# 1. Load the Dataset

from ucimlrepo import fetch_ucirepo
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

# Fetch the dataset
data = fetch_ucirepo(id=27)
X = data.data.features
y = data.data.targets
y_numeric = y.replace({'+': 1, '-': 0}).astype(int)

# Combine features and target for easy viewing
df = pd.concat([X, y], axis=1)

# 2. Examine the Structure
#print(df.head())
#print(df.info())
#print(df.describe())

# 3. Visualize Data
import matplotlib.pyplot as plt
import seaborn as sns

# Just see counts for approval ('+' and '-')
sns.countplot(x=df.columns[-1], data=df)
#plt.title('Credit Approval Status Counts')
#plt.show()

sns.histplot(df['A2'].dropna(), bins=20)
#plt.title('Distribution of A2')
#plt.show()

# 4. Handle Missing Values
from sklearn.impute import SimpleImputer
import numpy as np

# Separate numerical and categorical columns
data_num = X.select_dtypes(include=[np.number])
data_cat = X.select_dtypes(exclude=[np.number])

# Median imputation for numerics
imputer_num = SimpleImputer(strategy='median')
data_num = pd.DataFrame(imputer_num.fit_transform(data_num), columns=data_num.columns)

# Mode imputation for categoricals
imputer_cat = SimpleImputer(strategy='most_frequent')
data_cat = pd.DataFrame(imputer_cat.fit_transform(data_cat), columns=data_cat.columns)

# Combine back for next steps
X_clean = pd.concat([data_num, data_cat], axis=1)

# 5. Encode Categorical Variables
from sklearn.preprocessing import LabelEncoder

for col in data_cat.columns:
    le = LabelEncoder()
    X_clean[col] = le.fit_transform(X_clean[col].astype(str))

# 6. Normalize Numerical Features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_clean[data_num.columns] = scaler.fit_transform(X_clean[data_num.columns])

# 7. Check Class Balance
#print(y.value_counts())
#print(y.value_counts(normalize=True))

#8. Weighted Loss Functions
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
# Then fit as usual

# 8. Creating Meaningful Features
if 'A2' in X_clean.columns and 'A8' in X_clean.columns:
    X_clean['debt_to_income'] = X_clean['A8'] / (X_clean['A2'] + 1e-5)  # add small constant to avoid zero division

# 9. Feature Selection with Correlation Analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Temporarily add y to X_clean for correlation
X_y = X_clean.copy()
X_y['target'] = y.replace({'+': 1, '-': 0}).astype(int)

corr = X_y.corr()['target'].drop('target')
#print("Correlation with target:\n", corr)

# Visualize
plt.figure(figsize=(10, 4))
sns.barplot(x=corr.index, y=corr.values)
plt.xticks(rotation=45)
plt.title('Feature correlation with target')
#plt.show()

# 10. Feature Importance from Model
from sklearn.ensemble import RandomForestClassifier

# Fit with class_weight since you already decided that
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_clean, y_numeric.values.ravel())  

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# print("Feature ranking:")
# for f in range(len(indices)):
#     print(f"{f + 1}. {X_clean.columns[indices[f]]}: {importances[indices[f]]:.4f}")

# Visualize the top 10 features
plt.figure(figsize=(10, 5))
sns.barplot(x=X_clean.columns[indices][:10], y=importances[indices][:10])
plt.xticks(rotation=45)
plt.title('Random Forest Feature Importances')
#plt.show()

# 11. Remove Low-Impact or Redundant Features
top_features = X_clean.columns[indices][:10]
X_selected = X_clean[top_features]

# Optionally, overwrite X_clean
X_clean = X_selected

# 12. Split Data into Train & Test Sets
from sklearn.model_selection import train_test_split

# Convert target to numeric
y_numeric = y.replace({'+': 1, '-': 0}).astype(int)

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
)

# 13. Build & Train Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=1, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and predict
def train_and_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train.values.ravel())  
    score = model.score(X_test, y_test.values.ravel())  
    print(f"{type(model).__name__} Test Accuracy: {score:.4f}")
    return model

trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    trained_models[name] = train_and_score(model, X_train, y_train, X_test, y_test)

# 14. Hyperparameter Tuning: GridSearchCV Example
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                    param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train.values.ravel())

print("Best Params (Random Forest):", grid.best_params_)
print("Best Score:", grid.best_score_)

# 15. Import Metric Functions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 16. Make Predictions and Collect Probabilities (Test Set)
results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    # For ROC AUC, get probability estimates if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }
    print(f"\nModel: {name}")
    roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc_str}")

    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    #plt.show()

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve')
        plt.legend()
        #plt.show()

# 17. Compare All Models
import pandas as pd
results_df = pd.DataFrame(results).T  # .T to get models as rows
print("\nModel Comparison:")
print(results_df)

# Optional: Bar plot of key metrics
results_df[["accuracy", "f1", "roc_auc"]].plot(kind='bar', figsize=(10, 5))
plt.ylabel('Score')
plt.title('Model Metric Comparison')
plt.axhline(0.9, color='red', linestyle='--', label='90% Target')
plt.legend()
#plt.show()
