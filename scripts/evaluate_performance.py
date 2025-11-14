import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
import joblib

X_test = pd.read_csv('../splits/X_test_for_eval.csv')
y_test = pd.read_csv('../splits/y_test_for_eval.csv').squeeze()

model_names = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'XGBoost']
models = {name: joblib.load(f'../models/{name}_model.joblib') for name in model_names}

results = {}
for name, model in models.items():
    print(f'\nEvaluating {name}...')
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        y_proba = y_pred
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

results_df = pd.DataFrame(results).T
print('\nModel Comparison:')
print(results_df)
results_df[["accuracy","f1","roc_auc"]].plot(kind='bar', figsize=(10,5))
plt.ylabel('Score')
plt.title('Model Metric Comparison')
plt.axhline(0.9, color='red', linestyle='--', label='90% Target')
plt.legend()
plt.show()
