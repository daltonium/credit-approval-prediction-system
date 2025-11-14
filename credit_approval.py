# 1. Load the Dataset

from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch the dataset
data = fetch_ucirepo(id=27)
X = data.data.features
y = data.data.targets

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
rf.fit(X_clean, y.replace({'+': 1, '-': 0}))

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
