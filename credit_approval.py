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
print(y.value_counts())
print(y.value_counts(normalize=True))

#8. Weighted Loss Functions
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
# Then fit as usual
