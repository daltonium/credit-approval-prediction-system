import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
DATA_PATH = 'credit_approval_data.csv'  # Change this if you have another file name

# You can also fetch directly from UCI or your own download
# df = pd.read_csv(DATA_PATH, header=None)
df = pd.read_csv(DATA_PATH)

print('Initial Data Shape:', df.shape)
print(df.head())

# 2. DATA EXPLORATION
print('\n--- Data Info ---')
df.info()
print('\n--- Describe (Numeric Columns) ---')
print(df.describe())
print('\n--- Missing Values ---')
print(df.isnull().sum())

# Visualize missing values (Optional)
plt.figure(figsize=(8,4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title('Missing Value Heatmap')
plt.show()

# Class balance (target is usually the last column, e.g., 'A16' or 'target')
target_col = df.columns[-1]
print('\n--- Target Value Counts ---')
print(df[target_col].value_counts())
df[target_col].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# 3. MISSING VALUE HANDLING
# Replace '?' or blanks with NaN, then impute
for col in df.columns:
    df[col].replace('?', np.nan, inplace=True)

print('\n--- NA Count After Replacement ---')
print(df.isnull().sum())

# Separate by data type (object likely categorical, others likely numeric)
data_num = df.select_dtypes(include=[np.number])
data_cat = df.select_dtypes(exclude=[np.number])

# Impute numerical with median, categorical with mode
for col in data_num.columns:
    if df[col].isnull().any():
        median = df[col].median()
        df[col].fillna(median, inplace=True)
for col in data_cat.columns:
    if df[col].isnull().any():
        mode = df[col].mode()[0]
        df[col].fillna(mode, inplace=True)

# Double-check missing values removed
df.isnull().sum().sum()

print('\n--- Data Shape After Cleaning ---')
print(df.shape)

# 4. ADDRESS IMBALANCED DATA (If severely imbalanced, show recommendation)
target_counts = df[target_col].value_counts(normalize=True)
imbalance = target_counts.max() > 0.7
if imbalance:
    print('WARNING: Imbalanced data detected! Major class exceeds 70%. Consider resampling in modeling step.')
else:
    print('Class balance is acceptable.')

# 5. SAVE CLEAN DATA FOR NEXT STEP
CLEAN_PATH = 'credit_approval_cleaned.csv'
df.to_csv(CLEAN_PATH, index=False)
print(f'Data cleaned and saved to {CLEAN_PATH}')
