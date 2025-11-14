import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = '../data/credit_approval_data.csv'  # Change if needed

df = pd.read_csv(DATA_PATH, header=None)  # Add header=None for UCI dataset
print('Initial Data Shape:', df.shape)
print(df.head())

# Quick fix for missing values if "?" is used
for col in df.columns:
    df[col].replace('?', np.nan, inplace=True)

print('\n--- Missing Values ---')
print(df.isnull().sum())

# Quick exploration/visualization
plt.figure(figsize=(8,4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title('Missing Value Heatmap')
plt.show()

# Assume last column is target
TARGET_COL = df.columns[-1]
print("\n--- Target Value Counts ---")
print(df[TARGET_COL].value_counts())
df[TARGET_COL].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Impute missing values
for col in df.columns:
    if df[col].dtype == object:
        mode = df[col].mode()[0]
        df[col].fillna(mode, inplace=True)
    else:
        median = df[col].median()
        df[col].fillna(median, inplace=True)

print('\n--- Data Shape After Cleaning ---')
print(df.shape)

CLEAN_PATH = '../data/credit_approval_cleaned.csv'
df.to_csv(CLEAN_PATH, index=False, header=False)  # header=False for UCI
print(f'Data cleaned and saved to {CLEAN_PATH}')
