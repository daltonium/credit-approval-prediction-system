import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = 'data/credit_approval_data.csv'
df = pd.read_csv(DATA_PATH, header=None)
print('Initial Data Shape:', df.shape)

for col in df.columns:
    df[col].replace('?', np.nan, inplace=True)

print('\n--- Missing Values ---')
print(df.isnull().sum())

TARGET_COL = df.columns[-1]
print("\n--- Target Value Counts ---")
print(df[TARGET_COL].value_counts())

for col in df.columns:
    if df[col].dtype == object:
        mode = df[col].mode()[0] if not df[col].mode().empty else 'missing'
        df[col].fillna(mode, inplace=True)
    else:
        median = df[col].median()
        df[col].fillna(median, inplace=True)

CLEAN_PATH = 'data/credit_approval_cleaned.csv'
df.to_csv(CLEAN_PATH, index=False, header=False)
print(f'Data cleaned and saved to {CLEAN_PATH}')
