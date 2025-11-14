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
print(df.head())
print(df.info())
print(df.describe())

# 3. Visualize Data
import matplotlib.pyplot as plt
import seaborn as sns

# Just see counts for approval ('+' and '-')
sns.countplot(x=df.columns[-1], data=df)
plt.title('Credit Approval Status Counts')
plt.show()

sns.histplot(df['A2'].dropna(), bins=20)
plt.title('Distribution of A2')
plt.show()
