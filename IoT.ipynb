# Connect to SQL database
import sqlite3
conn = sqlite3.connect('database.db')

# Check what tables are available
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

# Check what fields are available
cursor.execute("PRAGMA table_info(table_name)")
print(cursor.fetchall())

# Get a few columns from the database
cursor.execute("SELECT column_1, column_2 FROM table_name")
data = cursor.fetchall()

# Get all the data
cursor.execute("SELECT * FROM table_name")
data = cursor.fetchall()

# Convert SQL result into a DataFrame
import pandas as pd
df = pd.DataFrame(data)

# Split data into train and testing sets to prevent data snooping bias
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target_column'), df['target_column'], test_size=0.2)

# Explore sensor data - bird's eye overview
print(df.describe())

# Take a look at distributions
import seaborn as sns
sns.pairplot(df)

# Calculate Pearson's correlation
corr = df.corr(method='pearson')
print(corr)

# Check if the target is balanced
counts = df['target_column'].value_counts()
print(counts)

# Preprocessing for IoT - split features and targets
X = df.drop(columns='target_column')
y = df['target_column']

# Rebalance data using Imblearn and SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X, y = smote.fit_resample(X, y)

# Reduce dimensionality with PCA
from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Train a PCA model
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)

# Evaluate components using a scree plot
import matplotlib.pyplot as plt
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Number of Components')
plt.ylabel('Proportion of Explained Variance')
plt.show()
