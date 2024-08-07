# Install necessary libraries
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve

# Download dataset from Kaggle
!pip install kaggle
import os
os.environ['KAGGLE_USERNAME'] = 'Your_Kaggle_Username'
os.environ['KAGGLE_KEY'] = 'Your_Kaggle_API_Key'
!kaggle datasets download -d mlg-ulb/creditcardfraud

# Extract the zip file
!unzip creditcardfraud.zip

# Load the data
df = pd.read_csv('creditcard.csv')

# Display the first rows of the data
print(df.head())

# Check for missing data
print(df.isnull().sum())

# Distribution of fraudulent and non-fraudulent data
print(df['Class'].value_counts())

# Visualize the distribution of the data
sns.countplot(x='Class', data=df)
plt.title('Distribution of Classes')
plt.show()

# Separate features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
