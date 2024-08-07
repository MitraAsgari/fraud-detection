import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def download_and_extract_data():
    # Download dataset from Kaggle
    os.environ['KAGGLE_USERNAME'] = 'Your_Kaggle_Username'
    os.environ['KAGGLE_KEY'] = 'Your_Kaggle_API_Key'
    os.system('kaggle datasets download -d mlg-ulb/creditcardfraud')
    
    # Extract the zip file
    os.system('unzip creditcardfraud.zip')

def load_and_preprocess_data():
    # Load the data
    df = pd.read_csv('creditcard.csv')
    
    # Display the first rows of the data
    print(df.head())

    # Check for missing data
    print(df.isnull().sum())

    # Distribution of fraudulent and non-fraudulent data
    print(df['Class'].value_counts())

    # Separate features and labels
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    download_and_extract_data()
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
