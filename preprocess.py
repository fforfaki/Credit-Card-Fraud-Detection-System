import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(filepath):
    """Load the credit card fraud dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean() * 100:.2f}%)")
    return df


def preprocess(df):
    """Scale features and split into train/test sets."""
    # Scale 'Amount' and 'Time' columns
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time']   = scaler.fit_transform(df[['Time']])

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to training set only
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print(f"After SMOTE — Training samples: {X_train_res.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train_res, X_test, y_train_res, y_test
