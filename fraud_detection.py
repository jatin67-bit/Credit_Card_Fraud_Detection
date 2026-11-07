import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter

def load_data(csv_path: str = None) -> pd.DataFrame:
    """Download from KaggleHub or load from existing path."""
    if csv_path and os.path.exists(csv_path):
        print(f"Loading dataset from: {csv_path}")
        return pd.read_csv(csv_path)

    print("Downloading Credit Card Fraud dataset via kagglehub...")
    import kagglehub
    dataset_dir = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_file = os.path.join(dataset_dir, "creditcard.csv")
    print(f"Dataset downloaded to: {csv_file}")
    return pd.read_csv(csv_file)

def train_and_evaluate(df: pd.DataFrame, model_output_path: str = "fraud_model.pkl"):
    print("\n--- 1. Exploratory Data Analysis ---")
    print("Dataset Shape:", df.shape)
    counts = df['Class'].value_counts()
    print("Class Counts:\n", counts)
    fraud_pct = (counts[1] / len(df)) * 100
    print(f"Fraud Percentage: {fraud_pct:.3f}% (Extreme Class Imbalance)")

    print("\n--- 2. Train-Test Stratified Split ---")
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Training Samples: {len(X_train)} (Frauds: {y_train.sum()})")
    print(f"Testing Samples : {len(X_test)} (Frauds: {y_test.sum()})")

    print("\n--- 3. Handling Imbalance via SMOTE ---")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("Resampled Training Distribution:", Counter(y_train_res))

    print("\n--- 4. Training Random Forest Classifier ---")
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    start_train = time.time()
    model.fit(X_train_res, y_train_res)
    train_duration = time.time() - start_train
    print(f"Training completed in {train_duration:.2f} seconds.")

    print("\n--- 5. Evaluating Model on Unseen Test Set ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}% (Prioritized to minimize False Negatives)")
    print(f"F1-Score  : {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    print(f"\n--- 6. Serializing Model ---")
    joblib.dump(model, model_output_path)
    print(f"Model successfully saved to: {model_output_path}")

    return model

if __name__ == "__main__":
    df = load_data()
    train_and_evaluate(df)
