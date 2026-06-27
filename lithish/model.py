import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("imbalanced-learn not installed, will skip SMOTE if it fails")

DATA_FILE = "creditcard.csv"

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Please run download_data.py first.")
    df = pd.read_csv(DATA_FILE)
    return df

def train_and_evaluate():
    print("Loading data...")
    df = load_data()
    
    # Check for NaN values
    df = df.dropna()
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"Original dataset shape: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE to handle imbalance on training data
    try:
        print("Applying SMOTE to handle imbalance...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"Resampled dataset shape: {pd.Series(y_train_res).value_counts().to_dict()}")
    except NameError:
        print("SMOTE is not available, training on original imbalanced data.")
        X_train_res, y_train_res = X_train, y_train
        
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    
    print("Evaluating model...")
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix plot to confusion_matrix.png")
    
    # Precision-Recall Curve (Good for imbalanced datasets)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker='.', label=f'Random Forest (PR AUC = {pr_auc:.2f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('precision_recall_curve.png')
    print("Saved precision-recall curve to precision_recall_curve.png")

if __name__ == "__main__":
    train_and_evaluate()
