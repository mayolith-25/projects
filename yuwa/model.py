import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

DATA_FILE = "gold_price_data.csv"

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Please run download_data.py first.")
    df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    return df

def preprocess_data(df):
    # Drop rows with NaN values (e.g., weekends/holidays if any)
    df = df.dropna()
    
    # We will use moving averages and previous day's close to predict today's close
    df['Prev_Close'] = df['Close'].shift(1)
    df['MA_5'] = df['Close'].rolling(window=5).mean().shift(1)
    df['MA_10'] = df['Close'].rolling(window=10).mean().shift(1)
    
    # Drop rows with NaN values introduced by shifting and rolling
    df = df.dropna()
    
    features = ['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'MA_5', 'MA_10']
    target = 'Close'
    
    X = df[features]
    y = df[target]
    
    return X, y

def train_and_evaluate():
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Split data (using chronological split for time series)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Plotting feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig('feature_importance.png')
    print("Saved feature importance plot to feature_importance.png")

if __name__ == "__main__":
    train_and_evaluate()
