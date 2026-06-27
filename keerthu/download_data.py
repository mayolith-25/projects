import urllib.request
import os
import pandas as pd

DATA_FILE = "employee_promotion.csv"
DATA_URL = "https://raw.githubusercontent.com/rajtulluri/Employee-Promotion-Prediction/master/employeePromotion.csv"

def download_data():
    if os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} already exists. Skipping download.")
        return pd.read_csv(DATA_FILE)
    
    print(f"Downloading data from {DATA_URL}...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print(f"Data saved to {DATA_FILE}")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
    
    return pd.read_csv(DATA_FILE)

if __name__ == "__main__":
    df = download_data()
    if df is not None:
        print(df.head())
        print(f"Target distribution:\n{df['is_promoted'].value_counts()}")
