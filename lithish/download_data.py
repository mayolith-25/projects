import urllib.request
import os
import pandas as pd

DATA_FILE = "creditcard.csv"
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

def download_data():
    if os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} already exists. Skipping download.")
        return pd.read_csv(DATA_FILE)
    
    print(f"Downloading data from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print(f"Data saved to {DATA_FILE}")
    
    return pd.read_csv(DATA_FILE)

if __name__ == "__main__":
    df = download_data()
    print(df.head())
    print(f"Class distribution:\n{df['Class'].value_counts()}")
