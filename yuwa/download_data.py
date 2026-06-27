import yfinance as yf
import pandas as pd
import os

DATA_FILE = "gold_price_data.csv"

def download_data():
    if os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} already exists. Skipping download.")
        return pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    
    print("Downloading historical gold prices...")
    # Ticker for Gold Futures
    ticker = "GC=F"
    gold_data = yf.download(ticker, start="2015-01-01", end="2023-12-31")
    
    if gold_data.empty:
        print("Failed to download data.")
        return None

    # Newer yfinance returns MultiIndex columns (Price, Ticker) — flatten them
    if isinstance(gold_data.columns, pd.MultiIndex):
        gold_data.columns = gold_data.columns.droplevel('Ticker')

    gold_data.to_csv(DATA_FILE)
    print(f"Data saved to {DATA_FILE}")
    return gold_data

if __name__ == "__main__":
    df = download_data()
    if df is not None:
        print(df.head())
