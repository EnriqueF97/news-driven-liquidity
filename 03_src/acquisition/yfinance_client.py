"""
yfinance_client.py
------------------
Downloads WTI Crude Oil futures (CL=F) OHLCV data from yfinance.
Primary price and liquidity data source for the thesis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PRICE_DIR = Path("01_data/raw/price")


def download_wti_hourly(period: str = "2y") -> pd.DataFrame:
    """
    Download WTI hourly OHLCV data.
    
    Args:
        period: yfinance period string (e.g. "2y", "1y")
    
    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume]
    """
    df = yf.download("CL=F", period=period, interval="1h", progress=False)
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index.name = "Datetime"
    print(f"Downloaded {len(df)} hourly records")
    print(f"Coverage: {df.index.min()} to {df.index.max()}")
    return df


def build_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute liquidity proxies from OHLCV data.
    
    Variables:
        log_volume    : Log-transformed hourly volume (primary liquidity proxy)
        price_range   : High - Low (Parkinson 1980 volatility proxy)
        log_return    : Log return Close(t) / Close(t-1)
        amihud        : |log_return| / Volume (Amihud 2002 illiquidity ratio)
    """
    df = df.copy()
    df['log_volume'] = np.log(df['Volume'] + 1)
    df['price_range'] = df['High'] - df['Low']
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['abs_return'] = df['log_return'].abs()
    df['amihud'] = df['abs_return'] / (df['Volume'] + 1)
    return df


def save_raw(df: pd.DataFrame, filename: str = "wti_hourly_raw.csv"):
    RAW_PRICE_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_PRICE_DIR / filename
    df.to_csv(path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    df = download_wti_hourly(period="2y")
    df = build_liquidity_features(df)
    save_raw(df)
    print(df.tail())
