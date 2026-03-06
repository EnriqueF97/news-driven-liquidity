"""
eia_downloader.py
-----------------
Downloads U.S. commercial crude oil inventories from the EIA public API.
Used as structured news events (bearish/bullish classification).

EIA Series: WCRSTUS1 — U.S. Ending Stocks of Crude Oil (excl. SPR)
Published: Every Wednesday at 10:30 AM ET
"""

import requests
import pandas as pd
from pathlib import Path

RAW_MACRO_DIR = Path("01_data/raw/macro")
EIA_BASE_URL = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"


def download_eia_inventories(start: str = "2020-01-01") -> pd.DataFrame:
    """
    Download weekly U.S. crude oil commercial inventory data from EIA.

    Args:
        start: Start date string in YYYY-MM-DD format

    Returns:
        DataFrame with columns [date, inventory_mbbl, inventory_change, news_direction]
    """
    params = {
        "api_key": "DEMO_KEY",
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "WCRSTUS1",
        "start": start,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000
    }

    response = requests.get(EIA_BASE_URL, params=params)
    json_data = response.json()
    df = pd.DataFrame(json_data['response']['data'])

    # Clean
    df = df[['period', 'value']].copy()
    df.columns = ['date', 'inventory_mbbl']
    df['date'] = pd.to_datetime(df['date'])
    df['inventory_mbbl'] = pd.to_numeric(df['inventory_mbbl'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    # Compute weekly change and direction
    df['inventory_change'] = df['inventory_mbbl'].diff()
    df['news_direction'] = df['inventory_change'].apply(
        lambda x: 'bearish' if x > 0 else ('bullish' if x < 0 else 'neutral')
    )

    # EIA publishes every Wednesday at 10:30 AM ET
    df['datetime_et'] = pd.to_datetime(df['date']) + pd.Timedelta(hours=10, minutes=30)
    df['datetime_et'] = df['datetime_et'].dt.tz_localize('US/Eastern')

    print(f"Downloaded {len(df)} weekly EIA records")
    print(f"Coverage: {df['date'].min()} to {df['date'].max()}")
    print(f"Bearish events: {(df['news_direction']=='bearish').sum()}")
    print(f"Bullish events: {(df['news_direction']=='bullish').sum()}")
    return df


def save_raw(df: pd.DataFrame, filename: str = "eia_inventories_raw.csv"):
    RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_MACRO_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


if __name__ == "__main__":
    df = download_eia_inventories(start="2020-01-01")
    save_raw(df)
    print(df.tail(10))
