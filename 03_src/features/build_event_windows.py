"""
build_event_windows.py
-----------------------
Constructs event windows around EIA report publication times.
Each window captures hourly liquidity metrics in the ±4 hour range
around each weekly EIA inventory report.
"""

import pandas as pd
import numpy as np
from pathlib import Path

FEATURES_DIR = Path("01_data/features")


def build_event_windows(
    df_price: pd.DataFrame,
    df_eia: pd.DataFrame,
    hours_before: int = 4,
    hours_after: int = 8,
    min_events_per_hour: int = 20
) -> pd.DataFrame:
    """
    Build event study windows around EIA report times.

    Args:
        df_price: Hourly OHLCV + liquidity features (from yfinance_client)
        df_eia: EIA inventory events (from eia_downloader)
        hours_before: Hours before event to include
        hours_after: Hours after event to include
        min_events_per_hour: Minimum events per relative hour (filters artifacts)

    Returns:
        DataFrame with liquidity features, event metadata, and hours_from_event
    """
    # Align timezones
    df_price.index = pd.to_datetime(df_price.index).tz_convert('US/Eastern')

    windows = []
    for _, event in df_eia.iterrows():
        t = event['datetime_et']
        window = df_price.loc[
            t - pd.Timedelta(hours=hours_before):
            t + pd.Timedelta(hours=hours_after)
        ].copy()

        if len(window) == 0:
            continue

        window['event_date'] = event['date']
        window['inventory_change'] = event['inventory_change']
        window['shock_magnitude'] = abs(event['inventory_change'])
        window['news_direction'] = event['news_direction']
        window['hours_from_event'] = (
            (window.index - t).total_seconds() / 3600
        ).round(1)
        windows.append(window)

    df_events = pd.concat(windows)

    # Filter hours with too few observations (market close artifacts)
    counts = df_events.groupby('hours_from_event')['log_volume'].count()
    valid_hours = counts[counts >= min_events_per_hour].index
    df_events = df_events[df_events['hours_from_event'].isin(valid_hours)]

    print(f"Event windows built: {df_events['event_date'].nunique()} events")
    print(f"Total records: {len(df_events)}")
    print(f"Valid hour range: {df_events['hours_from_event'].min()} to {df_events['hours_from_event'].max()}")
    return df_events


def save_features(df: pd.DataFrame, filename: str = "event_windows.csv"):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / filename
    df.to_csv(path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    from adquisicion.yfinance_client import download_wti_hourly, build_liquidity_features
    from adquisicion.eia_downloader import download_eia_inventories

    df_price = build_liquidity_features(download_wti_hourly())
    df_eia = download_eia_inventories()
    df_events = build_event_windows(df_price, df_eia)
    save_features(df_events)
