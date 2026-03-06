"""
ols_baseline.py
---------------
OLS regression baseline for RQ1 (asymmetry analysis).
Tests whether bearish EIA reports generate significantly more
trading volume than bullish reports.

Baseline result (Month 1):
  is_bearish coef = 0.210, p = 0.030
  ~23% more volume in bearish vs bullish events (hours 0-2)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path


def run_ols_baseline(df_events: pd.DataFrame, reaction_window: tuple = (0, 2)):
    """
    Run OLS asymmetry analysis on event window data.

    Args:
        df_events: Event windows DataFrame (from build_event_windows)
        reaction_window: Tuple (start_hour, end_hour) for reaction window

    Returns:
        Dictionary with model results
    """
    # Filter to reaction window
    h_start, h_end = reaction_window
    reg_data = df_events[
        df_events['hours_from_event'].between(h_start, h_end)
    ].copy()

    # Prepare variables
    reg_data['is_bearish'] = (reg_data['news_direction'] == 'bearish').astype(int)
    reg_data['shock_size_norm'] = (
        (reg_data['shock_magnitude'] - reg_data['shock_magnitude'].mean()) /
        reg_data['shock_magnitude'].std()
    )

    # Three model specifications
    m1 = smf.ols('log_volume ~ is_bearish', data=reg_data).fit()
    m2 = smf.ols('log_volume ~ is_bearish + shock_size_norm', data=reg_data).fit()
    m3 = smf.ols('log_volume ~ is_bearish * shock_size_norm', data=reg_data).fit()

    print("=" * 50)
    print("OLS BASELINE — ASYMMETRY ANALYSIS")
    print(f"Reaction window: hours {h_start} to {h_end}")
    print(f"N observations: {len(reg_data)}")
    print("=" * 50)

    for name, model in [("M1 — Direction only", m1),
                        ("M2 — Direction + Magnitude", m2),
                        ("M3 — With interaction", m3)]:
        print(f"\n{name}")
        print(f"  is_bearish: coef={model.params['is_bearish']:.3f}, "
              f"p={model.pvalues['is_bearish']:.4f} "
              f"{'✅' if model.pvalues['is_bearish'] < 0.05 else '❌'}")
        print(f"  R² = {model.rsquared:.4f}")

    return {"m1": m1, "m2": m2, "m3": m3, "data": reg_data}


if __name__ == "__main__":
    df_events = pd.read_csv("01_data/features/event_windows.csv",
                            index_col=0, parse_dates=True)
    results = run_ols_baseline(df_events)
