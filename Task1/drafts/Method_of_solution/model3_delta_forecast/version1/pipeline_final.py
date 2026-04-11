"""
================================================================================
Model 3 v1 FINAL: Delta Forecast — Full IV for all horizons
================================================================================
y_hat(t+h) = y(t) + delta_hat(t, h)

Train on ALL data: 2019-03 ... 2025-09
Forecast: 2025-10 ... 2026-03 (6 months)

M1: yield-only features (~41)
M2: yield + full IV features for ALL horizons (~52)
================================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

from data_loading.problem_1 import get_curve_train_dataframe, get_IV_train_dataframe

# ==============================================================================
# CONSTANTS
# ==============================================================================
YIELD_TENORS = ["O/N", "1W", "2W", "1M", "2M", "3M", "6M", "1Y", "2Y"]
N_FORECAST = 6
IV_KEY_TENORS = ["1M", "3M", "1Y", "5Y", "10Y"]


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_data():
    yield_df = get_curve_train_dataframe()
    yield_df.index = yield_df.index.to_period('M').to_timestamp()
    iv_raw = get_IV_train_dataframe()
    iv_agg = (
        iv_raw
        .groupby([pd.Grouper(key='Date', freq='MS'), 'Maturity'])['Volatility']
        .mean()
        .unstack('Maturity')
    )
    IV_TENORS = ["1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y",
                 "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]
    available_iv_tenors = [t for t in IV_TENORS if t in iv_agg.columns]
    iv_df = iv_agg[available_iv_tenors].copy()
    iv_df = iv_df.ffill()
    common_dates = yield_df.index.intersection(iv_df.index)
    yield_df = yield_df.loc[common_dates]
    iv_df = iv_df.loc[common_dates]
    return yield_df, iv_df


# ==============================================================================
# DELTA TARGET PREPARATION
# ==============================================================================
def prepare_delta_targets(yield_df):
    delta_targets = {}
    for h in range(1, N_FORECAST + 1):
        shifted = yield_df[YIELD_TENORS].shift(-h)
        delta = shifted - yield_df[YIELD_TENORS]
        delta_targets[h] = delta.dropna()
    return delta_targets


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def build_m1_features(yield_df):
    df = yield_df[YIELD_TENORS].copy()
    features = pd.DataFrame(index=df.index)

    for t in YIELD_TENORS:
        features[f"cur_{t}"] = df[t]

    d1 = df.diff(1)
    for t in YIELD_TENORS:
        features[f"delta1_{t}"] = d1[t]

    d3 = df.diff(3)
    for t in YIELD_TENORS:
        features[f"delta3_{t}"] = d3[t]

    features["spread_2Y_ON"] = df["2Y"] - df["O/N"]
    features["spread_1Y_3M"] = df["1Y"] - df["3M"]
    features["spread_6M_1M"] = df["6M"] - df["1M"]

    features["d_spread_2Y_ON"] = features["spread_2Y_ON"].diff(1)
    features["d_spread_1Y_3M"] = features["spread_1Y_3M"].diff(1)

    ma6 = df.rolling(6).mean()
    for t in YIELD_TENORS:
        features[f"dev_ma6_{t}"] = df[t] - ma6[t]

    return features


def build_m2_features(yield_df, iv_df):
    """M2: M1 + full IV for ALL horizons (~52 cols)."""
    features = build_m1_features(yield_df)

    available_key = [t for t in IV_KEY_TENORS if t in iv_df.columns]

    for t in available_key:
        features[f"iv_lag1_{t}"] = iv_df[t].shift(1)

    iv_d1 = iv_df.diff(1).shift(1)
    for t in available_key:
        features[f"iv_delta1_{t}"] = iv_d1[t]

    if "10Y" in iv_df.columns and "1M" in iv_df.columns:
        features["iv_spread_10Y_1M"] = (iv_df["10Y"] - iv_df["1M"]).shift(1)

    return features


# ==============================================================================
# TRAINING & PREDICTION
# ==============================================================================
def train_and_predict(yield_df, features_df, label):
    """Train on all data, predict 6 months ahead."""
    delta_targets = prepare_delta_targets(yield_df)
    last_known = yield_df[YIELD_TENORS].iloc[-1]
    anchor_date = yield_df.index[-1]

    pred_dates = pd.date_range(
        start=anchor_date + pd.DateOffset(months=1),
        periods=N_FORECAST,
        freq='MS'
    )
    predictions = pd.DataFrame(index=pred_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, N_FORECAST + 1):
        dt = delta_targets[h]
        valid_idx = dt.index.intersection(features_df.dropna().index)

        X_train = features_df.loc[valid_idx].ffill(axis=1).fillna(0)
        hist_deltas = dt.loc[valid_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        x_anchor = features_df.loc[[anchor_date]].ffill(axis=1).fillna(0)
        x_anchor_scaled = scaler.transform(x_anchor)

        target_date = pred_dates[h - 1]
        print(f"    h={h} -> {target_date.strftime('%Y-%m')}: ", end="")

        for j, tenor in enumerate(YIELD_TENORS):
            y_train = dt.loc[valid_idx, tenor].values
            model = ElasticNet(alpha=0.1, l1_ratio=0.7, max_iter=10000)
            model.fit(X_scaled, y_train)

            delta_pred = model.predict(x_anchor_scaled)[0]
            max_delta = hist_deltas[tenor].abs().quantile(0.95)
            delta_pred = np.clip(delta_pred, -max_delta * h, max_delta * h)

            predictions.loc[target_date, tenor] = last_known[tenor] + delta_pred

        row = predictions.loc[target_date]
        print(f"O/N={row['O/N']:.4f}  2Y={row['2Y']:.4f}")

    return predictions


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_results(yield_df, pred_m1, pred_m2, output_dir):
    fig, axes = plt.subplots(len(YIELD_TENORS), 2, figsize=(18, 4 * len(YIELD_TENORS)))

    history = yield_df[YIELD_TENORS]
    anchor_date = history.index[-1]
    last_known = history.iloc[-1]

    for i, tenor in enumerate(YIELD_TENORS):
        for j, (pred, model_name) in enumerate([
            (pred_m1, 'M1'), (pred_m2, 'M2')
        ]):
            ax = axes[i, j]

            ax.plot(history.index, history[tenor], color='#FF69B4', linewidth=1.5,
                    label='History')

            pred_dates_ext = [anchor_date] + list(pred.index)
            pred_values_ext = [last_known[tenor]] + list(pred[tenor].values.astype(float))
            ax.plot(pred_dates_ext, pred_values_ext,
                    color='#2ECC71', linewidth=2, marker='s', markersize=4,
                    label=f'Forecast {model_name}')

            ax.set_ylabel(tenor, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)

            if i == 0:
                subtitle = 'M1 (yield only)' if model_name == 'M1' else \
                    'M2 (yield + full IV all horizons)'
                ax.set_title(subtitle, fontsize=12, fontweight='bold')

    plt.suptitle('Final Forecast v1: 2025-10 to 2026-03  |  Delta Forecast + Full IV',
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()

    path = os.path.join(output_dir, "forecast_all_tenors.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("  FINAL FORECAST v1: Delta + Full IV (train 2019-03..2025-09)")
    print("=" * 70)

    print("\n[1] Loading data...")
    yield_df, iv_df = load_data()
    print(f"    Yield: {yield_df.shape}")
    print(f"    IV:    {iv_df.shape}")
    print(f"    Last date: {yield_df.index[-1].strftime('%Y-%m')}")

    print("\n[2] Building features...")
    feat_m1 = build_m1_features(yield_df)
    feat_m2 = build_m2_features(yield_df, iv_df)
    print(f"    M1: {feat_m1.shape[1]} features")
    print(f"    M2: {feat_m2.shape[1]} features")

    print("\n[3] Training M1...")
    pred_m1 = train_and_predict(yield_df, feat_m1, "M1")

    print("\n[4] Training M2 (full IV all horizons)...")
    pred_m2 = train_and_predict(yield_df, feat_m2, "M2")

    # --- Print ---
    print("\n" + "=" * 70)
    print("  FORECAST M1")
    print("=" * 70)
    print(pred_m1.round(4).to_string())

    print("\n" + "=" * 70)
    print("  FORECAST M2")
    print("=" * 70)
    print(pred_m2.round(4).to_string())

    # --- Save ---
    output_dir = os.path.dirname(os.path.abspath(__file__))

    xlsx_path = os.path.join(output_dir, "Problem_1_yield_curve_predict.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pred_m1.to_excel(writer, sheet_name="M1", index_label="date")
        pred_m2.to_excel(writer, sheet_name="M2", index_label="date")
    print(f"\n  Predictions saved: {xlsx_path}")

    print("\n[5] Plotting...")
    path = plot_results(yield_df, pred_m1, pred_m2, output_dir)
    print(f"  Plot: {path}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
