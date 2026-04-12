"""
================================================================================
Model 3 v3 FINAL: Ridge on PCA-components of deltas (FORECAST)
================================================================================
Train on ALL data: 2019-03 ... 2025-09
Forecast: 2025-10 ... 2026-03

3 PCA components x 6 horizons = 18 Ridge models (instead of 54).
Consistent curve shape guaranteed by PCA reconstruction.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

from data_loading.problem_1 import get_curve_train_dataframe, get_IV_train_dataframe

# ==============================================================================
# CONSTANTS
# ==============================================================================
YIELD_TENORS = ["O/N", "1W", "2W", "1M", "2M", "3M", "6M", "1Y", "2Y"]
N_FORECAST = 6
N_COMPONENTS = 3

IV_KEY_TENORS = ["1M", "3M", "1Y", "5Y", "10Y"]
IV_SHORT_HORIZON_MAX = 3


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
    available = [t for t in IV_TENORS if t in iv_agg.columns]
    iv_df = iv_agg[available].copy().ffill()
    common = yield_df.index.intersection(iv_df.index)
    return yield_df.loc[common], iv_df.loc[common]


# ==============================================================================
# DELTA TARGETS + PCA
# ==============================================================================
def prepare_delta_targets(yield_df):
    delta_targets = {}
    for h in range(1, N_FORECAST + 1):
        shifted = yield_df[YIELD_TENORS].shift(-h)
        delta = shifted - yield_df[YIELD_TENORS]
        delta_targets[h] = delta.dropna()
    return delta_targets


def fit_delta_pca(delta_targets):
    """Fit PCA on ALL deltas (all horizons pooled)."""
    all_deltas = []
    for h in range(1, N_FORECAST + 1):
        all_deltas.append(delta_targets[h].values)
    pooled = np.vstack(all_deltas)
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(pooled)
    return pca


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


def build_iv_features_full(iv_df):
    features = pd.DataFrame(index=iv_df.index)
    available = [t for t in IV_KEY_TENORS if t in iv_df.columns]
    for t in available:
        features[f"iv_lag1_{t}"] = iv_df[t].shift(1)
    iv_d1 = iv_df.diff(1).shift(1)
    for t in available:
        features[f"iv_delta1_{t}"] = iv_d1[t]
    if "10Y" in iv_df.columns and "1M" in iv_df.columns:
        features["iv_spread_10Y_1M"] = (iv_df["10Y"] - iv_df["1M"]).shift(1)
    return features


def build_iv_features_minimal(iv_df):
    features = pd.DataFrame(index=iv_df.index)
    if "10Y" in iv_df.columns and "1M" in iv_df.columns:
        features["iv_spread_10Y_1M"] = (iv_df["10Y"] - iv_df["1M"]).shift(1)
    return features


# ==============================================================================
# TRAINING & PREDICTION
# ==============================================================================
def train_and_predict(yield_df, features_df, delta_targets, pca, label):
    last_known = yield_df[YIELD_TENORS].iloc[-1]
    anchor_date = yield_df.index[-1]
    pred_dates = pd.date_range(start=anchor_date + pd.DateOffset(months=1),
                               periods=N_FORECAST, freq='MS')
    predictions = pd.DataFrame(index=pred_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, N_FORECAST + 1):
        dt = delta_targets[h]
        valid_idx = dt.index.intersection(features_df.dropna().index)
        X_train = features_df.loc[valid_idx].ffill(axis=1).fillna(0)
        delta_pca = pca.transform(dt.loc[valid_idx].values)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        x_anchor = features_df.loc[[anchor_date]].ffill(axis=1).fillna(0)
        x_anchor_scaled = scaler.transform(x_anchor)

        pred_comp = np.zeros(N_COMPONENTS)
        for k in range(N_COMPONENTS):
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, delta_pca[:, k])
            pred_comp[k] = model.predict(x_anchor_scaled)[0]

        for k in range(N_COMPONENTS):
            lo, hi = delta_pca[:, k].min(), delta_pca[:, k].max()
            margin = (hi - lo) * 0.1
            pred_comp[k] = np.clip(pred_comp[k], lo - margin, hi + margin)

        delta_9 = pca.inverse_transform(pred_comp.reshape(1, -1))[0]
        target_date = pred_dates[h - 1]
        for j, tenor in enumerate(YIELD_TENORS):
            predictions.loc[target_date, tenor] = last_known[tenor] + delta_9[j]

        row = predictions.loc[target_date]
        print(f"    h={h} -> {target_date.strftime('%Y-%m')}: "
              f"O/N={row['O/N']:.4f}  2Y={row['2Y']:.4f}  "
              f"(PC: {pred_comp[0]:+.2f}, {pred_comp[1]:+.2f}, {pred_comp[2]:+.2f})")

    return predictions


def train_and_predict_m2(yield_df, iv_df, delta_targets, pca):
    last_known = yield_df[YIELD_TENORS].iloc[-1]
    anchor_date = yield_df.index[-1]

    feat_m1 = build_m1_features(yield_df)
    iv_full = build_iv_features_full(iv_df)
    iv_minimal = build_iv_features_minimal(iv_df)
    feat_full = pd.concat([feat_m1, iv_full], axis=1)
    feat_min = pd.concat([feat_m1, iv_minimal], axis=1)

    print(f"    h<=3: {feat_full.shape[1]} cols, h>=4: {feat_min.shape[1]} cols")

    pred_dates = pd.date_range(start=anchor_date + pd.DateOffset(months=1),
                               periods=N_FORECAST, freq='MS')
    predictions = pd.DataFrame(index=pred_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, N_FORECAST + 1):
        features_df = feat_full if h <= IV_SHORT_HORIZON_MAX else feat_min
        iv_label = "full IV" if h <= IV_SHORT_HORIZON_MAX else "min IV"

        dt = delta_targets[h]
        valid_idx = dt.index.intersection(features_df.dropna().index)
        X_train = features_df.loc[valid_idx].ffill(axis=1).fillna(0)
        delta_pca = pca.transform(dt.loc[valid_idx].values)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        x_anchor = features_df.loc[[anchor_date]].ffill(axis=1).fillna(0)
        x_anchor_scaled = scaler.transform(x_anchor)

        pred_comp = np.zeros(N_COMPONENTS)
        for k in range(N_COMPONENTS):
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, delta_pca[:, k])
            pred_comp[k] = model.predict(x_anchor_scaled)[0]

        for k in range(N_COMPONENTS):
            lo, hi = delta_pca[:, k].min(), delta_pca[:, k].max()
            margin = (hi - lo) * 0.1
            pred_comp[k] = np.clip(pred_comp[k], lo - margin, hi + margin)

        delta_9 = pca.inverse_transform(pred_comp.reshape(1, -1))[0]
        target_date = pred_dates[h - 1]
        for j, tenor in enumerate(YIELD_TENORS):
            predictions.loc[target_date, tenor] = last_known[tenor] + delta_9[j]

        row = predictions.loc[target_date]
        print(f"    h={h} [{iv_label:>8}] -> {target_date.strftime('%Y-%m')}: "
              f"O/N={row['O/N']:.4f}  2Y={row['2Y']:.4f}")

    return predictions


# ==============================================================================
# VISUALIZATION (3 columns)
# ==============================================================================
def plot_results(yield_df, pred_m1, pred_m2, pred_avg, output_dir):
    fig, axes = plt.subplots(len(YIELD_TENORS), 3, figsize=(26, 4 * len(YIELD_TENORS)))
    history = yield_df[YIELD_TENORS]
    anchor_date = history.index[-1]
    last_known = history.iloc[-1]

    for i, tenor in enumerate(YIELD_TENORS):
        for j, (pred, name, color) in enumerate([
            (pred_m1, 'M1', '#2ECC71'),
            (pred_m2, 'M2', '#2ECC71'),
            (pred_avg, 'AVG', '#3498DB'),
        ]):
            ax = axes[i, j]
            ax.plot(history.index, history[tenor], color='#FF69B4', linewidth=1.5, label='History')
            ext_d = [anchor_date] + list(pred.index)
            ext_v = [last_known[tenor]] + list(pred[tenor].values.astype(float))
            ax.plot(ext_d, ext_v, color=color, linewidth=2, marker='s', markersize=4,
                    label=f'Forecast {name}')
            ax.set_ylabel(tenor, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            if i == 0:
                titles = ['M1 (PCA-delta Ridge)', 'M2 (PCA-delta + adaptive IV)',
                          'Average (M1+M2)/2']
                ax.set_title(titles[j], fontsize=12, fontweight='bold')

    plt.suptitle('Final Forecast v3: PCA-Delta Ridge  |  2025-10 to 2026-03',
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
    print("  FINAL FORECAST v3: Ridge on PCA-delta components")
    print("=" * 70)

    print("\n[1] Loading data...")
    yield_df, iv_df = load_data()
    print(f"    Yield: {yield_df.shape}, IV: {iv_df.shape}")
    print(f"    Last date: {yield_df.index[-1].strftime('%Y-%m')}")

    print("\n[2] Delta targets + PCA...")
    delta_targets = prepare_delta_targets(yield_df)
    pca = fit_delta_pca(delta_targets)
    print(f"    Explained variance: {pca.explained_variance_ratio_.round(4)}")
    print(f"    Cumulative: {pca.explained_variance_ratio_.cumsum().round(4)}")

    print("\n[3] Training M1...")
    feat_m1 = build_m1_features(yield_df)
    pred_m1 = train_and_predict(yield_df, feat_m1, delta_targets, pca, "M1")

    print("\n[4] Training M2...")
    pred_m2 = train_and_predict_m2(yield_df, iv_df, delta_targets, pca)

    pred_avg = (pred_m1.astype(float) + pred_m2.astype(float)) / 2

    print("\n" + "=" * 70)
    print("  FORECAST M1")
    print("=" * 70)
    print(pred_m1.round(4).to_string())
    print("\n" + "=" * 70)
    print("  FORECAST M2")
    print("=" * 70)
    print(pred_m2.round(4).to_string())
    print("\n" + "=" * 70)
    print("  FORECAST AVG (M1+M2)/2")
    print("=" * 70)
    print(pred_avg.round(4).to_string())

    output_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(output_dir, "Problem_1_yield_curve_predict.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pred_m1.to_excel(writer, sheet_name="M1", index_label="date")
        pred_m2.to_excel(writer, sheet_name="M2", index_label="date")
    print(f"\n  Saved: {xlsx_path}")

    print("\n[5] Plotting...")
    path = plot_results(yield_df, pred_m1, pred_m2, pred_avg, output_dir)
    print(f"  Plot: {path}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
