"""
================================================================================
Model 3 v2+3 FINAL: Ensemble ElasticNet + PCA-Delta Ridge (FORECAST)
================================================================================
Train on ALL data: 2019-03 ... 2025-09
Forecast: 2025-10 ... 2026-03

M1: pure v3 (PCA-delta Ridge)
M2: 70% v2 (ElasticNet) + 30% v3 (PCA-Ridge)
Weights from validation on 2025-04..09
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
from sklearn.linear_model import ElasticNet, Ridge
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

# Ensemble weights (from validation)
W_V2_M1 = 0.0   # M1: pure v3
W_V2_M2 = 0.7   # M2: 70% v2, 30% v3


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


def get_features(feat_m1, iv_full, iv_min, h, model_type):
    if model_type == "M1":
        return feat_m1
    else:
        if h <= IV_SHORT_HORIZON_MAX:
            return pd.concat([feat_m1, iv_full], axis=1)
        else:
            return pd.concat([feat_m1, iv_min], axis=1)


# ==============================================================================
# DELTA TARGETS
# ==============================================================================
def prepare_delta_targets(yield_df):
    delta_targets = {}
    for h in range(1, N_FORECAST + 1):
        shifted = yield_df[YIELD_TENORS].shift(-h)
        delta = shifted - yield_df[YIELD_TENORS]
        delta_targets[h] = delta.dropna()
    return delta_targets


# ==============================================================================
# V2: ElasticNet per-tenor (forecast mode)
# ==============================================================================
def predict_v2(yield_df, delta_targets, feat_m1, iv_full, iv_min, model_type):
    last_known = yield_df[YIELD_TENORS].iloc[-1]
    anchor_date = yield_df.index[-1]
    pred_dates = pd.date_range(start=anchor_date + pd.DateOffset(months=1),
                               periods=N_FORECAST, freq='MS')
    predictions = pd.DataFrame(index=pred_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, N_FORECAST + 1):
        features_df = get_features(feat_m1, iv_full, iv_min, h, model_type)
        dt = delta_targets[h]
        valid_idx = dt.index.intersection(features_df.dropna().index)

        X_train = features_df.loc[valid_idx].ffill(axis=1).fillna(0)
        hist_deltas = dt.loc[valid_idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        x_anchor = features_df.loc[[anchor_date]].ffill(axis=1).fillna(0)
        x_anchor_scaled = scaler.transform(x_anchor)

        target_date = pred_dates[h - 1]
        for j, tenor in enumerate(YIELD_TENORS):
            y_train = dt.loc[valid_idx, tenor].values
            model = ElasticNet(alpha=0.1, l1_ratio=0.7, max_iter=10000)
            model.fit(X_scaled, y_train)
            dp = model.predict(x_anchor_scaled)[0]
            md = hist_deltas[tenor].abs().quantile(0.95)
            dp = np.clip(dp, -md * h, md * h)
            predictions.loc[target_date, tenor] = last_known[tenor] + dp

    return predictions


# ==============================================================================
# V3: PCA-delta Ridge (forecast mode)
# ==============================================================================
def predict_v3(yield_df, delta_targets, pca, feat_m1, iv_full, iv_min, model_type):
    last_known = yield_df[YIELD_TENORS].iloc[-1]
    anchor_date = yield_df.index[-1]
    pred_dates = pd.date_range(start=anchor_date + pd.DateOffset(months=1),
                               periods=N_FORECAST, freq='MS')
    predictions = pd.DataFrame(index=pred_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, N_FORECAST + 1):
        features_df = get_features(feat_m1, iv_full, iv_min, h, model_type)
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
            lo, hi = delta_pca[:, k].min(), delta_pca[:, k].max()
            margin = (hi - lo) * 0.1
            pred_comp[k] = np.clip(pred_comp[k], lo - margin, hi + margin)

        delta_9 = pca.inverse_transform(pred_comp.reshape(1, -1))[0]
        target_date = pred_dates[h - 1]
        for j, tenor in enumerate(YIELD_TENORS):
            predictions.loc[target_date, tenor] = last_known[tenor] + delta_9[j]

    return predictions


# ==============================================================================
# VISUALIZATION (3 columns: M1, M2, AVG)
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
            ax.plot(history.index, history[tenor], color='#FF69B4', linewidth=1.5,
                    label='History')
            ext_d = [anchor_date] + list(pred.index)
            ext_v = [last_known[tenor]] + list(pred[tenor].values.astype(float))
            ax.plot(ext_d, ext_v, color=color, linewidth=2, marker='s', markersize=4,
                    label=f'Forecast {name}')
            ax.set_ylabel(tenor, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            if i == 0:
                titles = ['M1 (PCA-delta Ridge)',
                          f'M2 ({W_V2_M2:.0%} ElasticNet + {1-W_V2_M2:.0%} PCA-Ridge)',
                          'Average (M1+M2)/2']
                ax.set_title(titles[j], fontsize=12, fontweight='bold')

    plt.suptitle('Final Forecast v2+3: Ensemble  |  2025-10 to 2026-03',
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
    print("  FINAL FORECAST v2+3: Ensemble")
    print(f"  M1: w_v2={W_V2_M1:.0%} v2 + {1-W_V2_M1:.0%} v3")
    print(f"  M2: w_v2={W_V2_M2:.0%} v2 + {1-W_V2_M2:.0%} v3")
    print("=" * 70)

    print("\n[1] Loading data...")
    yield_df, iv_df = load_data()
    print(f"    Yield: {yield_df.shape}, IV: {iv_df.shape}")

    print("\n[2] Preparing features...")
    feat_m1 = build_m1_features(yield_df)
    iv_full = build_iv_features_full(iv_df)
    iv_min = build_iv_features_minimal(iv_df)
    delta_targets = prepare_delta_targets(yield_df)
    pca = PCA(n_components=N_COMPONENTS)
    all_d = [delta_targets[h].values for h in range(1, N_FORECAST + 1)]
    pca.fit(np.vstack(all_d))
    print(f"    PCA variance: {pca.explained_variance_ratio_.round(4)}")

    # --- M1: pure v3 ---
    print("\n[3] M1 forecast (pure v3)...")
    v3_m1 = predict_v3(yield_df, delta_targets, pca, feat_m1, iv_full, iv_min, "M1")
    pred_m1 = v3_m1  # w_v2=0

    for h in range(N_FORECAST):
        row = pred_m1.iloc[h]
        print(f"    {pred_m1.index[h].strftime('%Y-%m')}: O/N={row['O/N']:.4f}  2Y={row['2Y']:.4f}")

    # --- M2: 70% v2 + 30% v3 ---
    print(f"\n[4] M2 forecast ({W_V2_M2:.0%} v2 + {1-W_V2_M2:.0%} v3)...")
    v2_m2 = predict_v2(yield_df, delta_targets, feat_m1, iv_full, iv_min, "M2")
    v3_m2 = predict_v3(yield_df, delta_targets, pca, feat_m1, iv_full, iv_min, "M2")
    pred_m2 = pd.DataFrame(
        W_V2_M2 * v2_m2.values.astype(float) + (1 - W_V2_M2) * v3_m2.values.astype(float),
        index=v2_m2.index, columns=YIELD_TENORS
    )

    for h in range(N_FORECAST):
        row = pred_m2.iloc[h]
        print(f"    {pred_m2.index[h].strftime('%Y-%m')}: O/N={row['O/N']:.4f}  2Y={row['2Y']:.4f}")

    pred_avg = (pred_m1.astype(float) + pred_m2.astype(float)) / 2

    # --- Print ---
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

    # --- Save ---
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
