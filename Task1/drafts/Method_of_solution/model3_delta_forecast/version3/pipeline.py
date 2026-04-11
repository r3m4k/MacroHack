"""
================================================================================
Model 3 v3: Ridge on PCA-components of deltas (VALIDATION)
================================================================================
Instead of 9 separate delta models per horizon, decompose deltas via PCA
into 3 components (delta_level, delta_slope, delta_curvature).
Predict 3 components, reconstruct 9 tenors via inverse PCA.

18 models total (3 components x 6 horizons) instead of 54.
Guarantees consistent curve shape.

Validation: train to 2025-03, validate on 2025-04..2025-09
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
TRAIN_END = "2025-03"
VAL_START = "2025-04"
VAL_END = "2025-09"
N_FORECAST = 6
N_COMPONENTS = 3
WEIGHT_ON = 0.4
WEIGHT_OTHER = 0.6 / 8

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


def fit_delta_pca(delta_targets, train_end):
    """
    Fit PCA on training deltas (all horizons pooled).
    Returns fitted PCA object.
    """
    all_deltas = []
    for h in range(1, N_FORECAST + 1):
        dt = delta_targets[h]
        train_dt = dt.loc[dt.index <= train_end]
        all_deltas.append(train_dt.values)

    pooled = np.vstack(all_deltas)  # (N*6, 9)
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
# METRIC
# ==============================================================================
def weighted_rmse(y_true, y_pred):
    T = y_true.shape[0]
    weights = np.array([0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075])
    return np.sqrt(np.sum(weights[np.newaxis, :] * (y_true - y_pred) ** 2) / T)


# ==============================================================================
# TRAINING & PREDICTION
# ==============================================================================
def train_and_predict(yield_df, features_df, delta_targets, pca, label,
                      train_end=TRAIN_END, val_start=VAL_START, val_end=VAL_END):
    """
    For each horizon h:
      1. Transform 9-tenor deltas into 3 PCA components
      2. Train 3 Ridge models (one per component)
      3. Predict 3 components at anchor point
      4. Inverse transform back to 9 tenors
      5. Add to last known values
    """
    train_dates = yield_df.index[yield_df.index <= train_end]
    last_known = yield_df.loc[train_dates[-1], YIELD_TENORS]
    val_dates = yield_df.index[(yield_df.index >= val_start) & (yield_df.index <= val_end)]

    predictions = pd.DataFrame(index=val_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, len(val_dates) + 1):
        dt = delta_targets[h]
        valid_idx = dt.index.intersection(train_dates).intersection(features_df.dropna().index)

        X_train = features_df.loc[valid_idx].ffill(axis=1).fillna(0)
        delta_train = dt.loc[valid_idx].values  # (N, 9)

        # Transform deltas to PCA space
        delta_pca = pca.transform(delta_train)  # (N, 3)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        x_anchor = features_df.loc[[train_dates[-1]]].ffill(axis=1).fillna(0)
        x_anchor_scaled = scaler.transform(x_anchor)

        # Predict each PCA component
        pred_components = np.zeros(N_COMPONENTS)
        for k in range(N_COMPONENTS):
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, delta_pca[:, k])
            pred_components[k] = model.predict(x_anchor_scaled)[0]

        # Clip components based on training range
        for k in range(N_COMPONENTS):
            comp_min = delta_pca[:, k].min()
            comp_max = delta_pca[:, k].max()
            margin = (comp_max - comp_min) * 0.1
            pred_components[k] = np.clip(pred_components[k],
                                         comp_min - margin, comp_max + margin)

        # Inverse PCA: 3 components -> 9 tenor deltas
        delta_pred_9 = pca.inverse_transform(pred_components.reshape(1, -1))[0]  # (9,)

        target_date = val_dates[h - 1]
        for j, tenor in enumerate(YIELD_TENORS):
            predictions.loc[target_date, tenor] = last_known[tenor] + delta_pred_9[j]

        row = predictions.loc[target_date]
        print(f"    h={h} -> {target_date.strftime('%Y-%m')}: "
              f"O/N={row['O/N']:.4f}  2Y={row['2Y']:.4f}  "
              f"(PC: {pred_components[0]:+.2f}, {pred_components[1]:+.2f}, {pred_components[2]:+.2f})")

    return predictions


def train_and_predict_m2(yield_df, iv_df, delta_targets, pca):
    """M2: horizon-adaptive IV features."""
    train_dates = yield_df.index[yield_df.index <= TRAIN_END]
    last_known = yield_df.loc[train_dates[-1], YIELD_TENORS]
    val_dates = yield_df.index[(yield_df.index >= VAL_START) & (yield_df.index <= VAL_END)]

    feat_m1 = build_m1_features(yield_df)
    iv_full = build_iv_features_full(iv_df)
    iv_minimal = build_iv_features_minimal(iv_df)
    feat_full = pd.concat([feat_m1, iv_full], axis=1)
    feat_min = pd.concat([feat_m1, iv_minimal], axis=1)

    print(f"    h<=3: {feat_full.shape[1]} cols, h>=4: {feat_min.shape[1]} cols")

    predictions = pd.DataFrame(index=val_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, len(val_dates) + 1):
        features_df = feat_full if h <= IV_SHORT_HORIZON_MAX else feat_min
        iv_label = "full IV" if h <= IV_SHORT_HORIZON_MAX else "min IV"

        dt = delta_targets[h]
        valid_idx = dt.index.intersection(train_dates).intersection(features_df.dropna().index)

        X_train = features_df.loc[valid_idx].ffill(axis=1).fillna(0)
        delta_train = dt.loc[valid_idx].values

        delta_pca = pca.transform(delta_train)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        x_anchor = features_df.loc[[train_dates[-1]]].ffill(axis=1).fillna(0)
        x_anchor_scaled = scaler.transform(x_anchor)

        pred_components = np.zeros(N_COMPONENTS)
        for k in range(N_COMPONENTS):
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, delta_pca[:, k])
            pred_components[k] = model.predict(x_anchor_scaled)[0]

        for k in range(N_COMPONENTS):
            comp_min = delta_pca[:, k].min()
            comp_max = delta_pca[:, k].max()
            margin = (comp_max - comp_min) * 0.1
            pred_components[k] = np.clip(pred_components[k],
                                         comp_min - margin, comp_max + margin)

        delta_pred_9 = pca.inverse_transform(pred_components.reshape(1, -1))[0]

        target_date = val_dates[h - 1]
        for j, tenor in enumerate(YIELD_TENORS):
            predictions.loc[target_date, tenor] = last_known[tenor] + delta_pred_9[j]

        row = predictions.loc[target_date]
        print(f"    h={h} [{iv_label:>8}] -> {target_date.strftime('%Y-%m')}: "
              f"O/N={row['O/N']:.4f}  2Y={row['2Y']:.4f}")

    return predictions


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_results(yield_df, pred_m1, pred_m2, output_dir):
    fig, axes = plt.subplots(len(YIELD_TENORS), 2, figsize=(18, 4 * len(YIELD_TENORS)))
    history = yield_df[YIELD_TENORS]
    anchor_date = yield_df.loc[yield_df.index <= TRAIN_END].index[-1]
    last_known = history.loc[anchor_date]

    for i, tenor in enumerate(YIELD_TENORS):
        for j, (pred, name) in enumerate([(pred_m1, 'M1'), (pred_m2, 'M2')]):
            ax = axes[i, j]
            ax.plot(history.index, history[tenor], color='#FF69B4', linewidth=1.5, label='Actual')
            ext_dates = [anchor_date] + list(pred.index)
            ext_vals = [last_known[tenor]] + list(pred[tenor].values.astype(float))
            ax.plot(ext_dates, ext_vals, color='#2ECC71', linewidth=2,
                    marker='s', markersize=4, label=f'Forecast {name}')
            ax.set_ylabel(tenor, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            if i == 0:
                ax.set_title(f'{name} (PCA-delta Ridge)', fontsize=12, fontweight='bold')

    plt.suptitle('v3: Ridge on PCA-components of deltas (validation)',
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    path = os.path.join(output_dir, "validation_all_tenors.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("  v3 VALIDATION: Ridge on PCA-components of deltas")
    print(f"  {N_COMPONENTS} components, {N_COMPONENTS * N_FORECAST} models total")
    print("=" * 70)

    print("\n[1] Loading data...")
    yield_df, iv_df = load_data()
    print(f"    Yield: {yield_df.shape}, IV: {iv_df.shape}")

    print("\n[2] Preparing delta targets + PCA...")
    delta_targets = prepare_delta_targets(yield_df)
    pca = fit_delta_pca(delta_targets, TRAIN_END)
    print(f"    Explained variance: {pca.explained_variance_ratio_.round(4)}")
    print(f"    Cumulative: {pca.explained_variance_ratio_.cumsum().round(4)}")

    print("\n[3] Training M1 (yield-only features)...")
    feat_m1 = build_m1_features(yield_df)
    print(f"    Features: {feat_m1.shape[1]}")
    pred_m1 = train_and_predict(yield_df, feat_m1, delta_targets, pca, "M1")

    print("\n[4] Training M2 (adaptive IV)...")
    pred_m2 = train_and_predict_m2(yield_df, iv_df, delta_targets, pca)

    # --- Metrics ---
    val_dates = pred_m1.index
    actual = yield_df.loc[val_dates, YIELD_TENORS].values.astype(float)
    rmse_m1 = weighted_rmse(actual, pred_m1.values.astype(float))
    rmse_m2 = weighted_rmse(actual, pred_m2.values.astype(float))

    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n  Weighted RMSE M1: {rmse_m1:.6f}")
    print(f"  Weighted RMSE M2: {rmse_m2:.6f}")
    print(f"  M2 vs M1: {(rmse_m1 - rmse_m2) / rmse_m1 * 100:+.1f}%")

    print(f"\n  {'Tenor':>5} | {'RMSE M1':>10} | {'RMSE M2':>10} | {'Better':>6}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
    for j, tenor in enumerate(YIELD_TENORS):
        r1 = np.sqrt(np.mean((actual[:, j] - pred_m1.values.astype(float)[:, j]) ** 2))
        r2 = np.sqrt(np.mean((actual[:, j] - pred_m2.values.astype(float)[:, j]) ** 2))
        print(f"  {tenor:>5} | {r1:>10.4f} | {r2:>10.4f} | {'M2' if r2 < r1 else 'M1':>6}")

    # --- Save ---
    output_dir = os.path.dirname(os.path.abspath(__file__))
    deltas_path = os.path.join(output_dir, "deltas.xlsx")
    actual_df = yield_df.loc[val_dates, YIELD_TENORS]
    with pd.ExcelWriter(deltas_path, engine="openpyxl") as writer:
        (pred_m1.astype(float) - actual_df.astype(float)).to_excel(writer, sheet_name="Delta_M1")
        (pred_m2.astype(float) - actual_df.astype(float)).to_excel(writer, sheet_name="Delta_M2")
        pred_m1.to_excel(writer, sheet_name="Pred_M1")
        pred_m2.to_excel(writer, sheet_name="Pred_M2")
    print(f"\n  Saved: {deltas_path}")

    print("\n[5] Plotting...")
    path = plot_results(yield_df, pred_m1, pred_m2, output_dir)
    print(f"  Plot: {path}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
