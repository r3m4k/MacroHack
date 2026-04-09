"""
================================================================================
Model 3 v2: Delta Forecast — Horizon-Adaptive IV Features
================================================================================
y_hat(t+h) = y(t) + delta_hat(t, h)

Key change vs v1:
  h=1..3: M2 uses full IV features (11 cols)
  h=4..6: M2 uses only iv_spread_10Y_1M (1 col) — most stable long-range signal

M1 unchanged.
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
TRAIN_END = "2025-03"
VAL_START = "2025-04"
VAL_END = "2025-09"
N_VAL_MONTHS = 6
WEIGHT_ON = 0.4
WEIGHT_OTHER = 0.6 / 8

IV_KEY_TENORS = ["1M", "3M", "1Y", "5Y", "10Y"]
IV_SHORT_HORIZON_MAX = 3  # h=1..3: full IV, h=4..6: minimal IV


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
    for h in range(1, N_VAL_MONTHS + 1):
        shifted = yield_df[YIELD_TENORS].shift(-h)
        delta = shifted - yield_df[YIELD_TENORS]
        delta_targets[h] = delta.dropna()
    return delta_targets


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def build_m1_features(yield_df):
    """M1 features (~41)."""
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
    """Full IV feature set for short horizons (h=1..3). ~11 cols."""
    features = pd.DataFrame(index=iv_df.index)
    available_key = [t for t in IV_KEY_TENORS if t in iv_df.columns]

    for t in available_key:
        features[f"iv_lag1_{t}"] = iv_df[t].shift(1)

    iv_d1 = iv_df.diff(1).shift(1)
    for t in available_key:
        features[f"iv_delta1_{t}"] = iv_d1[t]

    if "10Y" in iv_df.columns and "1M" in iv_df.columns:
        features["iv_spread_10Y_1M"] = (iv_df["10Y"] - iv_df["1M"]).shift(1)

    return features


def build_iv_features_minimal(iv_df):
    """Minimal IV feature set for long horizons (h=4..6). 1 col."""
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
    sq_errors = (y_true - y_pred) ** 2
    return np.sqrt(np.sum(weights[np.newaxis, :] * sq_errors) / T)


# ==============================================================================
# TRAINING & PREDICTION — M1 (unchanged from v1)
# ==============================================================================
def train_and_predict_m1(yield_df, features_df, delta_targets):
    train_dates = yield_df.index[yield_df.index <= TRAIN_END]
    last_known = yield_df.loc[yield_df.index <= TRAIN_END, YIELD_TENORS].iloc[-1]
    val_dates = yield_df.index[(yield_df.index >= VAL_START) & (yield_df.index <= VAL_END)]
    n_val = len(val_dates)

    predictions = pd.DataFrame(index=val_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, n_val + 1):
        dt = delta_targets[h]
        valid_idx = dt.index.intersection(train_dates).intersection(features_df.dropna().index)

        X_train = features_df.loc[valid_idx].ffill(axis=1).fillna(0)
        hist_deltas = dt.loc[valid_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        anchor_date = train_dates[-1]
        x_anchor = features_df.loc[[anchor_date]].ffill(axis=1).fillna(0)
        x_anchor_scaled = scaler.transform(x_anchor)

        target_date = val_dates[h - 1]
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
# TRAINING & PREDICTION — M2 (horizon-adaptive IV)
# ==============================================================================
def train_and_predict_m2(yield_df, iv_df, delta_targets):
    """
    h=1..3: M1 features + full IV features (~52 cols)
    h=4..6: M1 features + minimal IV (iv_spread_10Y_1M only, ~42 cols)
    """
    train_dates = yield_df.index[yield_df.index <= TRAIN_END]
    last_known = yield_df.loc[yield_df.index <= TRAIN_END, YIELD_TENORS].iloc[-1]
    val_dates = yield_df.index[(yield_df.index >= VAL_START) & (yield_df.index <= VAL_END)]
    n_val = len(val_dates)

    # Pre-build both feature sets
    feat_m1 = build_m1_features(yield_df)
    iv_full = build_iv_features_full(iv_df)
    iv_minimal = build_iv_features_minimal(iv_df)

    feat_full = pd.concat([feat_m1, iv_full], axis=1)
    feat_min = pd.concat([feat_m1, iv_minimal], axis=1)

    print(f"    Short-horizon features (h<=3): {feat_full.shape[1]} cols "
          f"(+{iv_full.shape[1]} IV)")
    print(f"    Long-horizon features  (h>=4): {feat_min.shape[1]} cols "
          f"(+{iv_minimal.shape[1]} IV)")

    predictions = pd.DataFrame(index=val_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, n_val + 1):
        # Select feature set based on horizon
        if h <= IV_SHORT_HORIZON_MAX:
            features_df = feat_full
            iv_label = "full IV"
        else:
            features_df = feat_min
            iv_label = "min IV"

        dt = delta_targets[h]
        valid_idx = dt.index.intersection(train_dates).intersection(features_df.dropna().index)

        X_train = features_df.loc[valid_idx].ffill(axis=1).fillna(0)
        hist_deltas = dt.loc[valid_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        anchor_date = train_dates[-1]
        x_anchor = features_df.loc[[anchor_date]].ffill(axis=1).fillna(0)
        x_anchor_scaled = scaler.transform(x_anchor)

        target_date = val_dates[h - 1]
        print(f"    h={h} [{iv_label:>8}] -> {target_date.strftime('%Y-%m')}: ", end="")

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
    anchor_date = yield_df.loc[yield_df.index <= TRAIN_END].index[-1]
    last_known = history.loc[anchor_date]

    for i, tenor in enumerate(YIELD_TENORS):
        for j, (pred, model_name) in enumerate([
            (pred_m1, 'M1'), (pred_m2, 'M2')
        ]):
            ax = axes[i, j]

            ax.plot(history.index, history[tenor], color='#FF69B4', linewidth=1.5,
                    label='Actual')

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
                    f'M2 (h<=3: full IV, h>3: iv_spread only)'
                ax.set_title(subtitle, fontsize=12, fontweight='bold')

    plt.suptitle('Model 3 v2: Delta Forecast — Horizon-Adaptive IV',
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
    print("  Model 3 v2: Delta Forecast — Horizon-Adaptive IV")
    print("=" * 70)

    print("\n[1] Loading data...")
    yield_df, iv_df = load_data()
    print(f"    Yield: {yield_df.shape}")
    print(f"    IV:    {iv_df.shape}")

    print("\n[2] Preparing delta targets...")
    delta_targets = prepare_delta_targets(yield_df)

    print("\n[3] Building M1 features...")
    feat_m1 = build_m1_features(yield_df)
    print(f"    M1: {feat_m1.shape[1]} features")

    print("\n[4] Training M1...")
    pred_m1 = train_and_predict_m1(yield_df, feat_m1, delta_targets)

    print("\n[5] Training M2 (horizon-adaptive IV)...")
    pred_m2 = train_and_predict_m2(yield_df, iv_df, delta_targets)

    # --- Metrics ---
    val_dates = pred_m1.index
    actual = yield_df.loc[val_dates, YIELD_TENORS]
    actual_arr = actual.values.astype(float)
    pred_m1_arr = pred_m1.values.astype(float)
    pred_m2_arr = pred_m2.values.astype(float)

    rmse_m1 = weighted_rmse(actual_arr, pred_m1_arr)
    rmse_m2 = weighted_rmse(actual_arr, pred_m2_arr)

    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n  Weighted RMSE M1: {rmse_m1:.6f}")
    print(f"  Weighted RMSE M2: {rmse_m2:.6f}")
    improvement = (rmse_m1 - rmse_m2) / rmse_m1 * 100
    print(f"  M2 vs M1: {improvement:+.1f}%")

    print(f"\n  {'Tenor':>5} | {'RMSE M1':>10} | {'RMSE M2':>10} | {'Better':>6}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
    for j, tenor in enumerate(YIELD_TENORS):
        r1 = np.sqrt(np.mean((actual_arr[:, j] - pred_m1_arr[:, j]) ** 2))
        r2 = np.sqrt(np.mean((actual_arr[:, j] - pred_m2_arr[:, j]) ** 2))
        better = "M2" if r2 < r1 else "M1"
        print(f"  {tenor:>5} | {r1:>10.4f} | {r2:>10.4f} | {better:>6}")

    # --- Save ---
    output_dir = os.path.dirname(os.path.abspath(__file__))

    deltas_path = os.path.join(output_dir, "deltas.xlsx")
    with pd.ExcelWriter(deltas_path, engine="openpyxl") as writer:
        (pred_m1.astype(float) - actual.astype(float)).to_excel(writer, sheet_name="Delta_M1")
        (pred_m2.astype(float) - actual.astype(float)).to_excel(writer, sheet_name="Delta_M2")
        pred_m1.to_excel(writer, sheet_name="Pred_M1")
        pred_m2.to_excel(writer, sheet_name="Pred_M2")
        actual.to_excel(writer, sheet_name="Actual")
    print(f"\n  Saved: {deltas_path}")

    print("\n[6] Plotting...")
    path = plot_results(yield_df, pred_m1, pred_m2, output_dir)
    print(f"  Plot: {path}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
