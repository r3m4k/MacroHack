"""
================================================================================
Model 3: Last Value + Delta Forecast (ElasticNet)
================================================================================
y_hat(t+h) = y(t) + delta_hat(t, h)

Instead of predicting absolute yield values, we predict CHANGES (deltas).
The prediction is always anchored to the last known value, preventing divergence.

M1: yield-only features (~41)
M2: yield + IV features  (~52)
================================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

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
    """For horizon h, the target is y(t+h) - y(t)."""
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
    """
    M1 features (~41 total):
    - Current values (9)
    - Delta1: diff(1) (9)
    - Delta3: diff(3) (9)
    - 3 spreads: 2Y-O/N, 1Y-3M, 6M-1M
    - Change of spreads: diff(1) of spread_2Y_ON and spread_1Y_3M (2)
    - Deviation from MA6 for each tenor (9)
    """
    df = yield_df[YIELD_TENORS].copy()
    features = pd.DataFrame(index=df.index)

    # Current values
    for t in YIELD_TENORS:
        features[f"cur_{t}"] = df[t]

    # Delta1
    d1 = df.diff(1)
    for t in YIELD_TENORS:
        features[f"delta1_{t}"] = d1[t]

    # Delta3
    d3 = df.diff(3)
    for t in YIELD_TENORS:
        features[f"delta3_{t}"] = d3[t]

    # Spreads
    features["spread_2Y_ON"] = df["2Y"] - df["O/N"]
    features["spread_1Y_3M"] = df["1Y"] - df["3M"]
    features["spread_6M_1M"] = df["6M"] - df["1M"]

    # Change of spreads
    features["d_spread_2Y_ON"] = features["spread_2Y_ON"].diff(1)
    features["d_spread_1Y_3M"] = features["spread_1Y_3M"].diff(1)

    # Deviation from MA6 (mean-reversion signal)
    ma6 = df.rolling(6).mean()
    for t in YIELD_TENORS:
        features[f"dev_ma6_{t}"] = df[t] - ma6[t]

    return features


def build_m2_features(yield_df, iv_df):
    """
    M2 features: M1 + IV features (~52 total).
    Extra ~11 features:
    - IV delta1 for 5 key tenors, shifted by 1 (5)
    - IV lag1 for 5 key tenors (5)
    - IV term spread: 10Y-1M shifted by 1 (1)
    """
    features = build_m1_features(yield_df)

    available_key = [t for t in IV_KEY_TENORS if t in iv_df.columns]

    # IV lag1
    for t in available_key:
        features[f"iv_lag1_{t}"] = iv_df[t].shift(1)

    # IV delta1 (shifted by 1 to avoid look-ahead)
    iv_d1 = iv_df.diff(1).shift(1)
    for t in available_key:
        features[f"iv_delta1_{t}"] = iv_d1[t]

    # IV term spread
    if "10Y" in iv_df.columns and "1M" in iv_df.columns:
        features["iv_spread_10Y_1M"] = (iv_df["10Y"] - iv_df["1M"]).shift(1)

    return features


# ==============================================================================
# METRIC
# ==============================================================================
def weighted_rmse(y_true, y_pred):
    """
    Weighted RMSE: O/N gets 0.4, others get 0.075 each.
    y_true, y_pred: arrays of shape (T, 9)
    """
    T = y_true.shape[0]
    weights = np.array([0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075])
    sq_errors = (y_true - y_pred) ** 2
    return np.sqrt(np.sum(weights[np.newaxis, :] * sq_errors) / T)


# ==============================================================================
# TRAINING & PREDICTION
# ==============================================================================
def train_and_predict(yield_df, features_df, delta_targets, model_label):
    """
    Train 6 horizons x 9 tenors = 54 ElasticNet models.
    Returns predictions DataFrame (N_VAL_MONTHS x 9).
    """
    # Split
    train_mask = yield_df.index <= TRAIN_END
    train_dates = yield_df.index[train_mask]

    # Last known values (anchor point)
    last_known = yield_df.loc[yield_df.index <= TRAIN_END, YIELD_TENORS].iloc[-1]
    print(f"\n  [{model_label}] Last known values ({TRAIN_END}):")
    for t in YIELD_TENORS:
        print(f"    {t}: {last_known[t]:.4f}")

    # Validation dates
    val_dates = yield_df.index[(yield_df.index >= VAL_START) & (yield_df.index <= VAL_END)]
    n_val = len(val_dates)
    print(f"  [{model_label}] Validation months: {n_val}")

    predictions = pd.DataFrame(index=val_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, n_val + 1):
        dt = delta_targets[h]
        # Training rows: must be in train period AND have valid features/targets
        valid_idx = dt.index.intersection(train_dates).intersection(features_df.dropna().index)

        X_train = features_df.loc[valid_idx].copy()
        X_train = X_train.ffill(axis=1).fillna(0)

        # Historical deltas for clipping
        hist_deltas = dt.loc[valid_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # The anchor row: last known date in training set
        anchor_date = train_dates[-1]
        if anchor_date not in features_df.index:
            print(f"    WARNING: anchor date {anchor_date} not in features, skipping h={h}")
            continue

        x_anchor = features_df.loc[[anchor_date]].copy()
        x_anchor = x_anchor.ffill(axis=1).fillna(0)
        x_anchor_scaled = scaler.transform(x_anchor)

        target_date = val_dates[h - 1] if h - 1 < len(val_dates) else None
        if target_date is None:
            continue

        print(f"    h={h} -> {target_date.strftime('%Y-%m')}: ", end="")

        for j, tenor in enumerate(YIELD_TENORS):
            y_train = dt.loc[valid_idx, tenor].values

            model = ElasticNet(alpha=0.1, l1_ratio=0.7, max_iter=10000)
            model.fit(X_scaled, y_train)

            delta_pred = model.predict(x_anchor_scaled)[0]

            # Clip delta to prevent unrealistic jumps
            max_delta = hist_deltas[tenor].abs().quantile(0.95)
            delta_pred = np.clip(delta_pred, -max_delta * h, max_delta * h)

            # Final prediction anchored to last known value
            prediction = last_known[tenor] + delta_pred
            predictions.loc[target_date, tenor] = prediction

        preds_row = predictions.loc[target_date]
        print(f"O/N={preds_row['O/N']:.4f}  2Y={preds_row['2Y']:.4f}")

    return predictions


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("  Model 3: Last Value + Delta Forecast (ElasticNet)")
    print("=" * 70)

    # --- Load data ---
    print("\n[1] Loading data...")
    yield_df, iv_df = load_data()
    print(f"    Yield: {yield_df.shape[0]} months, {yield_df.shape[1]} tenors")
    print(f"    IV:    {iv_df.shape[0]} months, {iv_df.shape[1]} tenors")

    # --- Prepare delta targets ---
    print("\n[2] Preparing delta targets...")
    delta_targets = prepare_delta_targets(yield_df)
    for h in range(1, N_VAL_MONTHS + 1):
        print(f"    h={h}: {len(delta_targets[h])} samples")

    # --- Build features ---
    print("\n[3] Building features...")
    feat_m1 = build_m1_features(yield_df)
    feat_m2 = build_m2_features(yield_df, iv_df)
    print(f"    M1 features: {feat_m1.shape[1]}")
    print(f"    M2 features: {feat_m2.shape[1]}")

    # --- Train & predict M1 ---
    print("\n[4] Training M1 (yield-only)...")
    pred_m1 = train_and_predict(yield_df, feat_m1, delta_targets, "M1")

    # --- Train & predict M2 ---
    print("\n[5] Training M2 (yield + IV)...")
    pred_m2 = train_and_predict(yield_df, feat_m2, delta_targets, "M2")

    # --- Actual validation values ---
    val_dates = pred_m1.index
    actual = yield_df.loc[val_dates, YIELD_TENORS]

    # --- Compute metrics ---
    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)

    actual_arr = actual.values.astype(float)
    pred_m1_arr = pred_m1.values.astype(float)
    pred_m2_arr = pred_m2.values.astype(float)

    rmse_m1 = weighted_rmse(actual_arr, pred_m1_arr)
    rmse_m2 = weighted_rmse(actual_arr, pred_m2_arr)

    print(f"\n  Weighted RMSE (M1): {rmse_m1:.6f}")
    print(f"  Weighted RMSE (M2): {rmse_m2:.6f}")
    print(f"  Better model: {'M2' if rmse_m2 < rmse_m1 else 'M1'}")

    # Per-tenor RMSE
    print(f"\n  {'Tenor':>5} | {'Weight':>6} | {'RMSE M1':>10} | {'RMSE M2':>10}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for j, tenor in enumerate(YIELD_TENORS):
        w = WEIGHT_ON if j == 0 else WEIGHT_OTHER
        rmse_t_m1 = np.sqrt(np.mean((actual_arr[:, j] - pred_m1_arr[:, j]) ** 2))
        rmse_t_m2 = np.sqrt(np.mean((actual_arr[:, j] - pred_m2_arr[:, j]) ** 2))
        print(f"  {tenor:>5} | {w:>6.3f} | {rmse_t_m1:>10.6f} | {rmse_t_m2:>10.6f}")

    # --- Save deltas to Excel ---
    output_dir = os.path.dirname(os.path.abspath(__file__))

    deltas_path = os.path.join(output_dir, "deltas.xlsx")
    with pd.ExcelWriter(deltas_path, engine="openpyxl") as writer:
        # M1 deltas (predicted - last known)
        last_known = yield_df.loc[yield_df.index <= TRAIN_END, YIELD_TENORS].iloc[-1]
        deltas_m1 = pred_m1.astype(float) - last_known.values
        deltas_m2 = pred_m2.astype(float) - last_known.values
        deltas_m1.to_excel(writer, sheet_name="M1_deltas")
        deltas_m2.to_excel(writer, sheet_name="M2_deltas")
        pred_m1.to_excel(writer, sheet_name="M1_predictions")
        pred_m2.to_excel(writer, sheet_name="M2_predictions")
        actual.to_excel(writer, sheet_name="Actual")
    print(f"\n  Deltas saved to: {deltas_path}")

    # --- Visualization ---
    print("\n[6] Creating visualization...")
    fig, axes = plt.subplots(9, 2, figsize=(18, 36))
    fig.suptitle("Model 3: Delta Forecast (ElasticNet) — Validation",
                 fontsize=16, fontweight="bold", y=0.995)

    # History end point
    history = yield_df.loc[yield_df.index <= TRAIN_END, YIELD_TENORS]
    # Append anchor to forecast for connected line
    anchor_date = history.index[-1]

    for j, tenor in enumerate(YIELD_TENORS):
        # --- M1 subplot ---
        ax = axes[j, 0]
        # Full history
        ax.plot(history.index, history[tenor], color="hotpink", linewidth=1.0,
                label="History")
        # Forecast (connected from anchor)
        fc_dates_m1 = pd.DatetimeIndex([anchor_date]).append(pred_m1.index)
        fc_vals_m1 = np.concatenate([[history[tenor].iloc[-1]],
                                      pred_m1[tenor].values.astype(float)])
        ax.plot(fc_dates_m1, fc_vals_m1, color="limegreen", linewidth=2.0,
                marker="o", markersize=4, label="M1 Forecast")
        # Actual
        if len(actual) > 0:
            act_dates = pd.DatetimeIndex([anchor_date]).append(actual.index)
            act_vals = np.concatenate([[history[tenor].iloc[-1]],
                                       actual[tenor].values.astype(float)])
            ax.plot(act_dates, act_vals, color="blue", linewidth=1.5,
                    linestyle="--", marker="s", markersize=3, label="Actual")
        ax.axvline(x=anchor_date, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"M1 — {tenor}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

        # --- M2 subplot ---
        ax = axes[j, 1]
        ax.plot(history.index, history[tenor], color="hotpink", linewidth=1.0,
                label="History")
        fc_dates_m2 = pd.DatetimeIndex([anchor_date]).append(pred_m2.index)
        fc_vals_m2 = np.concatenate([[history[tenor].iloc[-1]],
                                      pred_m2[tenor].values.astype(float)])
        ax.plot(fc_dates_m2, fc_vals_m2, color="limegreen", linewidth=2.0,
                marker="o", markersize=4, label="M2 Forecast")
        if len(actual) > 0:
            ax.plot(act_dates, act_vals, color="blue", linewidth=1.5,
                    linestyle="--", marker="s", markersize=3, label="Actual")
        ax.axvline(x=anchor_date, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"M2 — {tenor}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    png_path = os.path.join(output_dir, "validation_all_tenors.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to: {png_path}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
