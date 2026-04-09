"""
================================================================================
Model 2: Theta method + HuberRegressor (Direct forecast) ensemble
================================================================================
M1 = yield features only
M2 = yield features + IV features
Ensemble: w * Huber + (1-w) * Theta, w optimised on validation
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
from sklearn.linear_model import LassoCV, HuberRegressor
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

IV_KEY_TENORS = ["1M", "3M", "6M", "1Y", "5Y"]


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
# METRIC
# ==============================================================================
def weighted_rmse(y_true, y_pred):
    T = y_true.shape[0]
    weights = np.array([0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075])
    sq_errors = (y_true - y_pred) ** 2
    return np.sqrt(np.sum(weights[np.newaxis, :] * sq_errors) / T)


# ==============================================================================
# THETA METHOD (manual)
# ==============================================================================
def theta_manual(series, n_forecast=6):
    """Theta = 0.5 * SES_forecast + 0.5 * linear_drift"""
    n = len(series)
    last = series[-1]
    # SES with alpha search
    best_alpha = 0.3
    best_sse = np.inf
    for alpha in np.arange(0.05, 1.0, 0.05):
        level = series[0]
        sse = 0
        for t in range(1, n):
            level = alpha * series[t] + (1 - alpha) * level
            sse += (series[t] - level) ** 2
        if sse < best_sse:
            best_sse = sse
            best_alpha = alpha
    level = series[0]
    for t in range(1, n):
        level = best_alpha * series[t] + (1 - best_alpha) * level
    ses_forecast = np.full(n_forecast, level)
    drift = (series[-1] - series[0]) / (n - 1)
    drift_forecast = last + drift * np.arange(1, n_forecast + 1)
    return 0.5 * ses_forecast + 0.5 * drift_forecast


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def build_yield_features(yield_df):
    """~30 yield-only features for M1."""
    feat = pd.DataFrame(index=yield_df.index)
    # Lags 1, 2 for all tenors (18)
    for lag in [1, 2]:
        shifted = yield_df[YIELD_TENORS].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in YIELD_TENORS]
        feat = pd.concat([feat, shifted], axis=1)
    # Delta1 (9)
    delta1 = yield_df[YIELD_TENORS].diff(1)
    delta1.columns = [f"{c}_delta1" for c in YIELD_TENORS]
    feat = pd.concat([feat, delta1], axis=1)
    # Spreads (3)
    feat['spread_2Y_ON'] = yield_df['2Y'] - yield_df['O/N']
    feat['spread_1Y_3M'] = yield_df['1Y'] - yield_df['3M']
    feat['spread_6M_1M'] = yield_df['6M'] - yield_df['1M']
    return feat


def build_iv_features(iv_df):
    """~7 IV features for M2 addition."""
    feat = pd.DataFrame(index=iv_df.index)
    # IV lag1 for 5 key tenors
    available_key = [t for t in IV_KEY_TENORS if t in iv_df.columns]
    for t in available_key:
        feat[f"iv_{t}_lag1"] = iv_df[t].shift(1)
    # IV spread
    if "10Y" in iv_df.columns and "1M" in iv_df.columns:
        feat['iv_spread_10Y_1M'] = iv_df['10Y'].shift(1) - iv_df['1M'].shift(1)
    # Vol of vol
    iv_std3 = iv_df.rolling(3).std().shift(1)
    feat['iv_vol_of_vol'] = iv_std3.mean(axis=1)
    return feat


# ==============================================================================
# HUBER PIPELINE (direct, per-tenor, per-horizon)
# ==============================================================================
def train_huber_direct(X_train, y_train_df, X_val, n_horizons=6):
    """
    For each tenor and each horizon h=1..6:
      1. LassoCV selects features
      2. HuberRegressor trains on selected features
    Returns predictions array (n_horizons, n_tenors).
    """
    n_tenors = len(YIELD_TENORS)
    preds = np.zeros((n_horizons, n_tenors))

    for j, tenor in enumerate(YIELD_TENORS):
        print(f"  Huber training tenor={tenor} ...", end=" ")
        for h in range(1, n_horizons + 1):
            target = y_train_df[tenor].shift(-h)
            mask = target.notna() & X_train.notna().all(axis=1)
            X_tr = X_train.loc[mask]
            y_tr = target.loc[mask].values

            if len(X_tr) < 10:
                preds[h - 1, j] = y_train_df[tenor].iloc[-1]
                continue

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_tr)

            # LassoCV feature selection
            lasso = LassoCV(alphas=np.logspace(-5, 1, 80), cv=5, max_iter=20000)
            lasso.fit(X_sc, y_tr)
            selected = np.abs(lasso.coef_) > 1e-8
            if selected.sum() < 2:
                # fallback: top 5 by absolute coefficient
                top_idx = np.argsort(np.abs(lasso.coef_))[-5:]
                selected = np.zeros(X_sc.shape[1], dtype=bool)
                selected[top_idx] = True

            X_sel = X_sc[:, selected]

            # HuberRegressor
            huber = HuberRegressor(epsilon=1.35, max_iter=500, alpha=0.01)
            huber.fit(X_sel, y_tr)

            # Predict on last row of validation features
            X_val_sc = scaler.transform(X_val.values.reshape(1, -1))
            pred = huber.predict(X_val_sc[:, selected])
            preds[h - 1, j] = pred[0]
        print("done")

    return preds


# ==============================================================================
# FIND OPTIMAL ENSEMBLE WEIGHT ON VALIDATION
# ==============================================================================
def find_best_weight(theta_preds, huber_preds, actual):
    """Search w in [0,1] that minimises weighted_rmse on validation."""
    best_w = 0.5
    best_rmse = np.inf
    for w in np.arange(0.0, 1.01, 0.05):
        combined = w * huber_preds + (1 - w) * theta_preds
        rmse = weighted_rmse(actual, combined)
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w
    return best_w, best_rmse


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("  Model 2: Theta + HuberRegressor ensemble")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/6] Loading data ...")
    yield_df, iv_df = load_data()
    print(f"  Yield shape: {yield_df.shape}, IV shape: {iv_df.shape}")
    print(f"  Date range: {yield_df.index[0]} .. {yield_df.index[-1]}")

    # --- Split ---
    train_yield = yield_df.loc[:TRAIN_END]
    val_yield = yield_df.loc[VAL_START:VAL_END]
    train_iv = iv_df.loc[:TRAIN_END]

    print(f"  Train: {train_yield.shape[0]} months, Val: {val_yield.shape[0]} months")

    if val_yield.shape[0] < N_VAL_MONTHS:
        print(f"  WARNING: only {val_yield.shape[0]} val months available "
              f"(expected {N_VAL_MONTHS})")

    actual_val = val_yield[YIELD_TENORS].values  # (n_val, 9)

    # --- Theta forecasts (same for M1 and M2) ---
    print("\n[2/6] Computing Theta forecasts ...")
    theta_preds = np.zeros((N_VAL_MONTHS, len(YIELD_TENORS)))
    for j, tenor in enumerate(YIELD_TENORS):
        series = train_yield[tenor].dropna().values
        fc = theta_manual(series, n_forecast=N_VAL_MONTHS)
        theta_preds[:, j] = fc
        print(f"  {tenor}: last={series[-1]:.4f}, "
              f"theta_forecast={fc[0]:.4f}..{fc[-1]:.4f}")

    # --- Build features ---
    print("\n[3/6] Building features ...")
    X_yield = build_yield_features(yield_df)
    X_iv = build_iv_features(iv_df)

    # M1 features
    X_m1_all = X_yield.copy()
    X_m1_all = X_m1_all.ffill(axis=1).fillna(0)

    # M2 features
    X_m2_all = pd.concat([X_yield, X_iv], axis=1)
    X_m2_all = X_m2_all.ffill(axis=1).fillna(0)

    X_m1_train = X_m1_all.loc[:TRAIN_END]
    X_m2_train = X_m2_all.loc[:TRAIN_END]

    # Last known feature row (end of training) for prediction
    X_m1_last = X_m1_all.loc[TRAIN_END].iloc[-1] if isinstance(X_m1_all.loc[TRAIN_END], pd.DataFrame) else X_m1_all.loc[TRAIN_END]
    X_m2_last = X_m2_all.loc[TRAIN_END].iloc[-1] if isinstance(X_m2_all.loc[TRAIN_END], pd.DataFrame) else X_m2_all.loc[TRAIN_END]

    print(f"  M1 features: {X_m1_train.shape[1]}")
    print(f"  M2 features: {X_m2_train.shape[1]}")

    # --- Huber M1 ---
    print("\n[4/6] Training Huber M1 (yield only) ...")
    huber_m1_preds = train_huber_direct(
        X_m1_train, train_yield[YIELD_TENORS], X_m1_last,
        n_horizons=N_VAL_MONTHS
    )

    # --- Huber M2 ---
    print("\n[5/6] Training Huber M2 (yield + IV) ...")
    huber_m2_preds = train_huber_direct(
        X_m2_train, train_yield[YIELD_TENORS], X_m2_last,
        n_horizons=N_VAL_MONTHS
    )

    # --- Ensemble ---
    print("\n[6/6] Optimising ensemble weights ...")
    n_actual = min(actual_val.shape[0], N_VAL_MONTHS)
    theta_trimmed = theta_preds[:n_actual]
    huber_m1_trimmed = huber_m1_preds[:n_actual]
    huber_m2_trimmed = huber_m2_preds[:n_actual]

    w_m1, rmse_m1 = find_best_weight(theta_trimmed, huber_m1_trimmed, actual_val[:n_actual])
    w_m2, rmse_m2 = find_best_weight(theta_trimmed, huber_m2_trimmed, actual_val[:n_actual])

    final_m1 = w_m1 * huber_m1_trimmed + (1 - w_m1) * theta_trimmed
    final_m2 = w_m2 * huber_m2_trimmed + (1 - w_m2) * theta_trimmed

    # Component RMSEs
    rmse_theta = weighted_rmse(actual_val[:n_actual], theta_trimmed)
    rmse_huber_m1 = weighted_rmse(actual_val[:n_actual], huber_m1_trimmed)
    rmse_huber_m2 = weighted_rmse(actual_val[:n_actual], huber_m2_trimmed)

    print(f"\n{'='*70}")
    print(f"  RESULTS (validation {VAL_START} .. {VAL_END})")
    print(f"{'='*70}")
    print(f"  Theta only RMSE:          {rmse_theta:.6f}")
    print(f"  Huber M1 only RMSE:       {rmse_huber_m1:.6f}")
    print(f"  Huber M2 only RMSE:       {rmse_huber_m2:.6f}")
    print(f"  Ensemble M1 (w={w_m1:.2f}): {rmse_m1:.6f}")
    print(f"  Ensemble M2 (w={w_m2:.2f}): {rmse_m2:.6f}")

    # --- Deltas xlsx ---
    out_dir = os.path.dirname(os.path.abspath(__file__))

    val_dates = val_yield.index[:n_actual]
    delta_rows = []
    for i, dt in enumerate(val_dates):
        for j, tenor in enumerate(YIELD_TENORS):
            delta_rows.append({
                'Date': dt,
                'Tenor': tenor,
                'Actual': actual_val[i, j],
                'Theta': theta_trimmed[i, j],
                'Huber_M1': huber_m1_trimmed[i, j],
                'Huber_M2': huber_m2_trimmed[i, j],
                'Ensemble_M1': final_m1[i, j],
                'Ensemble_M2': final_m2[i, j],
                'Delta_M1': final_m1[i, j] - actual_val[i, j],
                'Delta_M2': final_m2[i, j] - actual_val[i, j],
            })
    delta_df = pd.DataFrame(delta_rows)
    xlsx_path = os.path.join(out_dir, 'validation_deltas.xlsx')
    delta_df.to_excel(xlsx_path, index=False)
    print(f"\n  Deltas saved to {xlsx_path}")

    # --- Visualization: 9 rows x 2 cols ---
    print("  Creating validation plot ...")
    fig, axes = plt.subplots(9, 2, figsize=(18, 36))
    fig.suptitle('Model 2: Theta + HuberRegressor  |  M1 (left) vs M2 (right)',
                 fontsize=16, fontweight='bold', y=0.995)

    history_dates = yield_df.index
    val_plot_dates = val_yield.index[:n_actual]

    for j, tenor in enumerate(YIELD_TENORS):
        for col_idx, (model_name, ensemble_pred, huber_pred, w) in enumerate([
            ("M1", final_m1, huber_m1_trimmed, w_m1),
            ("M2", final_m2, huber_m2_trimmed, w_m2),
        ]):
            ax = axes[j, col_idx]

            # Full history
            ax.plot(history_dates, yield_df[tenor].values,
                    color='#FF69B4', linewidth=0.8, alpha=0.7, label='History')

            # Actual validation
            ax.plot(val_plot_dates, actual_val[:n_actual, j],
                    color='black', linewidth=1.5, linestyle='--',
                    marker='o', markersize=3, label='Actual')

            # Build forecast line starting from last train point
            last_train_date = train_yield.index[-1]
            last_train_val = train_yield[tenor].iloc[-1]

            fc_dates = [last_train_date] + list(val_plot_dates)
            fc_ensemble = [last_train_val] + list(ensemble_pred[:n_actual, j])

            ax.plot(fc_dates, fc_ensemble,
                    color='#2ECC71', linewidth=2, marker='s', markersize=3,
                    label=f'Ensemble (w={w:.2f})')

            ax.set_title(f'{tenor} - {model_name}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    png_path = os.path.join(out_dir, 'validation_all_tenors.png')
    fig.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved to {png_path}")

    print(f"\n{'='*70}")
    print(f"  FINAL weighted RMSE  M1={rmse_m1:.6f}  M2={rmse_m2:.6f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
