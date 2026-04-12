"""
================================================================================
ARIMA + Direct Ridge Ensemble — Yield Curve Forecasting (M1 & M2)
================================================================================
M1: yield features only
M2: yield features + implied volatility features
================================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV, Ridge
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    IV_TENORS = [
        "1M", "2M", "3M", "6M", "9M",
        "1Y", "2Y", "3Y", "4Y", "5Y",
        "6Y", "7Y", "8Y", "9Y", "10Y",
    ]
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
# FEATURE ENGINEERING
# ==============================================================================
def build_features(yield_df, iv_df, include_iv=False):
    """
    Build feature matrix.
    M1 (~30 features): lags, deltas, spreads from yield curve only.
    M2 (~37 features): M1 + IV lag1 for key tenors, iv spread, vol-of-vol.
    """
    feat = pd.DataFrame(index=yield_df.index)

    # Lags 1, 2 for each yield tenor (18 cols)
    for lag in [1, 2]:
        shifted = yield_df[YIELD_TENORS].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in YIELD_TENORS]
        feat = pd.concat([feat, shifted], axis=1)

    # Delta1 for each yield tenor (9 cols)
    delta1 = yield_df[YIELD_TENORS].diff(1)
    delta1.columns = [f"{c}_delta1" for c in YIELD_TENORS]
    feat = pd.concat([feat, delta1], axis=1)

    # 3 key spreads (3 cols)
    feat["spread_2Y_ON"] = yield_df["2Y"] - yield_df["O/N"]
    feat["spread_1Y_3M"] = yield_df["1Y"] - yield_df["3M"]
    feat["spread_6M_1M"] = yield_df["6M"] - yield_df["1M"]

    if include_iv:
        # IV lag1 for 5 key tenors (5 cols)
        available_iv_keys = [t for t in IV_KEY_TENORS if t in iv_df.columns]
        for t in available_iv_keys:
            feat[f"iv_{t}_lag1"] = iv_df[t].shift(1)

        # IV spread (1 col)
        if "10Y" in iv_df.columns and "1M" in iv_df.columns:
            feat["iv_spread_10Y_1M"] = (iv_df["10Y"] - iv_df["1M"]).shift(1)

        # Vol-of-vol: rolling std across all IV tenors, mean (1 col)
        iv_std3 = iv_df.rolling(3).std().shift(1)
        feat["iv_vol_of_vol"] = iv_std3.mean(axis=1)

    return feat


# ==============================================================================
# ARIMA COMPONENT
# ==============================================================================
def fit_arima_all_tenors(yield_train):
    """Fit one auto_arima per tenor, return dict of models."""
    models = {}
    for tenor in YIELD_TENORS:
        print(f"  ARIMA fitting: {tenor} ...", end=" ")
        series = yield_train[tenor].dropna()
        model = auto_arima(
            series,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=5, max_q=5, max_d=2,
        )
        models[tenor] = model
        print(f"order={model.order}")
    return models


def predict_arima(models, n_periods=6):
    """Return DataFrame (n_periods x len(YIELD_TENORS)) of ARIMA forecasts."""
    preds = {}
    for tenor in YIELD_TENORS:
        preds[tenor] = models[tenor].predict(n_periods=n_periods)
    return pd.DataFrame(preds)


# ==============================================================================
# DIRECT RIDGE COMPONENT
# ==============================================================================
def fit_direct_ridge(yield_df, features, tenor, h):
    """
    Train a Ridge model for a single tenor at horizon h.
    Uses LassoCV for feature selection, then Ridge for final model.
    Returns (ridge_model, scaler, selected_features_list).
    """
    target = yield_df[tenor].shift(-h)
    valid = features.dropna().index.intersection(target.dropna().index)
    X = features.loc[valid]
    y = target.loc[valid]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LassoCV for feature selection
    lasso = LassoCV(alphas=np.logspace(-5, 1, 80), cv=5, max_iter=20000)
    lasso.fit(X_scaled, y)
    mask = np.abs(lasso.coef_) > 1e-8
    selected_cols = [c for c, m in zip(X.columns, mask) if m]

    if len(selected_cols) < 2:
        # fallback: use top-5 by absolute coef
        top_idx = np.argsort(np.abs(lasso.coef_))[::-1][:5]
        selected_cols = [X.columns[i] for i in top_idx]

    # Ridge on selected features
    X_sel = X[selected_cols]
    scaler_sel = StandardScaler()
    X_sel_scaled = scaler_sel.fit_transform(X_sel)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_sel_scaled, y)

    return ridge, scaler_sel, selected_cols


def predict_direct_ridge(models_dict, last_features_row):
    """
    Given models_dict[tenor][h] = (ridge, scaler, cols) and the last row of features,
    produce predictions for h=1..6 for all tenors.
    Returns DataFrame (6 x len(YIELD_TENORS)).
    """
    preds = {}
    for tenor in YIELD_TENORS:
        tenor_preds = []
        for h in range(1, N_VAL_MONTHS + 1):
            ridge, scaler, cols = models_dict[tenor][h]
            x = last_features_row[cols].values.reshape(1, -1)
            x_scaled = scaler.transform(x)
            tenor_preds.append(ridge.predict(x_scaled)[0])
        preds[tenor] = tenor_preds
    return pd.DataFrame(preds)


# ==============================================================================
# ENSEMBLE: find optimal weight on validation
# ==============================================================================
def find_best_weight(arima_pred, ridge_pred, actual):
    """
    Grid search over w in [0, 1] for: final = w * ridge + (1-w) * arima.
    Minimises weighted RMSE. Returns best_w.
    """
    best_w = 0.5
    best_score = 1e9
    for w in np.linspace(0, 1, 101):
        blended = w * ridge_pred + (1 - w) * arima_pred
        score = weighted_rmse(actual, blended)
        if score < best_score:
            best_score = score
            best_w = w
    return best_w, best_score


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("  ARIMA + Direct Ridge Ensemble Pipeline")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading data ...")
    yield_df, iv_df = load_data()
    print(f"  Yield shape: {yield_df.shape}, IV shape: {iv_df.shape}")
    print(f"  Date range: {yield_df.index.min()} .. {yield_df.index.max()}")

    # ------------------------------------------------------------------
    # 2. Train / Val split
    # ------------------------------------------------------------------
    print("\n[2/7] Splitting train / validation ...")
    yield_train = yield_df.loc[:TRAIN_END]
    yield_val = yield_df.loc[VAL_START:VAL_END]
    iv_train = iv_df.loc[:TRAIN_END]

    print(f"  Train: {yield_train.shape[0]} months, Val: {yield_val.shape[0]} months")

    actual_val = yield_val[YIELD_TENORS].values
    val_index = yield_val.index

    # ------------------------------------------------------------------
    # 3. ARIMA (same for M1 and M2)
    # ------------------------------------------------------------------
    print("\n[3/7] Fitting ARIMA models ...")
    arima_models = fit_arima_all_tenors(yield_train)
    arima_pred_df = predict_arima(arima_models, n_periods=N_VAL_MONTHS)
    arima_pred_df.index = val_index
    arima_pred = arima_pred_df[YIELD_TENORS].values
    print(f"  ARIMA-only weighted RMSE: {weighted_rmse(actual_val, arima_pred):.6f}")

    # ------------------------------------------------------------------
    # 4. Direct Ridge — M1 (yield features only)
    # ------------------------------------------------------------------
    print("\n[4/7] Building M1 features & fitting Direct Ridge (M1) ...")
    feat_m1 = build_features(yield_df.loc[:TRAIN_END], iv_df.loc[:TRAIN_END], include_iv=False)
    feat_m1 = feat_m1.ffill(axis=1).fillna(0)
    print(f"  M1 feature count: {feat_m1.shape[1]}")

    last_row_m1 = feat_m1.iloc[-1]

    ridge_models_m1 = {}
    for tenor in YIELD_TENORS:
        ridge_models_m1[tenor] = {}
        for h in range(1, N_VAL_MONTHS + 1):
            ridge, scaler, cols = fit_direct_ridge(yield_train, feat_m1, tenor, h)
            ridge_models_m1[tenor][h] = (ridge, scaler, cols)
        print(f"  Ridge M1 done: {tenor}")

    ridge_pred_m1_df = predict_direct_ridge(ridge_models_m1, last_row_m1)
    ridge_pred_m1_df.index = val_index
    ridge_pred_m1 = ridge_pred_m1_df[YIELD_TENORS].values

    # ------------------------------------------------------------------
    # 5. Direct Ridge — M2 (yield + IV features)
    # ------------------------------------------------------------------
    print("\n[5/7] Building M2 features & fitting Direct Ridge (M2) ...")
    feat_m2 = build_features(yield_df.loc[:TRAIN_END], iv_df.loc[:TRAIN_END], include_iv=True)
    feat_m2 = feat_m2.ffill(axis=1).fillna(0)
    print(f"  M2 feature count: {feat_m2.shape[1]}")

    last_row_m2 = feat_m2.iloc[-1]

    ridge_models_m2 = {}
    for tenor in YIELD_TENORS:
        ridge_models_m2[tenor] = {}
        for h in range(1, N_VAL_MONTHS + 1):
            ridge, scaler, cols = fit_direct_ridge(yield_train, feat_m2, tenor, h)
            ridge_models_m2[tenor][h] = (ridge, scaler, cols)
        print(f"  Ridge M2 done: {tenor}")

    ridge_pred_m2_df = predict_direct_ridge(ridge_models_m2, last_row_m2)
    ridge_pred_m2_df.index = val_index
    ridge_pred_m2 = ridge_pred_m2_df[YIELD_TENORS].values

    # ------------------------------------------------------------------
    # 6. Ensemble — find best weights
    # ------------------------------------------------------------------
    print("\n[6/7] Finding ensemble weights ...")

    w_m1, score_m1 = find_best_weight(arima_pred, ridge_pred_m1, actual_val)
    ensemble_m1 = w_m1 * ridge_pred_m1 + (1 - w_m1) * arima_pred
    print(f"  M1: w_ridge={w_m1:.2f}, w_arima={1-w_m1:.2f}, RMSE={score_m1:.6f}")

    w_m2, score_m2 = find_best_weight(arima_pred, ridge_pred_m2, actual_val)
    ensemble_m2 = w_m2 * ridge_pred_m2 + (1 - w_m2) * arima_pred
    print(f"  M2: w_ridge={w_m2:.2f}, w_arima={1-w_m2:.2f}, RMSE={score_m2:.6f}")

    # ------------------------------------------------------------------
    # Deltas to xlsx
    # ------------------------------------------------------------------
    delta_rows = []
    for i, tenor in enumerate(YIELD_TENORS):
        for j, date in enumerate(val_index):
            delta_rows.append({
                "Date": date,
                "Tenor": tenor,
                "Actual": actual_val[j, i],
                "M1_Forecast": ensemble_m1[j, i],
                "M2_Forecast": ensemble_m2[j, i],
                "M1_Delta": ensemble_m1[j, i] - actual_val[j, i],
                "M2_Delta": ensemble_m2[j, i] - actual_val[j, i],
            })
    delta_df = pd.DataFrame(delta_rows)
    xlsx_path = os.path.join(SCRIPT_DIR, "validation_deltas.xlsx")
    delta_df.to_excel(xlsx_path, index=False)
    print(f"\n  Deltas saved to {xlsx_path}")

    # ------------------------------------------------------------------
    # 7. Visualization
    # ------------------------------------------------------------------
    print("\n[7/7] Plotting validation results ...")

    # History starting from 2019-03
    hist_start = "2019-03"
    yield_hist = yield_df.loc[hist_start:]

    fig, axes = plt.subplots(
        len(YIELD_TENORS), 2,
        figsize=(18, 3.2 * len(YIELD_TENORS)),
        sharex=False,
    )
    fig.suptitle(
        "ARIMA + Direct Ridge Ensemble: M1 (left) vs M2 (right)",
        fontsize=16, fontweight="bold", y=1.0,
    )

    last_train_date = yield_train.index[-1]
    last_actual_value = yield_train[YIELD_TENORS].iloc[-1]

    for row_idx, tenor in enumerate(YIELD_TENORS):
        for col_idx, (model_label, ens_pred) in enumerate(
            [("M1", ensemble_m1), ("M2", ensemble_m2)]
        ):
            ax = axes[row_idx, col_idx]

            # Full actual history (from 2019-03 through end of validation)
            actual_series = yield_hist[tenor]
            ax.plot(
                actual_series.index, actual_series.values,
                color="hotpink", linewidth=1.2, label="Actual",
            )

            # Forecast line: starts from last train point, extends through val
            forecast_dates = [last_train_date] + list(val_index)
            forecast_vals = [last_actual_value[tenor]] + list(ens_pred[:, row_idx])
            ax.plot(
                forecast_dates, forecast_vals,
                color="green", linewidth=1.8, linestyle="-", label=f"Forecast {model_label}",
            )

            ax.set_ylabel(tenor, fontsize=10, fontweight="bold")
            ax.legend(loc="upper left", fontsize=7)
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.set_title(f"Model {model_label}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    png_path = os.path.join(SCRIPT_DIR, "validation_all_tenors.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {png_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  M1 Weighted RMSE: {score_m1:.6f}")
    print(f"  M2 Weighted RMSE: {score_m2:.6f}")
    print(f"  ARIMA-only RMSE:  {weighted_rmse(actual_val, arima_pred):.6f}")
    print(f"  Ridge-only M1:    {weighted_rmse(actual_val, ridge_pred_m1):.6f}")
    print(f"  Ridge-only M2:    {weighted_rmse(actual_val, ridge_pred_m2):.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
