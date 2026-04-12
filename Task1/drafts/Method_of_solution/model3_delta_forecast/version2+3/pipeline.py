"""
================================================================================
Model 3 v2+3: Ensemble of ElasticNet (v2) + PCA-Delta Ridge (v3)
================================================================================
M1: pure v3 (PCA-delta Ridge) — w_v2=0.0
M2: 70% v2 + 30% v3 — w_v2 optimized on validation

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


def get_features_for_horizon(feat_m1, iv_full, iv_min, h, model_type):
    """Select feature set based on model type and horizon."""
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
# METRIC
# ==============================================================================
def weighted_rmse(y_true, y_pred):
    T = y_true.shape[0]
    weights = np.array([0.4, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075])
    return np.sqrt(np.sum(weights[np.newaxis, :] * (y_true - y_pred) ** 2) / T)


# ==============================================================================
# V2: ElasticNet per-tenor
# ==============================================================================
def predict_v2(yield_df, delta_targets, feat_m1, iv_full, iv_min,
               model_type, train_end):
    train_dates = yield_df.index[yield_df.index <= train_end]
    last_known = yield_df.loc[train_dates[-1], YIELD_TENORS]
    val_dates = yield_df.index[(yield_df.index >= VAL_START) & (yield_df.index <= VAL_END)]

    predictions = pd.DataFrame(index=val_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, len(val_dates) + 1):
        features_df = get_features_for_horizon(feat_m1, iv_full, iv_min, h, model_type)
        dt = delta_targets[h]
        valid_idx = dt.index.intersection(train_dates).intersection(features_df.dropna().index)

        X_train = features_df.loc[valid_idx].ffill(axis=1).fillna(0)
        hist_deltas = dt.loc[valid_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        x_anchor = features_df.loc[[train_dates[-1]]].ffill(axis=1).fillna(0)
        x_anchor_scaled = scaler.transform(x_anchor)

        target_date = val_dates[h - 1]
        for j, tenor in enumerate(YIELD_TENORS):
            y_train = dt.loc[valid_idx, tenor].values
            model = ElasticNet(alpha=0.1, l1_ratio=0.7, max_iter=10000)
            model.fit(X_scaled, y_train)
            delta_pred = model.predict(x_anchor_scaled)[0]
            max_delta = hist_deltas[tenor].abs().quantile(0.95)
            delta_pred = np.clip(delta_pred, -max_delta * h, max_delta * h)
            predictions.loc[target_date, tenor] = last_known[tenor] + delta_pred

    return predictions


# ==============================================================================
# V3: PCA-delta Ridge
# ==============================================================================
def fit_delta_pca(delta_targets, train_end):
    all_deltas = []
    for h in range(1, N_FORECAST + 1):
        dt = delta_targets[h]
        all_deltas.append(dt.loc[dt.index <= train_end].values)
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(np.vstack(all_deltas))
    return pca


def predict_v3(yield_df, delta_targets, pca, feat_m1, iv_full, iv_min,
               model_type, train_end):
    train_dates = yield_df.index[yield_df.index <= train_end]
    last_known = yield_df.loc[train_dates[-1], YIELD_TENORS]
    val_dates = yield_df.index[(yield_df.index >= VAL_START) & (yield_df.index <= VAL_END)]

    predictions = pd.DataFrame(index=val_dates, columns=YIELD_TENORS, dtype=float)

    for h in range(1, len(val_dates) + 1):
        features_df = get_features_for_horizon(feat_m1, iv_full, iv_min, h, model_type)
        dt = delta_targets[h]
        valid_idx = dt.index.intersection(train_dates).intersection(features_df.dropna().index)

        X_train = features_df.loc[valid_idx].ffill(axis=1).fillna(0)
        delta_pca = pca.transform(dt.loc[valid_idx].values)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        x_anchor = features_df.loc[[train_dates[-1]]].ffill(axis=1).fillna(0)
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
        target_date = val_dates[h - 1]
        for j, tenor in enumerate(YIELD_TENORS):
            predictions.loc[target_date, tenor] = last_known[tenor] + delta_9[j]

    return predictions


# ==============================================================================
# ENSEMBLE WEIGHT SEARCH
# ==============================================================================
def find_best_weight(pred_v2, pred_v3, actual):
    best_w, best_rmse = 0.0, np.inf
    for w in np.arange(0.0, 1.05, 0.05):
        ens = w * pred_v2 + (1 - w) * pred_v3
        rmse = weighted_rmse(actual, ens)
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w
    return best_w, best_rmse


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
            ax.plot(history.index, history[tenor], color='#FF69B4', linewidth=1.5,
                    label='Actual')
            ext_d = [anchor_date] + list(pred.index)
            ext_v = [last_known[tenor]] + list(pred[tenor].values.astype(float))
            ax.plot(ext_d, ext_v, color='#2ECC71', linewidth=2, marker='s',
                    markersize=4, label=f'Forecast {name}')
            ax.set_ylabel(tenor, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            if i == 0:
                subtitle = 'M1 (PCA-delta Ridge)' if name == 'M1' else \
                    'M2 (70% ElasticNet + 30% PCA-Ridge)'
                ax.set_title(subtitle, fontsize=12, fontweight='bold')

    plt.suptitle('v2+3 Ensemble: Validation 2025-04..2025-09',
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
    print("  v2+3 ENSEMBLE VALIDATION")
    print("=" * 70)

    print("\n[1] Loading data...")
    yield_df, iv_df = load_data()
    print(f"    Yield: {yield_df.shape}, IV: {iv_df.shape}")

    print("\n[2] Preparing features...")
    feat_m1 = build_m1_features(yield_df)
    iv_full = build_iv_features_full(iv_df)
    iv_min = build_iv_features_minimal(iv_df)
    delta_targets = prepare_delta_targets(yield_df)
    pca = fit_delta_pca(delta_targets, TRAIN_END)
    print(f"    PCA explained variance: {pca.explained_variance_ratio_.round(4)}")

    # --- M1 ---
    print("\n[3] M1: v2 vs v3...")
    v2_m1 = predict_v2(yield_df, delta_targets, feat_m1, iv_full, iv_min, "M1", TRAIN_END)
    v3_m1 = predict_v3(yield_df, delta_targets, pca, feat_m1, iv_full, iv_min, "M1", TRAIN_END)

    actual = yield_df.loc[v2_m1.index, YIELD_TENORS].values.astype(float)
    w_m1, rmse_m1 = find_best_weight(v2_m1.values.astype(float),
                                      v3_m1.values.astype(float), actual)
    print(f"    v2 M1 RMSE: {weighted_rmse(actual, v2_m1.values.astype(float)):.4f}")
    print(f"    v3 M1 RMSE: {weighted_rmse(actual, v3_m1.values.astype(float)):.4f}")
    print(f"    Best w_v2={w_m1:.2f} -> ensemble RMSE: {rmse_m1:.4f}")

    pred_m1 = pd.DataFrame(
        w_m1 * v2_m1.values.astype(float) + (1 - w_m1) * v3_m1.values.astype(float),
        index=v2_m1.index, columns=YIELD_TENORS
    )

    # --- M2 ---
    print("\n[4] M2: v2 vs v3...")
    v2_m2 = predict_v2(yield_df, delta_targets, feat_m1, iv_full, iv_min, "M2", TRAIN_END)
    v3_m2 = predict_v3(yield_df, delta_targets, pca, feat_m1, iv_full, iv_min, "M2", TRAIN_END)

    w_m2, rmse_m2 = find_best_weight(v2_m2.values.astype(float),
                                      v3_m2.values.astype(float), actual)
    print(f"    v2 M2 RMSE: {weighted_rmse(actual, v2_m2.values.astype(float)):.4f}")
    print(f"    v3 M2 RMSE: {weighted_rmse(actual, v3_m2.values.astype(float)):.4f}")
    print(f"    Best w_v2={w_m2:.2f} -> ensemble RMSE: {rmse_m2:.4f}")

    pred_m2 = pd.DataFrame(
        w_m2 * v2_m2.values.astype(float) + (1 - w_m2) * v3_m2.values.astype(float),
        index=v2_m2.index, columns=YIELD_TENORS
    )

    # --- Results ---
    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n  M1: w_v2={w_m1:.2f}, w_v3={1-w_m1:.2f} -> RMSE={rmse_m1:.4f}")
    print(f"  M2: w_v2={w_m2:.2f}, w_v3={1-w_m2:.2f} -> RMSE={rmse_m2:.4f}")
    print(f"  RMSE_total = {0.5*rmse_m1 + 0.5*rmse_m2:.4f}")

    print(f"\n  {'Tenor':>5} | {'RMSE M1':>10} | {'RMSE M2':>10} | {'Better':>6}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
    m1_arr = pred_m1.values.astype(float)
    m2_arr = pred_m2.values.astype(float)
    for j, tenor in enumerate(YIELD_TENORS):
        r1 = np.sqrt(np.mean((actual[:, j] - m1_arr[:, j]) ** 2))
        r2 = np.sqrt(np.mean((actual[:, j] - m2_arr[:, j]) ** 2))
        print(f"  {tenor:>5} | {r1:>10.4f} | {r2:>10.4f} | {'M2' if r2 < r1 else 'M1':>6}")

    # --- Save ---
    output_dir = os.path.dirname(os.path.abspath(__file__))

    deltas_path = os.path.join(output_dir, "deltas.xlsx")
    actual_df = yield_df.loc[pred_m1.index, YIELD_TENORS]
    with pd.ExcelWriter(deltas_path, engine="openpyxl") as writer:
        (pred_m1 - actual_df.astype(float)).to_excel(writer, sheet_name="Delta_M1")
        (pred_m2 - actual_df.astype(float)).to_excel(writer, sheet_name="Delta_M2")
        pred_m1.to_excel(writer, sheet_name="Pred_M1")
        pred_m2.to_excel(writer, sheet_name="Pred_M2")
    print(f"\n  Saved: {deltas_path}")

    # --- Save weights for pipeline_final ---
    weights_path = os.path.join(output_dir, "ensemble_weights.txt")
    with open(weights_path, 'w') as f:
        f.write(f"w_v2_m1={w_m1:.2f}\n")
        f.write(f"w_v2_m2={w_m2:.2f}\n")
    print(f"  Weights: {weights_path}")

    print("\n[5] Plotting...")
    path = plot_results(yield_df, pred_m1, pred_m2, output_dir)
    print(f"  Plot: {path}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
