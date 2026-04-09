"""
================================================================================
ОТБОР ФИЧЕЙ ЧЕРЕЗ L1-РЕГУЛЯРИЗАЦИЮ (Lasso) — M1 и M2
================================================================================
Вход:  обучающая выборка (yield curve + IV surface)
Выход: списки отобранных фичей для каждого тенора, отдельно для M1 и M2
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from data_loading.problem_1 import get_IV_train_dataframe, get_curve_train_dataframe


target_columns = ['O/N', '1W', '2W', '1M', '2M', '3M', '6M', '1Y', '2Y']
iv_tenors = ['1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y',
             '5Y', '6Y', '7Y', '8Y', '9Y', '10Y']

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        ЗАГЛУШКА — ЗАМЕНИ НА СВОИ ДАННЫЕ                   ║
# ╠════════════════════════════════════════════════════════════════════════════╣
# ║  yield_df: DataFrame, index=даты (monthly), columns=target_columns       ║
# ║  iv_df:    DataFrame, index=даты (monthly), columns=iv_tenors            ║
# ║                                                                           ║
# ║  Замени блок ниже на:                                                     ║
# ║    yield_df = pd.read_excel('yield_curve_filled.xlsx',                    ║
# ║                              index_col='Month', parse_dates=True)         ║
# ║    iv_df = pd.read_excel('iv_surface.xlsx',                               ║
# ║                           index_col='Month', parse_dates=True)            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

np.random.seed(42)
n = 78
dates = pd.date_range('2019-03', periods=n, freq='MS')
base_yield = np.linspace(7, 9, len(target_columns))
yield_df = pd.DataFrame(
    np.tile(base_yield, (n, 1)) + np.cumsum(np.random.randn(n, len(target_columns)) * 0.15, axis=0),
    index=dates, columns=target_columns
)
base_iv = np.linspace(12, 18, len(iv_tenors))
iv_df = pd.DataFrame(
    np.tile(base_iv, (n, 1)) + np.cumsum(np.random.randn(n, len(iv_tenors)) * 0.3, axis=0),
    index=dates, columns=iv_tenors
)

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        КОНЕЦ ЗАГЛУШКИ                                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝


# ==============================================================================
# 1. FEATURE ENGINEERING
# ==============================================================================

def make_yield_features(df):
    """Фичи из кривой доходности (для M1 и M2)."""
    features = pd.DataFrame(index=df.index)

    # Лаги
    for lag in [1, 2, 3]:
        shifted = df[target_columns].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in target_columns]
        features = pd.concat([features, shifted], axis=1)

    # Дельты
    for d in [1, 3]:
        delta = df[target_columns].diff(d)
        delta.columns = [f"{c}_delta{d}" for c in target_columns]
        features = pd.concat([features, delta], axis=1)

    # Спреды
    features['spread_2Y_ON'] = df['2Y'] - df['O/N']
    features['spread_1Y_ON'] = df['1Y'] - df['O/N']
    features['spread_2Y_1Y'] = df['2Y'] - df['1Y']
    features['spread_6M_3M'] = df['6M'] - df['3M']
    features['spread_3M_1M'] = df['3M'] - df['1M']

    # Скользящие средние
    for w in [3, 6]:
        ma = df[target_columns].rolling(w).mean()
        ma.columns = [f"{c}_ma{w}" for c in target_columns]
        features = pd.concat([features, ma], axis=1)

    return features


def make_iv_features(iv_df):
    """Фичи из поверхности IV (только для M2)."""
    features = pd.DataFrame(index=iv_df.index)

    # Лагированные IV (shift=1 для предотвращения look-ahead)
    iv_lag1 = iv_df.shift(1)
    iv_lag1.columns = [f"iv_{c}_lag1" for c in iv_df.columns]
    features = pd.concat([features, iv_lag1], axis=1)

    # Дельты IV
    iv_delta = iv_df.diff(1).shift(1)
    iv_delta.columns = [f"iv_{c}_delta1" for c in iv_df.columns]
    features = pd.concat([features, iv_delta], axis=1)

    # Спреды термструктуры IV
    features['iv_spread_10Y_1M'] = iv_df['10Y'].shift(1) - iv_df['1M'].shift(1)
    features['iv_spread_5Y_1Y'] = iv_df['5Y'].shift(1) - iv_df['1Y'].shift(1)
    features['iv_spread_2Y_6M'] = iv_df['2Y'].shift(1) - iv_df['6M'].shift(1)

    # Скользящие средние IV
    iv_ma3 = iv_df.rolling(3).mean().shift(1)
    iv_ma3.columns = [f"iv_{c}_ma3" for c in iv_df.columns]
    features = pd.concat([features, iv_ma3], axis=1)

    # Vol-of-vol
    iv_std3 = iv_df.rolling(3).std().shift(1)
    features['iv_vol_of_vol_short'] = iv_std3.mean(axis=1)
    iv_std6 = iv_df.rolling(6).std().shift(1)
    features['iv_vol_of_vol_long'] = iv_std6.mean(axis=1)

    return features


# ==============================================================================
# 2. СБОРКА ФИЧЕЙ
# ==============================================================================
X_yield = make_yield_features(yield_df)
X_iv = make_iv_features(iv_df)

# M1: только yield-фичи
X_m1 = X_yield.copy()

# M2: yield-фичи + IV-фичи
X_m2 = pd.concat([X_yield, X_iv], axis=1)

# Убираем NaN
common_idx_m1 = X_m1.dropna().index.intersection(yield_df.dropna().index)
common_idx_m2 = X_m2.dropna().index.intersection(yield_df.dropna().index)

X_m1 = X_m1.loc[common_idx_m1]
y_m1 = yield_df.loc[common_idx_m1]

X_m2 = X_m2.loc[common_idx_m2]
y_m2 = yield_df.loc[common_idx_m2]

print(f"M1: {X_m1.shape[0]} наблюдений, {X_m1.shape[1]} фичей")
print(f"M2: {X_m2.shape[0]} наблюдений, {X_m2.shape[1]} фичей")
print(f"    из них IV-фичей: {X_iv.shape[1]}")

# ==============================================================================
# 3. L1-ОТБОР
# ==============================================================================

def lasso_select(X, y, model_name, target_columns):
    """
    Для каждого тенора:
    1. LassoCV подбирает оптимальный alpha через 5-fold CV
    2. Фичи с ненулевым коэффициентом → отобраны
    Возвращает dict: {тенор: {'features': [...], 'coefs': [...], 'alpha': float}}
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = X.columns.tolist()

    results = {}

    print(f"\n{'='*70}")
    print(f"  ОТБОР ФИЧЕЙ — {model_name}")
    print(f"{'='*70}")
    print(f"{'Тенор':>5} | {'Alpha':>10} | {'Отобрано':>10} | {'Из':>5} | Топ-3 фичи")
    print(f"{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*5}-+-{'-'*35}")

    for tenor in target_columns:
        target = y[tenor].values

        lasso_cv = LassoCV(
            alphas=np.logspace(-5, 1, 120),
            cv=5,
            max_iter=20000,
        )
        lasso_cv.fit(X_scaled, target)

        coefs = lasso_cv.coef_
        selected_mask = np.abs(coefs) > 1e-8
        selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
        selected_coefs = coefs[selected_mask]

        # Топ-3 по абсолютному значению
        if len(selected_features) > 0:
            top_idx = np.argsort(np.abs(selected_coefs))[::-1][:3]
            top3 = [selected_features[i] for i in top_idx]
        else:
            top3 = ['—']

        results[tenor] = {
            'alpha': lasso_cv.alpha_,
            'features': selected_features,
            'coefs': selected_coefs,
            'all_coefs': coefs,
            'n_selected': len(selected_features),
        }

        print(f"{tenor:>5} | {lasso_cv.alpha_:>10.5f} | "
              f"{len(selected_features):>10} | {len(feature_names):>5} | "
              f"{', '.join(top3)}")

    return results


results_m1 = lasso_select(X_m1, y_m1, "M1 (без IV)", target_columns)
results_m2 = lasso_select(X_m2, y_m2, "M2 (с IV)", target_columns)

# ==============================================================================
# 4. АНАЛИЗ: ЧТО ДОБАВИЛА IV
# ==============================================================================
print(f"\n{'='*70}")
print(f"  АНАЛИЗ: ЧТО ДОБАВИЛА IV в M2")
print(f"{'='*70}")

iv_feature_prefix = 'iv_'

for tenor in target_columns:
    m1_set = set(results_m1[tenor]['features'])
    m2_set = set(results_m2[tenor]['features'])

    m2_iv_features = [f for f in results_m2[tenor]['features'] if f.startswith(iv_feature_prefix)]
    m2_yield_features = [f for f in results_m2[tenor]['features'] if not f.startswith(iv_feature_prefix)]

    # Какие yield-фичи были в M1, но выпали в M2 (вытеснены IV)
    dropped = m1_set - set(m2_yield_features)
    # Какие yield-фичи добавились в M2
    added_yield = set(m2_yield_features) - m1_set

    print(f"\n  {tenor}:")
    print(f"    M1: {results_m1[tenor]['n_selected']} фичей")
    print(f"    M2: {results_m2[tenor]['n_selected']} фичей "
          f"({len(m2_yield_features)} yield + {len(m2_iv_features)} IV)")
    if m2_iv_features:
        print(f"    IV-фичи: {', '.join(m2_iv_features[:5])}"
              f"{'...' if len(m2_iv_features) > 5 else ''}")
    if dropped:
        print(f"    Вытеснены из M1: {', '.join(sorted(dropped)[:3])}")

# ==============================================================================
# 5. ЭКСПОРТ ОТОБРАННЫХ ФИЧЕЙ
# ==============================================================================

# Сохраняем как словарь — можно использовать в LightGBM
selected_features_m1 = {t: results_m1[t]['features'] for t in target_columns}
selected_features_m2 = {t: results_m2[t]['features'] for t in target_columns}

# Сохраняем в CSV для удобства
rows = []
for tenor in target_columns:
    for f in results_m1[tenor]['features']:
        idx = results_m1[tenor]['features'].index(f)
        rows.append({
            'model': 'M1',
            'tenor': tenor,
            'feature': f,
            'coef': results_m1[tenor]['coefs'][idx],
        })
    for f in results_m2[tenor]['features']:
        idx = results_m2[tenor]['features'].index(f)
        rows.append({
            'model': 'M2',
            'tenor': tenor,
            'feature': f,
            'coef': results_m2[tenor]['coefs'][idx],
        })

export_df = pd.DataFrame(rows)
export_df.to_csv('selected_features.csv', index=False)
print(f"\nОтобранные фичи сохранены в selected_features.csv")

# ==============================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Отбор фичей L1 (Lasso) — M1 vs M2', fontsize=16, fontweight='bold')

# --- График 1: Количество фичей M1 vs M2 ---
ax = axes[0, 0]
x_pos = np.arange(len(target_columns))
width = 0.35

n_m1 = [results_m1[t]['n_selected'] for t in target_columns]
n_m2 = [results_m2[t]['n_selected'] for t in target_columns]
n_m2_iv = [len([f for f in results_m2[t]['features'] if f.startswith('iv_')]) for t in target_columns]

bars1 = ax.bar(x_pos - width/2, n_m1, width, label='M1 (без IV)', color='#3498db', edgecolor='white')
bars2 = ax.bar(x_pos + width/2, n_m2, width, label='M2 (всего)', color='#e74c3c', edgecolor='white')
# IV-часть внутри M2
ax.bar(x_pos + width/2, n_m2_iv, width, label='M2 (IV-фичи)', color='#f39c12', edgecolor='white')

ax.set_xticks(x_pos)
ax.set_xticklabels(target_columns)
ax.set_ylabel('Количество отобранных фичей')
ax.set_title('M1 vs M2: сколько фичей отобрано')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar, n in zip(bars1, n_m1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(n),
            ha='center', fontweight='bold', fontsize=9, color='#3498db')
for bar, n in zip(bars2, n_m2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(n),
            ha='center', fontweight='bold', fontsize=9, color='#e74c3c')

# --- График 2: Top-10 фичей для O/N (M1) ---
ax = axes[0, 1]
coefs = results_m1['O/N']['all_coefs']
names = X_m1.columns.tolist()
abs_c = np.abs(coefs)
top_idx = np.argsort(abs_c)[-10:][::-1]
top_names = [names[i] for i in top_idx]
top_vals = coefs[top_idx]
colors = ['#e74c3c' if v > 0 else '#3498db' for v in top_vals]

ax.barh(range(len(top_names)), top_vals, color=colors, edgecolor='white')
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names, fontsize=9)
ax.set_title('M1 — Top-10 фичей для O/N')
ax.axvline(x=0, color='gray', linewidth=0.8)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# --- График 3: Top-10 фичей для O/N (M2) ---
ax = axes[1, 0]
coefs = results_m2['O/N']['all_coefs']
names = X_m2.columns.tolist()
abs_c = np.abs(coefs)
top_idx = np.argsort(abs_c)[-10:][::-1]
top_names = [names[i] for i in top_idx]
top_vals = coefs[top_idx]
colors_m2 = []
for v, nm in zip(top_vals, top_names):
    if nm.startswith('iv_'):
        colors_m2.append('#f39c12')  # IV-фичи — оранжевый
    elif v > 0:
        colors_m2.append('#e74c3c')
    else:
        colors_m2.append('#3498db')

ax.barh(range(len(top_names)), top_vals, color=colors_m2, edgecolor='white')
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names, fontsize=9)
ax.set_title('M2 — Top-10 фичей для O/N (оранжевый = IV)')
ax.axvline(x=0, color='gray', linewidth=0.8)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# --- График 4: Матрица отбора M2 (какие IV-фичи каким тенорам) ---
ax = axes[1, 1]
all_iv_features = sorted(set(
    f for t in target_columns
    for f in results_m2[t]['features']
    if f.startswith('iv_')
))

if len(all_iv_features) > 0:
    # Ограничим до 25 самых частых
    freq = {}
    for f in all_iv_features:
        freq[f] = sum(1 for t in target_columns if f in results_m2[t]['features'])
    top_iv = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)[:25]

    matrix = np.zeros((len(target_columns), len(top_iv)))
    for i, t in enumerate(target_columns):
        for j, f in enumerate(top_iv):
            if f in results_m2[t]['features']:
                idx = results_m2[t]['features'].index(f)
                matrix[i, j] = results_m2[t]['coefs'][idx]

    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest',
                   vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    ax.set_xticks(range(len(top_iv)))
    ax.set_xticklabels([f.replace('iv_', '') for f in top_iv], rotation=90, fontsize=7)
    ax.set_yticks(range(len(target_columns)))
    ax.set_yticklabels(target_columns, fontsize=10)
    ax.set_title('M2: коэффициенты IV-фичей по тенорам')
    plt.colorbar(im, ax=ax, shrink=0.8)
else:
    ax.text(0.5, 0.5, 'IV-фичи не отобраны\n(на синтетических данных)',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('M2: IV-фичи')

plt.tight_layout()
plt.savefig('lasso_m1_vs_m2.png', dpi=150, bbox_inches='tight')
plt.close()
print("График сохранён: lasso_m1_vs_m2.png")

# ==============================================================================
# 7. КАК ИСПОЛЬЗОВАТЬ РЕЗУЛЬТАТ В LightGBM
# ==============================================================================
print(f"\n{'='*70}")
print("  ГОТОВЫЕ СПИСКИ ФИЧЕЙ ДЛЯ LightGBM")
print(f"{'='*70}")

print("\n# --- Копируй в свой код ---")
print("selected_features_m1 = {")
for t in target_columns:
    print(f"    '{t}': {results_m1[t]['features']},")
print("}")

print("\nselected_features_m2 = {")
for t in target_columns:
    print(f"    '{t}': {results_m2[t]['features']},")
print("}")

print("""
# --- Использование в LightGBM ---
# for tenor in target_columns:
#     features = selected_features_m1[tenor]  # или m2
#     model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
#     model.fit(X_train[features], y_train[tenor])
#     pred = model.predict(X_test[features])
""")
