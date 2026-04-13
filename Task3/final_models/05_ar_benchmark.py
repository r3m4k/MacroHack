# 05_ar_benchmark.py
# Тест MPU-индексов против AR(p) бенчмарка.
#
# Логика:
#   Бенчмарк в 04_forecast_tests.py — константа (среднее IS).
#   Это слабый соперник. Здесь проверяем: бьёт ли MPU
#   авторегрессионную модель, которая использует только
#   историю самой целевой переменной?
#
#   AR(p): y(t+h) = φ_0 + φ_1·y(t) + ... + φ_p·y(t-p+1) + ε
#   MPU:   y(t+h) = α + β·MPU(t) + ε
#   ARX:   y(t+h) = φ_0 + φ_1·y(t) + β·MPU(t) + ε  ← добавляет ли MPU?
#
# Три сравнения на каждом горизонте h:
#   R²_oos(AR)  — AR против константы
#   R²_oos(MPU) — MPU против константы
#   R²_oos(ARX) — AR+MPU против константы
#
# Ключевой вопрос: R²_oos(ARX) > R²_oos(AR)?
# Если да — MPU добавляет информацию сверх авторегрессии.
#
# Результаты -> Task3/final_models/ar_benchmark/
# Запуск: python 05_ar_benchmark.py

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))
MPU_DIR  = BASE_DIR / 'mpu_models'
OUT_DIR  = BASE_DIR / 'ar_benchmark'
PLOT_DIR = OUT_DIR / 'plots'

from data_loading.case_2 import get_key_rate_dataframe

HORIZONS   = [1, 2, 3, 6, 12]
SPLIT_FRAC = 0.6
AR_ORDER   = 1          # p в AR(p); можно менять на 2, 3
INDEX_COLORS = {
    'MPU_rv_std': '#8E44AD',
    'MPU_decay':  '#E65100',
    'MPU_atm':    '#27AE60',
}


# ════════════════════════════════════════════════════════════
# Утилиты
# ════════════════════════════════════════════════════════════

def _save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {name}")


def _add_const(X):
    """Добавляет столбец единиц к матрице регрессоров любого размера."""
    X = np.atleast_2d(np.array(X, dtype=float))
    if X.ndim == 1 or X.shape[0] == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])


# ════════════════════════════════════════════════════════════
# Загрузка данных
# ════════════════════════════════════════════════════════════

def load_indices() -> dict[str, pd.Series]:
    files = {
        'MPU_rv_std': ('MPU_rv_std.csv', 'MPU_rv_std'),
        'MPU_decay':  ('MPU_decay.csv',  'MPU_decay_norm'),
        'MPU_atm':    ('MPU_atm.csv',    'MPU_atm_norm'),
    }
    out = {}
    print("Загрузка MPU-индексов:")
    for name, (fname, col) in files.items():
        p = MPU_DIR / fname
        if not p.exists():
            raise FileNotFoundError(
                f"{p} не найден. Запустите 03_mpu_models.py"
            )
        df = pd.read_csv(p, parse_dates=['Date'])
        s  = df.set_index('Date')[col].sort_index().dropna()
        s.index = pd.to_datetime(s.index)
        out[name] = s
        print(f"  {name:<14} N={len(s)}  "
              f"[{s.index.min().date()} — {s.index.max().date()}]")
    return out


def load_targets(key_rate_df) -> dict[str, pd.Series]:
    """Возвращает RV и инфляцию как pd.Series с DatetimeIndex."""
    # RV ключевой ставки
    kr = (key_rate_df[['Date','Key Rate']]
          .dropna(subset=['Date'])
          .rename(columns={'Key Rate':'Key_Rate'})
          .sort_values('Date')
          .set_index('Date'))
    rv = kr['Key_Rate'].diff().rolling(3, min_periods=3).std()
    rv.name = 'RV'
    rv.index = pd.to_datetime(rv.index)

    # Инфляция
    inf = (key_rate_df[['Date','Inflation']]
           .dropna(subset=['Date','Inflation'])
           .sort_values('Date')
           .set_index('Date')['Inflation'])
    inf.index = pd.to_datetime(inf.index)

    print(f"\nЦелевые переменные:")
    print(f"  RV:        N={rv.dropna().__len__()}  "
          f"mean={rv.mean():.3f}  max={rv.max():.3f}")
    print(f"  Инфляция:  N={len(inf)}  "
          f"mean={inf.mean():.2f}%  max={inf.max():.2f}%")
    return {'RV': rv, 'Inflation': inf}


# ════════════════════════════════════════════════════════════
# Объединение рядов
# ════════════════════════════════════════════════════════════

def merge_series(mpu: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Объединяет MPU и целевой ряд по ближайшей дате (±15 дней)."""
    mpu_df = pd.DataFrame({
        'Date': pd.to_datetime(mpu.index),
        'MPU':  mpu.values,
    }).dropna()
    tgt_df = pd.DataFrame({
        'Date':   pd.to_datetime(target.index),
        'Target': target.values,
    }).dropna()
    m = pd.merge_asof(
        mpu_df.sort_values('Date'),
        tgt_df.sort_values('Date'),
        on='Date',
        tolerance=pd.Timedelta('15D'),
        direction='nearest'
    ).dropna()
    return m.reset_index(drop=True)


# ════════════════════════════════════════════════════════════
# Ядро OOS теста
# ════════════════════════════════════════════════════════════

def oos_r2(y_oos: np.ndarray,
            y_pred: np.ndarray,
            bench: np.ndarray) -> float:
    """R²_oos = 1 - MSE_model / MSE_bench."""
    mse_m = float(np.mean((y_oos - y_pred) ** 2))
    mse_b = float(np.mean((y_oos - bench)  ** 2))
    return 1.0 - mse_m / max(mse_b, 1e-10)


def run_oos(y_is, X_is, y_oos, X_oos, bench):
    """
    Оценивает OLS на IS, прогнозирует на OOS, возвращает R²_oos.
    X_is и X_oos уже содержат константу.
    """
    try:
        model  = sm.OLS(y_is, X_is).fit(
            cov_type='HAC', cov_kwds={'maxlags': 3}
        )
        y_pred = model.predict(X_oos)
        r2     = oos_r2(y_oos, y_pred, bench)
        beta   = model.params[1] if model.params.shape[0] > 1 else np.nan
        pval   = model.pvalues[1] if model.pvalues.shape[0] > 1 else np.nan
        return r2, float(beta), float(pval)
    except Exception:
        return np.nan, np.nan, np.nan


# ════════════════════════════════════════════════════════════
# Основная функция теста
# ════════════════════════════════════════════════════════════

def ar_benchmark_test(mpu_name: str,
                       mpu: pd.Series,
                       target: pd.Series,
                       target_name: str,
                       ar_order: int = AR_ORDER) -> pd.DataFrame:
    """
    Сравнивает три модели на каждом горизонте h:

    CONST:  ŷ = ȳ_IS                              ← наивный бенчмарк
    AR(p):  ŷ = φ_0 + φ_1·y(t) + ... + φ_p·y(t-p+1)
    MPU:    ŷ = α + β·MPU(t)
    ARX:    ŷ = φ_0 + φ_1·y(t) + β·MPU(t)

    Для AR и ARX используем AR_ORDER лагов целевой переменной.
    Лаги берём прямо из merged DataFrame через shift.

    Вопросы:
        1. R²_oos(AR)  > 0?  →  авторегрессия бьёт константу?
        2. R²_oos(MPU) > 0?  →  MPU бьёт константу?
        3. R²_oos(ARX) > R²_oos(AR)?  →  MPU добавляет сверх AR?
    """
    merged  = merge_series(mpu, target)
    n       = len(merged)
    n_split = int(n * SPLIT_FRAC)
    records = []

    for h in HORIZONS:
        # Целевая переменная: y(t+h)
        fwd = merged['Target'].shift(-h)

        # Лаги целевой переменной для AR
        lags = {}
        for p in range(1, ar_order + 1):
            lags[f'lag_{p}'] = merged['Target'].shift(p - 1)

        # Маска: все компоненты не NaN
        lag_mask = pd.concat(lags, axis=1).notna().all(axis=1)
        mask     = fwd.notna() & merged['MPU'].notna() & lag_mask

        idx  = merged.index[mask]
        is_  = idx[idx < n_split]
        oos_ = idx[idx >= n_split]

        if len(is_) < max(12, ar_order + 3) or len(oos_) < 3:
            continue

        y_is  = fwd.loc[is_].values.astype(float)
        y_oos = fwd.loc[oos_].values.astype(float)

        # Регрессоры
        mpu_is  = merged.loc[is_,  'MPU'].values.astype(float)
        mpu_oos = merged.loc[oos_, 'MPU'].values.astype(float)

        ar_is  = np.column_stack(
            [lags[f'lag_{p}'].loc[is_].values.astype(float)
             for p in range(1, ar_order + 1)]
        )
        ar_oos = np.column_stack(
            [lags[f'lag_{p}'].loc[oos_].values.astype(float)
             for p in range(1, ar_order + 1)]
        )

        bench = np.full_like(y_oos, y_is.mean())

        # ── CONST ─────────────────────────────────────────
        r2_const = 0.0   # по определению

        # ── AR(p) ─────────────────────────────────────────
        r2_ar, _, _ = run_oos(
            y_is,  _add_const(ar_is),
            y_oos, _add_const(ar_oos),
            bench
        )

        # ── MPU ───────────────────────────────────────────
        r2_mpu, beta_mpu, pval_mpu = run_oos(
            y_is,  _add_const(mpu_is.reshape(-1, 1)),
            y_oos, _add_const(mpu_oos.reshape(-1, 1)),
            bench
        )

        # ── ARX: AR + MPU ──────────────────────────────────
        arx_is  = np.column_stack([ar_is,  mpu_is])
        arx_oos = np.column_stack([ar_oos, mpu_oos])
        r2_arx, beta_arx, pval_arx = run_oos(
            y_is,  _add_const(arx_is),
            y_oos, _add_const(arx_oos),
            bench
        )

        # MPU добавляет сверх AR?
        gain = (r2_arx - r2_ar) if not np.isnan(r2_arx) else np.nan

        records.append(dict(
            mpu          = mpu_name,
            target       = target_name,
            h            = h,
            N_is         = len(is_),
            N_oos        = len(oos_),
            R2_const     = 0.0,
            R2_ar        = round(r2_ar,  4),
            R2_mpu       = round(r2_mpu, 4),
            R2_arx       = round(r2_arx, 4),
            gain_vs_ar   = round(gain,   4),
            beta_mpu     = round(beta_mpu, 4),
            pval_mpu     = round(pval_mpu, 4),
            beta_arx_mpu = round(beta_arx, 4),
            pval_arx_mpu = round(pval_arx, 4),
            mpu_beats_const = r2_mpu > 0,
            mpu_beats_ar    = gain > 0 if not np.isnan(gain) else False,
        ))

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════
# Вывод результатов
# ════════════════════════════════════════════════════════════

def print_results(df: pd.DataFrame) -> None:
    """Читаемый вывод результатов для одного MPU и одной целевой."""
    mpu_name = df['mpu'].iloc[0]
    tgt_name = df['target'].iloc[0]

    print(f"\n  {mpu_name} → {tgt_name}:")
    print(f"  {'h':>3}  {'CONST':>7}  {'AR':>7}  {'MPU':>7}  "
          f"{'ARX':>7}  {'gain':>7}  "
          f"{'p_mpu':>6}  {'p_arx':>6}  beats_AR")
    print("  " + "─" * 72)

    for _, row in df.iterrows():
        gain_mark = '✅' if row['mpu_beats_ar'] else '✗ '
        p_mpu_str = f"{row['pval_mpu']:.3f}"
        p_arx_str = f"{row['pval_arx_mpu']:.3f}"
        sig_mpu   = '**' if row['pval_mpu']     < 0.05 else '  '
        sig_arx   = '**' if row['pval_arx_mpu'] < 0.05 else '  '

        print(f"  {row['h']:>3}M "
              f" {row['R2_const']:>+7.3f}"
              f"  {row['R2_ar']:>+7.3f}"
              f"  {row['R2_mpu']:>+7.3f}"
              f"  {row['R2_arx']:>+7.3f}"
              f"  {row['gain_vs_ar']:>+7.3f}"
              f"  {p_mpu_str}{sig_mpu}"
              f"  {p_arx_str}{sig_arx}"
              f"  {gain_mark}")


def print_summary(all_results: list[pd.DataFrame]) -> None:
    """Сводная таблица: бьёт ли MPU AR-бенчмарк."""
    df = pd.concat(all_results, ignore_index=True)

    print(f"\n{'='*65}")
    print("СВОДКА: MPU бьёт AR-бенчмарк? (gain_vs_ar > 0)")
    print(f"{'='*65}")
    print(f"  {'MPU':<14} {'Цель':<12} "
          + "  ".join(f"h={h:2d}M" for h in HORIZONS))
    print("  " + "─" * 60)

    for (mpu, tgt), grp in df.groupby(['mpu', 'target']):
        row_str = f"  {mpu:<14} {tgt:<12} "
        for h in HORIZONS:
            sub = grp[grp['h'] == h]
            if sub.empty:
                row_str += "  n/a "
                continue
            gain = sub['gain_vs_ar'].values[0]
            if np.isnan(gain):
                row_str += "  n/a "
            elif gain > 0:
                row_str += f" +{gain:.3f}"
            else:
                row_str += f"  {gain:.3f}"
        print(row_str)


# ════════════════════════════════════════════════════════════
# Графики
# ════════════════════════════════════════════════════════════

def plot_r2_comparison(all_results: list[pd.DataFrame],
                        target_name: str) -> None:
    """
    Для каждого MPU: линии R²_oos по горизонтам для CONST/AR/MPU/ARX.
    """
    sub = pd.concat(all_results)
    sub = sub[sub['target'] == target_name]
    if sub.empty:
        return

    mpu_names = sub['mpu'].unique()
    n = len(mpu_names)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f'AR-бенчмарк: R²_oos по моделям | Цель: {target_name}',
        fontweight='bold', fontsize=13
    )

    styles = {
        'CONST': ('grey',   ':',  'Константа',  1.2),
        'AR':    ('black',  '--', f'AR({AR_ORDER})', 2.0),
        'MPU':   (None,     '-',  'MPU',         2.5),
        'ARX':   (None,     '-.', f'AR+MPU',     2.0),
    }

    for ax, mpu_name in zip(axes, mpu_names):
        color = INDEX_COLORS.get(mpu_name, '#333')
        grp   = sub[sub['mpu'] == mpu_name].sort_values('h')

        for key, (clr, ls, label, lw) in styles.items():
            col = f'R2_{key.lower()}'
            if col not in grp.columns:
                continue
            c = clr if clr else color
            ax.plot(grp['h'], grp[col],
                    color=c, ls=ls, lw=lw,
                    marker='o', markersize=6,
                    label=label)
            for _, row in grp.iterrows():
                v = row[col]
                if not np.isnan(v):
                    ax.annotate(f'{v:+.3f}',
                                (row['h'], v),
                                textcoords='offset points',
                                xytext=(4, 4), fontsize=7,
                                color=c)

        # Заливка: зона где ARX > AR (MPU добавляет ценность)
        ar_vals  = grp['R2_ar'].values
        arx_vals = grp['R2_arx'].values
        hs       = grp['h'].values
        ax.fill_between(hs, ar_vals, arx_vals,
                        where=(arx_vals > ar_vals),
                        alpha=0.12, color=color,
                        label='Прирост от MPU')

        ax.axhline(0, color='black', lw=1, ls='-', alpha=0.4)
        ax.set_title(mpu_name, fontweight='bold')
        ax.set_xlabel('Горизонт h (месяцев)')
        ax.set_ylabel('R²_oos')
        ax.set_xticks(HORIZONS)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    fname = f'ar_comparison_{target_name.lower()}.png'
    _save(fig, fname)


def plot_gain_heatmap(all_results: list[pd.DataFrame]) -> None:
    """
    Тепловая карта gain_vs_ar = R²_oos(ARX) - R²_oos(AR).
    Строки = (MPU, цель), столбцы = горизонты.
    """
    df  = pd.concat(all_results, ignore_index=True)
    df['row_label'] = df['mpu'] + ' → ' + df['target']
    labels  = df['row_label'].unique()
    n_rows  = len(labels)
    n_cols  = len(HORIZONS)

    data = np.full((n_rows, n_cols), np.nan)
    for i, label in enumerate(labels):
        for j, h in enumerate(HORIZONS):
            sub = df[(df['row_label'] == label) & (df['h'] == h)]
            if not sub.empty:
                data[i, j] = sub['gain_vs_ar'].values[0]

    vmax = max(np.nanmax(np.abs(data)), 0.01)
    fig, ax = plt.subplots(figsize=(10, max(4, n_rows)))
    im = ax.imshow(data, cmap='RdYlGn',
                   vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax,
                 label='gain = R²_oos(ARX) − R²_oos(AR)')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f'h={h}M' for h in HORIZONS])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if not np.isnan(v):
                c = 'white' if abs(v) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{v:+.3f}',
                        ha='center', va='center',
                        fontsize=9, fontweight='bold', color=c)

    ax.set_title(
        f'Прирост R²_oos от добавления MPU к AR({AR_ORDER})\n'
        f'Зелёный = MPU добавляет ценность сверх авторегрессии',
        fontweight='bold'
    )
    plt.tight_layout()
    _save(fig, 'gain_heatmap.png')


def plot_ar_vs_mpu_scatter(all_results: list[pd.DataFrame]) -> None:
    """
    Scatter: R²_oos(AR) vs R²_oos(MPU) для всех горизонтов.
    Точки выше диагонали → MPU лучше AR.
    """
    df = pd.concat(all_results, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('R²_oos(AR) vs R²_oos(MPU) по всем горизонтам',
                 fontweight='bold')

    for ax, tgt in zip(axes, ['RV', 'Inflation']):
        sub = df[df['target'] == tgt]
        if sub.empty:
            continue
        for mpu_name in sub['mpu'].unique():
            grp   = sub[sub['mpu'] == mpu_name]
            color = INDEX_COLORS.get(mpu_name, '#333')
            ax.scatter(grp['R2_ar'], grp['R2_mpu'],
                       color=color, s=80, zorder=5,
                       label=mpu_name, alpha=0.8)
            for _, row in grp.iterrows():
                ax.annotate(f"h={row['h']}",
                            (row['R2_ar'], row['R2_mpu']),
                            textcoords='offset points',
                            xytext=(5, 3), fontsize=7,
                            color=color)

        # Диагональ: MPU = AR
        lims = [min(sub['R2_ar'].min(), sub['R2_mpu'].min()) - 0.1,
                max(sub['R2_ar'].max(), sub['R2_mpu'].max()) + 0.1]
        ax.plot(lims, lims, color='black', lw=1.5, ls='--',
                label='MPU = AR', alpha=0.6)
        ax.axhline(0, color='grey', lw=0.8, ls=':')
        ax.axvline(0, color='grey', lw=0.8, ls=':')

        ax.fill_between(lims, lims, [lims[1]]*2,
                        alpha=0.04, color='green')
        ax.text(lims[0]+0.02, lims[1]-0.05,
                'MPU лучше AR', fontsize=8,
                color='green', alpha=0.7)

        ax.set_xlabel(f'R²_oos  AR({AR_ORDER})')
        ax.set_ylabel('R²_oos  MPU')
        ax.set_title(f'Цель: {tgt}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    _save(fig, 'ar_vs_mpu_scatter.png')


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"05 | AR-БЕНЧМАРК  (AR_ORDER={AR_ORDER}, split={SPLIT_FRAC:.0%})")
    print("=" * 65)

    key_rate_df = get_key_rate_dataframe()
    indices     = load_indices()
    targets     = load_targets(key_rate_df)

    all_results = []

    for tgt_name, target in targets.items():
        print(f"\n{'─'*65}")
        print(f"Цель: {tgt_name}")
        print(f"{'─'*65}")
        print(f"  {'h':>3}  {'CONST':>7}  {'AR':>7}  {'MPU':>7}  "
              f"{'ARX':>7}  {'gain':>7}  "
              f"{'p_mpu':>6}  {'p_arx':>6}  beats_AR")

        for mpu_name, mpu in indices.items():
            res = ar_benchmark_test(
                mpu_name, mpu,
                target, tgt_name,
                ar_order=AR_ORDER
            )
            if res.empty:
                print(f"  {mpu_name}: нет данных")
                continue

            print_results(res)
            all_results.append(res)

            # Сохраняем детальные результаты
            fname = f'ar_{mpu_name}_{tgt_name}.csv'
            res.to_csv(OUT_DIR / fname, index=False)

    if not all_results:
        print("Нет результатов для сохранения.")
        return

    # Сводная таблица
    print_summary(all_results)
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUT_DIR / 'ar_benchmark_all.csv', index=False)
    print(f"\nТаблицы сохранены в {OUT_DIR}")

    # Графики
    print("\nГенерация графиков...")
    for tgt_name in targets:
        plot_r2_comparison(all_results, tgt_name)
    plot_gain_heatmap(all_results)
    plot_ar_vs_mpu_scatter(all_results)

    # Итоговый вывод
    print(f"\n{'='*65}")
    print("ИТОГ: когда MPU добавляет ценность сверх AR?")
    print(f"{'='*65}")
    df = combined.copy()
    wins = df[df['mpu_beats_ar'] == True]
    if wins.empty:
        print("  MPU не превосходит AR ни на одном горизонте.")
    else:
        for _, row in wins.iterrows():
            print(f"  ✅ {row['mpu']:<14} → {row['target']:<12} "
                  f"h={row['h']:2d}M  "
                  f"gain={row['gain_vs_ar']:+.3f}  "
                  f"(R²_ar={row['R2_ar']:+.3f} → "
                  f"R²_arx={row['R2_arx']:+.3f})")

    print(f"\nГотово → {OUT_DIR.resolve()}")


if __name__ == '__main__':
    main()
