# 04_forecast_tests.py
# Тестирование прогнозной силы MPU_rv_std, MPU_decay, MPU_atm
# для предсказывания:
#   (A) Реализованной волатильности (RV) ключевой ставки
#       → результаты в Task3/final_models/rv_predict/
#   (B) Инфляции (ИПЦ г/г)
#       → результаты в Task3/final_models/inflation_predict/
#
# Читает CSV из mpu_models/.
# Запуск: python 04_forecast_tests.py

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
RV_DIR   = BASE_DIR / 'rv_predict'
INF_DIR  = BASE_DIR / 'inflation_predict'

from data_loading.case_2 import get_key_rate_dataframe

HORIZONS    = [1, 2, 3, 6, 12]
SPLIT_FRAC  = 0.6
INDEX_COLORS = {
    'MPU_rv_std': '#8E44AD',
    'MPU_decay':  '#E65100',
    'MPU_atm':    '#27AE60',
}


# ════════════════════════════════════════════════════════════
# Загрузка данных
# ════════════════════════════════════════════════════════════

def load_indices():
    files = {
        'MPU_rv_std': ('MPU_rv_std.csv', 'MPU_rv_std'),
        'MPU_decay':  ('MPU_decay.csv',  'MPU_decay_norm'),
        'MPU_atm':    ('MPU_atm.csv',    'MPU_atm_norm'),
    }
    out = {}
    for name, (fname, col) in files.items():
        p  = MPU_DIR / fname
        if not p.exists():
            raise FileNotFoundError(
                f"{p} не найден. Запустите 03_mpu_models.py"
            )
        df = pd.read_csv(p, parse_dates=['Date'])
        s  = df.set_index('Date')[col].sort_index().dropna()
        s.index = pd.to_datetime(s.index)
        out[name] = s
        print(f"  {name:<14} N={len(s)}")
    return out


def compute_rv(key_rate_df, window=3):
    df = (key_rate_df[['Date','Key Rate']]
          .dropna(subset=['Date'])
          .rename(columns={'Key Rate':'Key_Rate'})
          .sort_values('Date').set_index('Date'))
    rv = df['Key_Rate'].diff().rolling(window, min_periods=window).std()
    rv.name = 'RV'
    return rv


def load_inflation(key_rate_df):
    df = (key_rate_df[['Date','Key Rate','Inflation']]
          .rename(columns={'Key Rate':'Key_Rate'})
          .dropna(subset=['Date','Inflation'])
          .sort_values('Date').reset_index(drop=True))
    return df


# ════════════════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════════════════

def _add_const(x):
    x = np.atleast_1d(np.array(x, dtype=float)).reshape(-1)
    return np.column_stack([np.ones(len(x)), x])


def _merge(mpu: pd.Series, target: pd.Series) -> pd.DataFrame:
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


def oos_test(mpu: pd.Series, target: pd.Series, name: str) -> pd.DataFrame:
    """
    OOS тест: R2_oos = 1 - MSE_model / MSE_benchmark.
    Модель: Target(t+h) = α + β·MPU(t).
    Бенчмарк: среднее Target по IS.
    """
    merged  = _merge(mpu, target)
    n_split = int(len(merged) * SPLIT_FRAC)
    records = []

    for h in HORIZONS:
        fwd  = merged['Target'].shift(-h)
        mask = fwd.notna() & merged['MPU'].notna()
        idx  = merged.index[mask]
        is_  = idx[idx < n_split]
        oos_ = idx[idx >= n_split]

        if len(is_) < 8 or len(oos_) < 3:
            continue

        y_is  = fwd.loc[is_].values.astype(float)
        x_is  = merged.loc[is_,  'MPU'].values.astype(float)
        y_oos = fwd.loc[oos_].values.astype(float)
        x_oos = merged.loc[oos_, 'MPU'].values.astype(float)

        m_fit   = sm.OLS(y_is, _add_const(x_is)).fit(
            cov_type='HAC', cov_kwds={'maxlags': h})
        y_pred  = m_fit.predict(_add_const(x_oos))
        bench   = np.full_like(y_oos, y_is.mean())

        mse_m = float(np.mean((y_oos - y_pred)**2))
        mse_b = float(np.mean((y_oos - bench)**2))
        r2    = 1. - mse_m / max(mse_b, 1e-10)

        records.append(dict(
            name=name, h=h,
            N_is=len(is_), N_oos=len(oos_),
            beta=round(float(m_fit.params[1]), 4),
            pval=round(float(m_fit.pvalues[1]), 4),
            R2_oos=round(r2, 4),
            beats_bench=r2 > 0,
        ))

    return pd.DataFrame(records)


def compare(indices, target, label):
    """Прогоняет все индексы и возвращает сводную таблицу."""
    all_res = {}
    for name, mpu in indices.items():
        r = oos_test(mpu, target, name)
        all_res[name] = r
        for _, row in r.iterrows():
            sig = '**' if row['pval'] < 0.05 else '  '
            print(f"    {name}  h={row['h']:2d}M  "
                  f"beta={row['beta']:+.3f}(p={row['pval']:.3f}){sig}  "
                  f"R2_oos={row['R2_oos']:+.3f}")

    frames = [r[['h','R2_oos']].rename(columns={'R2_oos': n}).set_index('h')
              for n, r in all_res.items()]
    summary = pd.concat(frames, axis=1)
    summary['best'] = summary.idxmax(axis=1)
    return summary, all_res


# ════════════════════════════════════════════════════════════
# Графики
# ════════════════════════════════════════════════════════════

def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_r2oos(summary, title, save_path):
    names = [c for c in summary.columns if c != 'best']
    hs    = list(summary.index)
    n_h   = len(hs)

    fig, axes = plt.subplots(1, n_h, figsize=(4*n_h, 5))
    if n_h == 1: axes = [axes]
    fig.suptitle(title, fontweight='bold', fontsize=12)

    for ax, h in zip(axes, hs):
        vals   = [summary.loc[h, n] for n in names]
        colors = [INDEX_COLORS.get(n, '#555') for n in names]
        bars   = ax.bar(range(len(names)), vals,
                        color=colors, alpha=0.85, edgecolor='white', width=0.6)
        ax.axhline(0, color='black', lw=1.5, ls='--')
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + .005*(1 if v>=0 else -1),
                        f'{v:+.3f}', ha='center',
                        va='bottom' if v>=0 else 'top',
                        fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.set_title(f'h = {h}M', fontweight='bold')
        if h == hs[0]: ax.set_ylabel('R2_oos')

    plt.tight_layout(); _save(fig, save_path)


def plot_oos_ts(all_res, indices, target, label, horizon, save_path):
    """Факт vs прогнозы (OOS период) для одного горизонта."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Факт
    ax.plot(target.index, target.values,
            color='black', lw=2.5, zorder=10, label=f'{label} (факт)')

    split_shown = False
    for name, mpu in indices.items():
        merged  = _merge(mpu, target)
        n_split = int(len(merged) * SPLIT_FRAC)
        fwd     = merged['Target'].shift(-horizon)
        mask    = fwd.notna() & merged['MPU'].notna()
        is_     = merged.index[mask & (merged.index < n_split)]
        oos_    = merged.index[mask & (merged.index >= n_split)]
        if len(is_) < 5 or len(oos_) < 2: continue

        m_fit  = sm.OLS(fwd.loc[is_].values.astype(float),
                        _add_const(merged.loc[is_,'MPU'].values.astype(float))).fit()
        y_pred = m_fit.predict(_add_const(merged.loc[oos_,'MPU'].values.astype(float)))
        d_oos  = merged.loc[oos_,'Date'].values
        r      = all_res[name]
        row    = r[r['h']==horizon]
        r2     = row['R2_oos'].values[0] if not row.empty else np.nan
        color  = INDEX_COLORS.get(name,'#333')

        ax.plot(d_oos, y_pred, color=color, lw=2, ls='--',
                label=f'{name}  R2={r2:+.3f}')

        if not split_shown:
            sd = merged.loc[oos_[0],'Date']
            ax.axvline(pd.Timestamp(sd), color='grey', lw=1.5, ls='-.',
                       label=f'Начало OOS: {pd.Timestamp(sd).date()}')
            split_shown = True

    ax.set_title(f'Прогноз {label}(t+{horizon}M): факт vs MPU-индексы',
                 fontweight='bold')
    ax.set(xlabel='Дата', ylabel=label)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout(); _save(fig, save_path)


def plot_r2oos_by_horizon(summary, title, save_path):
    """R2_oos как функция горизонта h."""
    names = [c for c in summary.columns if c != 'best']
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        s = summary[name].sort_index()
        ax.plot(s.index, s.values, color=INDEX_COLORS.get(name,'#555'),
                lw=2.5, marker='o', markersize=8, label=name)
        for h, v in s.items():
            ax.annotate(f'{v:+.3f}', (h, v),
                        textcoords='offset points', xytext=(5,5),
                        fontsize=8, color=INDEX_COLORS.get(name,'#555'))
    ax.axhline(0, color='black', lw=1.5, ls='--', label='Бенчмарк')
    ax.set(xlabel='Горизонт h (месяцев)', ylabel='R2_oos',
           title=title)
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); _save(fig, save_path)


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    RV_DIR.mkdir(parents=True, exist_ok=True)
    INF_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("04 | ТЕСТЫ ПРОГНОЗНОЙ СИЛЫ")
    print("=" * 55)

    key_rate_df = get_key_rate_dataframe()
    print("\nЗагрузка MPU-индексов:")
    indices = load_indices()

    # ── A. Прогнозирование RV ─────────────────────────────
    print("\n" + "─"*55)
    print("A. ПРОГНОЗИРОВАНИЕ RV КЛЮЧЕВОЙ СТАВКИ")
    print("─"*55)

    rv = compute_rv(key_rate_df, window=3)
    print(f"RV: mean={rv.mean():.3f}  max={rv.max():.3f}")

    rv_summary, rv_results = compare(indices, rv, 'RV')

    print("\nСводная таблица R2_oos (RV):")
    print(rv_summary.to_string())
    print("\nЛучший на каждом горизонте:")
    for h, row in rv_summary.iterrows():
        best = row['best']; val = row[best]
        print(f"  h={h:2d}M  {best}  {val:+.4f}")

    # Сохранение
    rv_summary.to_csv(RV_DIR / 'rv_r2oos_summary.csv')
    for name, res in rv_results.items():
        res.to_csv(RV_DIR / f'rv_{name}.csv', index=False)

    # Графики RV
    plot_r2oos(rv_summary,
               'Прогнозирование RV: R2_oos',
               RV_DIR / 'plots' / 'rv_r2oos_by_horizon_bars.png')
    plot_r2oos_by_horizon(rv_summary,
               'Прогнозирование RV: R2_oos по горизонтам',
               RV_DIR / 'plots' / 'rv_r2oos_by_horizon_lines.png')
    for h in [2, 3, 6]:
        plot_oos_ts(rv_results, indices, rv, 'RV', h,
                    RV_DIR / 'plots' / f'rv_forecast_h{h}.png')

    # ── B. Прогнозирование инфляции ───────────────────────
    print("\n" + "─"*55)
    print("B. ПРОГНОЗИРОВАНИЕ ИНФЛЯЦИИ")
    print("─"*55)

    inf_df = load_inflation(key_rate_df)
    inf    = inf_df.set_index('Date')['Inflation']
    inf.index = pd.to_datetime(inf.index)
    print(f"Инфляция: mean={inf.mean():.2f}%  max={inf.max():.2f}%")

    inf_summary, inf_results = compare(indices, inf, 'Inflation')

    print("\nСводная таблица R2_oos (Инфляция):")
    print(inf_summary.to_string())
    print("\nЛучший на каждом горизонте:")
    for h, row in inf_summary.iterrows():
        best = row['best']; val = row[best]
        print(f"  h={h:2d}M  {best}  {val:+.4f}")

    # Сохранение
    inf_summary.to_csv(INF_DIR / 'inflation_r2oos_summary.csv')
    for name, res in inf_results.items():
        res.to_csv(INF_DIR / f'inflation_{name}.csv', index=False)

    # Графики инфляции
    plot_r2oos(inf_summary,
               'Прогнозирование инфляции: R2_oos',
               INF_DIR / 'plots' / 'inf_r2oos_by_horizon_bars.png')
    plot_r2oos_by_horizon(inf_summary,
               'Прогнозирование инфляции: R2_oos по горизонтам',
               INF_DIR / 'plots' / 'inf_r2oos_by_horizon_lines.png')
    for h in [2, 3, 6]:
        plot_oos_ts(inf_results, indices, inf, 'Инфляция', h,
                    INF_DIR / 'plots' / f'inf_forecast_h{h}.png')

    print(f"\nRV результаты   → {RV_DIR}")
    print(f"Инфляция        → {INF_DIR}")
    print("\nГотово.")


if __name__ == '__main__':
    main()
