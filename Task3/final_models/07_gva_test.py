# 07_gva_test.py
# Тест прогнозной силы MPU-индексов для прогнозирования
# Валовой Добавленной Стоимости (GVA, ВДС).
#
# GVA — квартальные данные Росстата, интерполированные на месячную частоту.
# Используем год-к-году темп роста: gva_yoy(t) = GVA(t)/GVA(t-12) - 1
# Это убирает трендовую компоненту и делает ряд более стационарным,
# а AR-бенчмарк слабее (чем для уровня) — честнее проверять MPU.
#
# Экономическая логика:
#   MPU ↑ → неопределённость заимствований → инвестиции откладываются
#          → снижение ВДС через 2–6 кварталов (6–18 месяцев)
#
# Результаты -> Task3/final_models/gva_test/
# Запуск: python 07_gva_test.py

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
OUT_DIR  = BASE_DIR / 'gva_test'
PLOT_DIR = OUT_DIR / 'plots'

from data_loading.extra_data import get_gva_monthly_dataframe

HORIZONS     = [1, 2, 3, 6, 12]
SPLIT_FRAC   = 0.6
AR_ORDER     = 1
SAMPLE_START = '2019-03-01'   # начало MPU данных

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
    X = np.atleast_2d(np.array(X, dtype=float))
    if X.ndim == 1 or X.shape[0] == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])


def oos_r2(y_oos, y_pred, bench):
    mse_m = float(np.mean((y_oos - y_pred) ** 2))
    mse_b = float(np.mean((y_oos - bench)  ** 2))
    return 1.0 - mse_m / max(mse_b, 1e-10)


def run_oos(y_is, X_is, y_oos, X_oos, bench):
    try:
        model  = sm.OLS(y_is, X_is).fit(
            cov_type='HAC', cov_kwds={'maxlags': 3}
        )
        y_pred = model.predict(X_oos)
        r2     = oos_r2(y_oos, y_pred, bench)
        beta   = float(model.params[1]) if len(model.params) > 1 else np.nan
        pval   = float(model.pvalues[1]) if len(model.pvalues) > 1 else np.nan
        return r2, beta, pval
    except Exception:
        return np.nan, np.nan, np.nan


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


def load_gva() -> tuple[pd.Series, pd.Series]:
    """
    Загружает GVA, строит:
      - gva_level: уровень (млрд руб.)
      - gva_yoy:   год-к-году темп роста (%)

    Обрезает по SAMPLE_START.
    Возвращает оба ряда.
    """
    df = get_gva_monthly_dataframe()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Год-к-году темп роста
    df['gva_yoy'] = (df['Value'] / df['Value'].shift(12) - 1) * 100

    # Обрезаем по началу выборки MPU
    df_sample = (df[df['Date'] >= SAMPLE_START]
                 .dropna(subset=['gva_yoy'])
                 .reset_index(drop=True))

    level = df_sample.set_index('Date')['Value']
    yoy   = df_sample.set_index('Date')['gva_yoy']
    level.index = yoy.index = pd.to_datetime(level.index)

    print(f"\nGVA уровень:  N={len(level)}  "
          f"[{level.index.min().date()} — {level.index.max().date()}]  "
          f"mean={level.mean():.1f}  млрд руб.")
    print(f"GVA YoY (%):  "
          f"mean={yoy.mean():.2f}%  std={yoy.std():.2f}%  "
          f"min={yoy.min():.2f}%  max={yoy.max():.2f}%")

    return level, yoy


# ════════════════════════════════════════════════════════════
# Объединение рядов
# ════════════════════════════════════════════════════════════

def merge_series(mpu: pd.Series, target: pd.Series) -> pd.DataFrame:
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
        tolerance=pd.Timedelta('20D'),
        direction='nearest'
    ).dropna()
    return m.reset_index(drop=True)


# ════════════════════════════════════════════════════════════
# Основной тест
# ════════════════════════════════════════════════════════════

def ar_benchmark_test(mpu_name: str,
                       mpu: pd.Series,
                       target: pd.Series,
                       ar_order: int = AR_ORDER) -> pd.DataFrame:
    merged  = merge_series(mpu, target)
    n       = len(merged)
    n_split = int(n * SPLIT_FRAC)
    records = []

    for h in HORIZONS:
        fwd = merged['Target'].shift(-h)

        lags = {f'lag_{p}': merged['Target'].shift(p - 1)
                for p in range(1, ar_order + 1)}
        lag_mask = pd.concat(lags, axis=1).notna().all(axis=1)
        mask     = fwd.notna() & merged['MPU'].notna() & lag_mask

        idx  = merged.index[mask]
        is_  = idx[idx < n_split]
        oos_ = idx[idx >= n_split]

        if len(is_) < max(12, ar_order + 3) or len(oos_) < 3:
            continue

        y_is  = fwd.loc[is_].values.astype(float)
        y_oos = fwd.loc[oos_].values.astype(float)

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

        r2_ar,  _,        _        = run_oos(
            y_is, _add_const(ar_is),
            y_oos, _add_const(ar_oos), bench
        )
        r2_mpu, beta_mpu, pval_mpu = run_oos(
            y_is, _add_const(mpu_is.reshape(-1, 1)),
            y_oos, _add_const(mpu_oos.reshape(-1, 1)), bench
        )
        arx_is  = np.column_stack([ar_is,  mpu_is])
        arx_oos = np.column_stack([ar_oos, mpu_oos])
        r2_arx, beta_arx, pval_arx = run_oos(
            y_is, _add_const(arx_is),
            y_oos, _add_const(arx_oos), bench
        )

        gain = (r2_arx - r2_ar) if not np.isnan(r2_arx) else np.nan

        records.append(dict(
            mpu          = mpu_name,
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
    name = df['mpu'].iloc[0]
    print(f"\n  {name}:")
    print(f"  {'h':>3}  {'CONST':>7}  {'AR':>7}  "
          f"{'MPU':>7}  {'ARX':>7}  {'gain':>7}  "
          f"{'beta_mpu':>9}  {'p_mpu':>6}  beats_AR")
    print("  " + "─" * 78)
    for _, row in df.iterrows():
        mark = '✅' if row['mpu_beats_ar'] else '✗ '
        sig  = '**' if row['pval_mpu'] < 0.05 else '  '
        print(f"  {row['h']:>3}M "
              f" {row['R2_const']:>+7.3f}"
              f"  {row['R2_ar']:>+7.3f}"
              f"  {row['R2_mpu']:>+7.3f}"
              f"  {row['R2_arx']:>+7.3f}"
              f"  {row['gain_vs_ar']:>+7.3f}"
              f"  {row['beta_mpu']:>+9.3f}"
              f"  {row['pval_mpu']:.3f}{sig}"
              f"  {mark}")


# ════════════════════════════════════════════════════════════
# Графики
# ════════════════════════════════════════════════════════════

def plot_gva_overview(level: pd.Series, yoy: pd.Series) -> None:
    """GVA уровень и YoY рост за весь период."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Валовая добавленная стоимость (ВДС)', fontweight='bold')

    ax1.fill_between(level.index, level.values,
                     alpha=0.15, color='#1565C0')
    ax1.plot(level.index, level.values, color='#1565C0', lw=2)
    ax1.axvline(pd.Timestamp(SAMPLE_START), color='grey',
                lw=1.5, ls=':', label='Начало выборки MPU')
    ax1.set_ylabel('ВДС (млрд руб.)')
    ax1.set_title('Уровень ВДС')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)

    # YoY с цветом по знаку
    colors_yoy = np.where(yoy.values >= 0, '#27AE60', '#C62828')
    ax2.bar(yoy.index, yoy.values, color=colors_yoy,
            alpha=0.7, width=25, label='ВДС YoY (%)')
    ax2.axhline(0, color='black', lw=1.2)
    ax2.axvline(pd.Timestamp(SAMPLE_START), color='grey', lw=1.5, ls=':')
    ax2.set_ylabel('Год-к-году изменение (%)')
    ax2.set_title('Темп роста ВДС год-к-году')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.25)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    _save(fig, 'gva_overview.png')


def plot_mpu_vs_gva(indices: dict, yoy: pd.Series,
                     horizons_show: list = None) -> None:
    """
    MPU(t) vs GVA_YoY(t+h) для трёх горизонтов.
    Показывает корреляцию и scatter.
    """
    if horizons_show is None:
        horizons_show = [3, 6, 12]

    n_h = len(horizons_show)
    n_m = len(indices)
    fig, axes = plt.subplots(n_m, n_h,
                              figsize=(5 * n_h, 4.5 * n_m))
    if n_m == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Scatter: MPU(t) vs ВДС YoY(t+h)',
                 fontweight='bold', fontsize=13)

    for i, (name, mpu) in enumerate(indices.items()):
        color  = INDEX_COLORS.get(name, '#333')
        merged = merge_series(mpu, yoy)

        for j, h in enumerate(horizons_show):
            ax  = axes[i, j]
            fwd = merged['Target'].shift(-h)
            msk = fwd.notna() & merged['MPU'].notna()
            x   = merged.loc[msk, 'MPU'].values.astype(float)
            y   = fwd[msk].values.astype(float)

            ax.scatter(x, y, alpha=0.55, color=color, s=35, zorder=5)

            if len(x) > 4:
                z  = np.polyfit(x, y, 1)
                xf = np.linspace(x.min(), x.max(), 100)
                ax.plot(xf, np.polyval(z, xf),
                        color='black', lw=2,
                        label=f'slope={z[0]:.3f}')

            corr = np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan
            ax.axhline(0, color='grey', lw=0.8, ls=':')
            ax.axvline(0, color='grey', lw=0.8, ls=':')
            ax.set_title(f'{name}\nh={h}M  r={corr:.3f}',
                         fontsize=9, fontweight='bold')
            ax.set_xlabel('MPU', fontsize=8)
            ax.set_ylabel('ВДС YoY(t+h) %', fontsize=8)
            if len(x) > 4:
                ax.legend(fontsize=7)
            ax.grid(alpha=0.2)

    plt.tight_layout()
    _save(fig, 'scatter_mpu_vs_gva.png')


def plot_r2_comparison(all_results: list) -> None:
    """R²_oos четырёх моделей по горизонтам."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle('AR-бенчмарк: R²_oos | Цель: ВДС YoY',
                 fontweight='bold', fontsize=13)

    for ax, df in zip(axes, all_results):
        name  = df['mpu'].iloc[0]
        color = INDEX_COLORS.get(name, '#333')
        df_s  = df.sort_values('h')

        for col, label, clr, ls, lw in [
            ('R2_const', 'Константа',       'grey',  ':',  1.2),
            ('R2_ar',    f'AR({AR_ORDER})',  'black', '--', 2.0),
            ('R2_mpu',   'MPU',              color,   '-',  2.5),
            ('R2_arx',   'AR+MPU',           color,   '-.', 2.0),
        ]:
            ax.plot(df_s['h'], df_s[col], color=clr, ls=ls, lw=lw,
                    marker='o', markersize=6, label=label)
            for _, row in df_s.iterrows():
                v = row[col]
                if not np.isnan(v):
                    ax.annotate(f'{v:+.3f}', (row['h'], v),
                                textcoords='offset points',
                                xytext=(4, 4), fontsize=7, color=clr)

        # Зона прироста от MPU
        ax.fill_between(df_s['h'],
                        df_s['R2_ar'].values,
                        df_s['R2_arx'].values,
                        where=(df_s['R2_arx'].values > df_s['R2_ar'].values),
                        alpha=0.12, color=color, label='Прирост от MPU')

        ax.axhline(0, color='black', lw=1, ls='-', alpha=0.3)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Горизонт h (месяцев)')
        ax.set_ylabel('R²_oos')
        ax.set_xticks(HORIZONS)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    _save(fig, 'r2oos_comparison.png')


def plot_gain_heatmap(all_results: list) -> None:
    """Тепловая карта gain_vs_ar."""
    df     = pd.concat(all_results, ignore_index=True)
    names  = [r['mpu'].iloc[0] for r in all_results]
    n_rows = len(names)
    n_cols = len(HORIZONS)

    data = np.full((n_rows, n_cols), np.nan)
    for i, name in enumerate(names):
        for j, h in enumerate(HORIZONS):
            sub = df[(df['mpu'] == name) & (df['h'] == h)]
            if not sub.empty:
                data[i, j] = sub['gain_vs_ar'].values[0]

    vmax = max(np.nanmax(np.abs(data)), 0.01)
    fig, ax = plt.subplots(figsize=(10, max(3, n_rows + 1)))
    im = ax.imshow(data, cmap='RdYlGn',
                   vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax,
                 label='gain = R²_oos(ARX) − R²_oos(AR)')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f'h={h}M' for h in HORIZONS])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(names, fontsize=10)

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if not np.isnan(v):
                c = 'white' if abs(v) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{v:+.3f}',
                        ha='center', va='center',
                        fontsize=10, fontweight='bold', color=c)

    ax.set_title(
        f'Прирост R²_oos от добавления MPU к AR({AR_ORDER})\n'
        f'Цель: ВДС YoY (%)  |  Зелёный = MPU добавляет ценность',
        fontweight='bold'
    )
    plt.tight_layout()
    _save(fig, 'gain_heatmap.png')


def plot_beta_sign(all_results: list) -> None:
    """
    Beta MPU в базовой модели по горизонтам.
    Ожидаем отрицательную бету: рост MPU → снижение ВДС.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Beta MPU → ВДС YoY: знак и значимость',
                 fontweight='bold', fontsize=13)

    for df in all_results:
        name  = df['mpu'].iloc[0]
        color = INDEX_COLORS.get(name, '#333')
        df_s  = df.sort_values('h')

        ax1.plot(df_s['h'], df_s['beta_mpu'],
                 color=color, lw=2.5, marker='o', markersize=7,
                 label=name)
        ax2.plot(df_s['h'], df_s['pval_mpu'],
                 color=color, lw=2.5, marker='o', markersize=7,
                 label=name)

    # Теория: ожидаем beta < 0
    ax1.fill_between(HORIZONS,
                     [min(-3, min(df['beta_mpu'].min()
                                  for df in all_results)) - 0.5] * len(HORIZONS),
                     [0] * len(HORIZONS),
                     alpha=0.05, color='green',
                     label='Ожидаемая зона (beta < 0)')
    ax1.axhline(0, color='grey', lw=1.2, ls='--')
    ax1.set(xlabel='Горизонт h (месяцев)', ylabel='Beta',
            title='Beta MPU (базовая модель)\n'
                  '< 0 соответствует теории (MPU ↑ → ВДС ↓)')
    ax1.set_xticks(HORIZONS); ax1.legend(fontsize=9); ax1.grid(alpha=0.25)

    ax2.axhline(0.05, color='black', lw=1.5, ls='--', label='p=0.05')
    ax2.axhline(0.10, color='grey',  lw=1.0, ls=':',  label='p=0.10')
    ax2.fill_between(HORIZONS, 0, 0.05, alpha=0.08, color='green')
    ax2.set(xlabel='Горизонт h (месяцев)', ylabel='p-value',
            title='p-value beta\n(зелёная зона = статистически значимо)',
            ylim=(0, 1))
    ax2.set_xticks(HORIZONS); ax2.legend(fontsize=9); ax2.grid(alpha=0.25)

    plt.tight_layout()
    _save(fig, 'beta_horizon.png')


def plot_oos_timeseries(all_results: list,
                         indices: dict,
                         yoy: pd.Series,
                         horizon: int = 6) -> None:
    """Факт ВДС YoY vs прогноз MPU (OOS период)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(yoy.index, yoy.values,
            color='black', lw=2.5, zorder=10,
            label='ВДС YoY (факт)')
    ax.axhline(0, color='grey', lw=1, ls=':', alpha=0.6)

    split_shown = False
    for name, mpu in indices.items():
        merged  = merge_series(mpu, yoy)
        n_split = int(len(merged) * SPLIT_FRAC)
        fwd     = merged['Target'].shift(-horizon)
        mask    = fwd.notna() & merged['MPU'].notna()
        is_     = merged.index[mask & (merged.index < n_split)]
        oos_    = merged.index[mask & (merged.index >= n_split)]

        if len(is_) < 5 or len(oos_) < 2:
            continue

        y_is   = fwd.loc[is_].values.astype(float)
        x_is   = merged.loc[is_,  'MPU'].values.astype(float)
        x_oos  = merged.loc[oos_, 'MPU'].values.astype(float)
        model  = sm.OLS(y_is, _add_const(x_is.reshape(-1, 1))).fit()
        y_pred = model.predict(_add_const(x_oos.reshape(-1, 1)))

        d_oos  = merged.loc[oos_, 'Date'].values
        color  = INDEX_COLORS.get(name, '#333')

        res = next((r for r in all_results if r['mpu'].iloc[0] == name),
                   None)
        r2  = np.nan
        if res is not None:
            row = res[res['h'] == horizon]
            if not row.empty:
                r2 = row['R2_mpu'].values[0]

        ax.plot(d_oos, y_pred, color=color, lw=2, ls='--',
                label=f'{name}  R²_mpu={r2:+.3f}')

        if not split_shown and len(oos_) > 0:
            sd = merged.loc[oos_[0], 'Date']
            ax.axvline(pd.Timestamp(sd), color='grey', lw=1.5, ls='-.',
                       label=f'Начало OOS: {pd.Timestamp(sd).date()}')
            split_shown = True

    ax.set_title(f'Прогноз ВДС YoY(t+{horizon}M): факт vs MPU',
                 fontweight='bold', fontsize=12)
    ax.set(xlabel='Дата', ylabel='ВДС YoY (%)')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    _save(fig, f'oos_timeseries_h{horizon}.png')


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"07 | ТЕСТ MPU → ВДС YoY  "
          f"(AR_ORDER={AR_ORDER}, split={SPLIT_FRAC:.0%})")
    print("=" * 65)

    indices        = load_indices()
    gva_level, yoy = load_gva()

    # Сохраняем данные
    pd.DataFrame({
        'Date': gva_level.index,
        'GVA_level': gva_level.values,
        'GVA_yoy':   yoy.reindex(gva_level.index).values,
    }).to_csv(OUT_DIR / 'gva_sample.csv', index=False)

    print(f"\n{'─'*65}")
    print("Результаты (целевая переменная: ВДС YoY %):")
    print(f"{'─'*65}")
    print(f"  {'h':>3}  {'CONST':>7}  {'AR':>7}  "
          f"{'MPU':>7}  {'ARX':>7}  {'gain':>7}  "
          f"{'beta_mpu':>9}  {'p_mpu':>6}  beats_AR")

    all_results = []
    for name, mpu in indices.items():
        res = ar_benchmark_test(name, mpu, yoy)
        if res.empty:
            print(f"\n  {name}: нет данных")
            continue
        print_results(res)
        all_results.append(res)
        res.to_csv(OUT_DIR / f'gva_{name}.csv', index=False)

    if not all_results:
        print("Нет результатов.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUT_DIR / 'gva_all.csv', index=False)

    # Итог
    print(f"\n{'='*65}")
    print("ИТОГ: MPU добавляет ценность сверх AR для ВДС YoY?")
    print(f"{'='*65}")
    wins = combined[combined['mpu_beats_ar'] == True]
    if wins.empty:
        print("  Нет ни одного горизонта где MPU бьёт AR.")
        print("  Авторегрессия ВДС YoY поглощает сигнал MPU.")
    else:
        for _, row in wins.iterrows():
            sig = '(p<0.05)' if row['pval_arx_mpu'] < 0.05 else ''
            bsig = '(p<0.05)' if row['pval_mpu'] < 0.05 else ''
            print(f"  ✅ {row['mpu']:<14}  h={row['h']:2d}M  "
                  f"gain={row['gain_vs_ar']:+.4f}  "
                  f"beta_mpu={row['beta_mpu']:+.3f} {bsig}  "
                  f"p_arx={row['pval_arx_mpu']:.3f} {sig}")

    # Графики
    print("\nГенерация графиков...")
    plot_gva_overview(gva_level, yoy.reindex(gva_level.index))
    plot_mpu_vs_gva(indices, yoy, horizons_show=[3, 6, 12])
    plot_r2_comparison(all_results)
    plot_gain_heatmap(all_results)
    plot_beta_sign(all_results)
    for h in [3, 6, 12]:
        plot_oos_timeseries(all_results, indices, yoy, horizon=h)

    print(f"\nГотово → {OUT_DIR.resolve()}")


if __name__ == '__main__':
    main()
