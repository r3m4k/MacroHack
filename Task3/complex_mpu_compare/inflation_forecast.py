# inflation_forecast.py
# Автономный скрипт для тестирования прогнозной силы MPU
# относительно инфляции (ИПЦ г/г).
#
# Запускается ОТДЕЛЬНО после main.py:
#   python main.py              <- строит индексы, сохраняет CSV
#   python inflation_forecast.py <- читает CSV, тестирует, строит графики
#
# Читает из results/:
#   mpu_aggregated.csv   -> MPU_decay_norm
#   mpu_iv_surface.csv   -> MPU_atm_norm
#   mpu_rv.csv           -> MPU_rv_std_norm
#
# Целевая переменная: инфляция г/г (%) из key_rate_df
#
# Методология:
#   Базовая модель:       Inflation(t+h) = α + β·MPU(t) + ε
#   Модель с контролями:  Inflation(t+h) = α + β·MPU(t)
#                                            + γ·Inflation(t)
#                                            + δ·Key_Rate(t) + ε
#   Критерий: R2_oos = 1 - MSE_model / MSE_benchmark
#   Бенчмарк: среднее инфляции по in-sample периоду

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Путь к папке проекта (на случай запуска из другой директории)
PROJECT_DIR = Path(__file__).parent
RESULT_DIR  = PROJECT_DIR / 'results'
PLOT_DIR    = RESULT_DIR / 'plots'

HORIZONS   = [1, 2, 3, 6, 12]
SPLIT_FRAC = 0.6

INDEX_COLORS = {
    'MPU_rv_std': '#8E44AD',
    'MPU_decay':  '#E65100',
    'MPU_atm':    '#27AE60',
}


# ============================================================
# Утилиты
# ============================================================

def _save(fig: plt.Figure, name: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def _add_const(x: np.ndarray) -> np.ndarray:
    """Надёжный add_constant для любого размера выборки."""
    x = np.atleast_1d(np.array(x, dtype=float)).reshape(-1)
    return np.column_stack([np.ones(len(x)), x])


def _add_const_multi(X: np.ndarray) -> np.ndarray:
    """add_constant для матрицы регрессоров."""
    X = np.atleast_2d(np.array(X, dtype=float))
    if X.shape[0] == 1:
        X = X.reshape(-1, X.shape[1]) if X.ndim > 1 else X.reshape(1, -1)
    return np.column_stack([np.ones(X.shape[0]), X])


# ============================================================
# Шаг 1: Загрузка индексов из сохранённых CSV
# ============================================================

def load_indices() -> dict[str, pd.Series]:
    """
    Читает три MPU-индекса из CSV-файлов, сохранённых main.py.

    Возвращает:
        dict {name: pd.Series с DatetimeIndex}

    Бросает FileNotFoundError если CSV не найден —
    в этом случае нужно сначала запустить main.py.
    """
    files = {
        'MPU_decay':  ('mpu_aggregated.csv',  'MPU_decay_norm'),
        'MPU_atm':    ('mpu_iv_surface.csv',  'MPU_atm_norm'),
        'MPU_rv_std': ('mpu_rv.csv',          'MPU_rv_std_norm'),
    }

    indices = {}
    print("Загрузка MPU-индексов из CSV:")
    for name, (fname, col) in files.items():
        path = RESULT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Файл {path} не найден.\n"
                f"Сначала запустите: python main.py"
            )
        df = pd.read_csv(path, parse_dates=['Date'])
        if col not in df.columns:
            raise KeyError(f"Колонка '{col}' не найдена в {fname}. "
                           f"Доступные: {df.columns.tolist()}")
        s = df.set_index('Date')[col].sort_index()
        s.index = pd.to_datetime(s.index)
        s = s.dropna()
        indices[name] = s
        print(f"  {name:<14}  N={len(s)}  "
              f"[{s.index.min().date()} — {s.index.max().date()}]")

    return indices


# ============================================================
# Шаг 2: Загрузка инфляции
# ============================================================

def load_inflation() -> pd.DataFrame:
    """
    Загружает данные об инфляции через get_key_rate_dataframe().

    Добавляет колонки:
        Delta_inf     — изменение инфляции м/м
        Inflation_gap — отклонение от таргета ЦБ (4%)

    Возвращает:
        DataFrame с колонками Date, Inflation, Key_Rate,
                               Delta_inf, Inflation_gap
    """
    # Добавляем папку проекта в sys.path для импорта data_loading
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from data_loading.case_2 import get_key_rate_dataframe

    key_rate_df = get_key_rate_dataframe()

    df = (key_rate_df[['Date', 'Key Rate', 'Inflation']]
          .rename(columns={'Key Rate': 'Key_Rate'})
          .dropna(subset=['Date', 'Inflation'])   # убираем NaT
          .sort_values('Date')
          .reset_index(drop=True))

    df['Delta_inf']     = df['Inflation'].diff()
    df['Inflation_gap'] = df['Inflation'] - 4.0

    print(f"\nИнфляция загружена: {len(df)} наблюдений")
    print(f"  Период: {df['Date'].min().date()} — "
          f"{df['Date'].max().date()}")
    print(f"  mean={df['Inflation'].mean():.2f}%  "
          f"std={df['Inflation'].std():.2f}%  "
          f"min={df['Inflation'].min():.2f}%  "
          f"max={df['Inflation'].max():.2f}%")

    return df


# ============================================================
# Шаг 3: Объединение MPU и инфляции
# ============================================================

def merge_mpu_inflation(mpu_series: pd.Series,
                        inf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет MPU и инфляцию по ближайшей дате (±15 дней).
    """
    mpu_df = pd.DataFrame({
        'Date': pd.to_datetime(mpu_series.index),
        'MPU':  mpu_series.values,
    }).dropna(subset=['Date', 'MPU'])

    inf_part = (inf_df[['Date', 'Inflation', 'Key_Rate', 'Delta_inf']]
                .dropna(subset=['Date', 'Inflation'])
                .copy())
    inf_part['Date'] = pd.to_datetime(inf_part['Date'])

    merged = pd.merge_asof(
        mpu_df.sort_values('Date'),
        inf_part.sort_values('Date'),
        on='Date',
        tolerance=pd.Timedelta('15D'),
        direction='nearest'
    ).dropna(subset=['MPU', 'Inflation'])

    return merged.reset_index(drop=True)


# ============================================================
# Шаг 4: OOS тест для одного индекса
# ============================================================

def oos_inflation(mpu_series: pd.Series,
                  inf_df: pd.DataFrame,
                  name: str) -> pd.DataFrame:
    """
    OOS тест прогнозной силы MPU для инфляции.

    Базовая модель:
        Inflation(t+h) = α + β·MPU(t) + ε

    Модель с контролями:
        Inflation(t+h) = α + β·MPU(t)
                           + γ·Inflation(t)   <- инерция инфляции
                           + δ·Key_Rate(t)    <- реакция ЦБ
                           + ε

    Контроли важны: нужно показать что MPU несёт информацию
    СВЕРХ уже известного уровня инфляции и ставки.

    Возвращает DataFrame с R2_oos, beta, p-value по горизонтам.
    """
    merged  = merge_mpu_inflation(mpu_series, inf_df)
    n_split = int(len(merged) * SPLIT_FRAC)
    records = []

    for h in HORIZONS:
        inf_fwd = merged['Inflation'].shift(-h)
        mask    = inf_fwd.notna() & merged['MPU'].notna()

        idx_all = merged.index[mask]
        idx_is  = idx_all[idx_all < n_split]
        idx_oos = idx_all[idx_all >= n_split]

        if len(idx_is) < 8 or len(idx_oos) < 3:
            continue

        y_is  = inf_fwd.loc[idx_is].values.astype(float)
        x_is  = merged.loc[idx_is,  'MPU'].values.astype(float)
        y_oos = inf_fwd.loc[idx_oos].values.astype(float)
        x_oos = merged.loc[idx_oos, 'MPU'].values.astype(float)

        # ── Базовая модель ─────────────────────────────────
        m_base      = sm.OLS(y_is, _add_const(x_is)).fit(
            cov_type='HAC', cov_kwds={'maxlags': h}
        )
        y_pred_base = m_base.predict(_add_const(x_oos))
        bench       = np.full_like(y_oos, y_is.mean())

        mse_base  = float(np.mean((y_oos - y_pred_base) ** 2))
        mse_bench = float(np.mean((y_oos - bench) ** 2))
        r2_base   = 1.0 - mse_base / max(mse_bench, 1e-10)

        # ── Модель с контролями ────────────────────────────
        inf_lag = merged['Inflation'].shift(1)
        kr_col  = merged['Key_Rate']
        mask_c  = (mask
                   & inf_lag.notna()
                   & kr_col.notna()
                   & merged['MPU'].notna())

        idx_is_c  = merged.index[mask_c & (merged.index < n_split)]
        idx_oos_c = merged.index[mask_c & (merged.index >= n_split)]

        r2_ctrl = beta_ctrl = pval_ctrl = np.nan

        if len(idx_is_c) >= 10 and len(idx_oos_c) >= 3:
            try:
                y_is_c  = inf_fwd.loc[idx_is_c].values.astype(float)
                y_oos_c = inf_fwd.loc[idx_oos_c].values.astype(float)

                X_is_c = _add_const_multi(np.column_stack([
                    merged.loc[idx_is_c,  'MPU'].values.astype(float),
                    inf_lag.loc[idx_is_c].values.astype(float),
                    kr_col.loc[idx_is_c].values.astype(float),
                ]))
                X_oos_c = _add_const_multi(np.column_stack([
                    merged.loc[idx_oos_c, 'MPU'].values.astype(float),
                    inf_lag.loc[idx_oos_c].values.astype(float),
                    kr_col.loc[idx_oos_c].values.astype(float),
                ]))

                m_ctrl      = sm.OLS(y_is_c, X_is_c).fit(
                    cov_type='HAC', cov_kwds={'maxlags': h}
                )
                y_pred_ctrl = m_ctrl.predict(X_oos_c)
                bench_c     = np.full_like(y_oos_c, y_is_c.mean())

                mse_ctrl  = float(np.mean((y_oos_c - y_pred_ctrl) ** 2))
                mse_bch_c = float(np.mean((y_oos_c - bench_c) ** 2))
                r2_ctrl   = 1.0 - mse_ctrl / max(mse_bch_c, 1e-10)
                beta_ctrl = float(m_ctrl.params[1])
                pval_ctrl = float(m_ctrl.pvalues[1])
            except Exception as e:
                print(f"    [WARN] {name} h={h} ctrl: {e}")

        records.append({
            'name':        name,
            'h':           h,
            'N_is':        len(idx_is),
            'N_oos':       len(idx_oos),
            'beta_base':   round(float(m_base.params[1]), 4),
            'pval_base':   round(float(m_base.pvalues[1]), 4),
            'R2_oos_base': round(r2_base, 4),
            'R2_oos_ctrl': round(r2_ctrl, 4) if not np.isnan(r2_ctrl) else np.nan,
            'beta_ctrl':   round(beta_ctrl, 4) if not np.isnan(beta_ctrl) else np.nan,
            'pval_ctrl':   round(pval_ctrl, 4) if not np.isnan(pval_ctrl) else np.nan,
            'beats_bench': r2_base > 0,
        })

    return pd.DataFrame(records)


# ============================================================
# Шаг 5: Сравнение трёх индексов
# ============================================================

def compare_inflation_forecast(indices: dict[str, pd.Series],
                               inf_df: pd.DataFrame
                               ) -> tuple[pd.DataFrame, dict]:
    """
    Прогоняет все три индекса и выводит сводную таблицу.
    """
    print(f"\n{'='*60}")
    print("ПРОГНОЗИРОВАНИЕ ИНФЛЯЦИИ: OOS ТЕСТ")
    print(f"Горизонты: {HORIZONS}  |  Split: {SPLIT_FRAC:.0%} IS")
    print("=" * 60)

    all_results = {}
    for name, series in indices.items():
        print(f"\n  {name}:")
        res = oos_inflation(series, inf_df, name)
        all_results[name] = res

        for _, row in res.iterrows():
            sig  = '**' if row['pval_base'] < 0.05 else '  '
            ctrl = (f"  R2_ctrl={row['R2_oos_ctrl']:+.3f}"
                    if not np.isnan(row['R2_oos_ctrl']) else '')
            print(f"    h={row['h']:2d}M: "
                  f"beta={row['beta_base']:+.3f} "
                  f"(p={row['pval_base']:.3f}){sig}  "
                  f"R2_oos={row['R2_oos_base']:+.3f}{ctrl}")

    # Сводная таблица R2_oos (базовая)
    frames = []
    for name, res in all_results.items():
        sub = (res[['h', 'R2_oos_base']]
               .rename(columns={'R2_oos_base': name})
               .set_index('h'))
        frames.append(sub)
    summary = pd.concat(frames, axis=1)
    summary['best'] = summary.idxmax(axis=1)

    print(f"\n{'='*60}")
    print("СВОДНАЯ ТАБЛИЦА R2_oos (базовая модель)")
    print("=" * 60)
    print(summary.to_string())
    print("\nЛучший индекс на каждом горизонте:")
    for h, row in summary.iterrows():
        best = row['best']
        val  = row[best]
        print(f"  h={h:2d}M  ->  {best}  R2_oos={val:+.4f}")

    return summary, all_results


# ============================================================
# Визуализации
# ============================================================

def plot_inflation_overview(inf_df: pd.DataFrame) -> None:
    """Инфляция и ключевая ставка за всю историю."""
    df  = inf_df.sort_values('Date')
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.fill_between(df['Date'], df['Inflation'],
                     alpha=0.15, color='#C62828')
    ax1.plot(df['Date'], df['Inflation'],
             color='#C62828', lw=2.5, label='Инфляция г/г (%)')
    ax1.axhline(4.0, color='#C62828', lw=1, ls='--',
                alpha=0.6, label='Таргет ЦБ (4%)')

    ax2 = ax1.twinx()
    ax2.step(df['Date'], df['Key_Rate'],
             where='post', color='#1565C0', lw=2,
             label='Ключевая ставка (%)')
    ax2.set_ylabel('Ключевая ставка (%)', color='#1565C0')
    ax2.tick_params(axis='y', labelcolor='#1565C0')

    ax1.set_xlabel('Дата')
    ax1.set_ylabel('Инфляция г/г (%)', color='#C62828')
    ax1.tick_params(axis='y', labelcolor='#C62828')
    ax1.set_title('Инфляция и ключевая ставка ЦБ РФ',
                  fontsize=12, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    _save(fig, 'inflation_overview.png')


def plot_inflation_r2oos(summary: pd.DataFrame) -> None:
    """Столбчатая диаграмма R2_oos по горизонтам."""
    names    = [c for c in summary.columns if c != 'best']
    horizons = list(summary.index)
    n_h      = len(horizons)

    fig, axes = plt.subplots(1, n_h, figsize=(4 * n_h, 6))
    if n_h == 1:
        axes = [axes]

    fig.suptitle(
        'Прогнозирование инфляции: R2_oos по горизонтам\n'
        f'OOS = последние {100*(1-SPLIT_FRAC):.0f}% выборки',
        fontsize=13, fontweight='bold'
    )

    for ax, h in zip(axes, horizons):
        vals   = [summary.loc[h, n] for n in names]
        colors = [INDEX_COLORS.get(n, '#333') for n in names]

        bars = ax.bar(range(len(names)), vals,
                      color=colors, alpha=0.85,
                      edgecolor='white', width=0.6)
        ax.axhline(0, color='black', lw=1.5, ls='--',
                   label='Бенчмарк')

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.005 * (1 if val >= 0 else -1),
                        f'{val:+.3f}',
                        ha='center',
                        va='bottom' if val >= 0 else 'top',
                        fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.set_title(f'h = {h}M', fontweight='bold')
        if h == horizons[0]:
            ax.set_ylabel('R2_oos')
            ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, 'inflation_r2oos_comparison.png')


def plot_inflation_beta_horizon(all_results: dict) -> None:
    """Beta и p-value как функции горизонта h."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Прогнозирование инфляции: beta и значимость по горизонтам',
        fontsize=13, fontweight='bold'
    )

    for name, res in all_results.items():
        color = INDEX_COLORS.get(name, '#333')
        hs    = res['h'].values
        betas = res['beta_base'].values
        pvals = res['pval_base'].values

        ax1.plot(hs, betas, color=color, lw=2.5,
                 marker='o', markersize=7, label=name)
        ax2.plot(hs, pvals, color=color, lw=2.5,
                 marker='o', markersize=7, label=name)

    ax1.axhline(0, color='grey', lw=1, ls='--')
    ax1.set_xlabel('Горизонт h (месяцев)')
    ax1.set_ylabel('Beta')
    ax1.set_title('Коэффициент beta (базовая модель)')
    ax1.set_xticks(HORIZONS)
    ax1.legend(fontsize=9)

    ax2.axhline(0.05, color='black', lw=1.5, ls='--',
                label='p = 0.05')
    ax2.axhline(0.10, color='grey', lw=1, ls=':',
                label='p = 0.10')
    ax2.fill_between(HORIZONS, 0, 0.05,
                     alpha=0.08, color='green')
    ax2.set_xlabel('Горизонт h (месяцев)')
    ax2.set_ylabel('p-value')
    ax2.set_title('p-value  (зелёная зона = p < 0.05)')
    ax2.set_ylim(0, 1)
    ax2.set_xticks(HORIZONS)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, 'inflation_beta_horizon.png')


def plot_inflation_forecast_ts(all_results: dict,
                               indices: dict[str, pd.Series],
                               inf_df: pd.DataFrame,
                               horizon: int = 3) -> None:
    """Временной ряд: факт инфляции vs прогнозы MPU (OOS период)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    df_sorted = inf_df.sort_values('Date')
    ax.plot(df_sorted['Date'], df_sorted['Inflation'],
            color='black', lw=2.5, zorder=10,
            label='Инфляция (факт)')
    ax.axhline(4.0, color='grey', lw=1, ls='--',
               alpha=0.7, label='Таргет ЦБ (4%)')

    split_date_shown = False
    for name, series in indices.items():
        merged  = merge_mpu_inflation(series, inf_df)
        n_split = int(len(merged) * SPLIT_FRAC)

        inf_fwd = merged['Inflation'].shift(-horizon)
        mask    = inf_fwd.notna() & merged['MPU'].notna()
        idx_is  = merged.index[mask & (merged.index < n_split)]
        idx_oos = merged.index[mask & (merged.index >= n_split)]

        if len(idx_is) < 5 or len(idx_oos) < 2:
            continue

        y_is  = inf_fwd.loc[idx_is].values.astype(float)
        x_is  = merged.loc[idx_is,  'MPU'].values.astype(float)
        x_oos = merged.loc[idx_oos, 'MPU'].values.astype(float)

        model  = sm.OLS(y_is, _add_const(x_is)).fit()
        y_pred = model.predict(_add_const(x_oos))

        dates_oos = merged.loc[idx_oos, 'Date'].values
        color     = INDEX_COLORS.get(name, '#333')

        res = all_results.get(name, pd.DataFrame())
        row = res[res['h'] == horizon] if not res.empty else pd.DataFrame()
        r2  = row['R2_oos_base'].values[0] if not row.empty else np.nan

        ax.plot(dates_oos, y_pred, color=color, lw=2, ls='--',
                label=f'{name}  R2_oos={r2:+.3f}')

        if not split_date_shown and len(idx_oos) > 0:
            sd = merged.loc[idx_oos[0], 'Date']
            ax.axvline(pd.Timestamp(sd), color='grey',
                       lw=1.5, ls='-.',
                       label=f'Начало OOS: {pd.Timestamp(sd).date()}')
            split_date_shown = True

    ax.set_title(
        f'Прогнозирование инфляции: факт vs MPU (h={horizon}M)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('Дата')
    ax.set_ylabel('Инфляция г/г (%)')
    ax.legend(fontsize=9, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    _save(fig, f'inflation_forecast_h{horizon}.png')


def plot_inflation_scatter(indices: dict[str, pd.Series],
                           inf_df: pd.DataFrame,
                           horizon: int = 6) -> None:
    """Scatter MPU(t) vs Inflation(t+h) для трёх индексов."""
    names = list(indices.keys())
    n     = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f'Scatter MPU(t) vs Инфляция(t+{horizon}M)',
                 fontsize=13, fontweight='bold')

    for ax, name in zip(axes, names):
        merged  = merge_mpu_inflation(indices[name], inf_df)
        inf_fwd = merged['Inflation'].shift(-horizon)
        mask    = inf_fwd.notna() & merged['MPU'].notna()
        x       = merged.loc[mask, 'MPU'].values.astype(float)
        y       = inf_fwd[mask].values.astype(float)
        color   = INDEX_COLORS.get(name, '#333')

        ax.scatter(x, y, alpha=0.55, color=color, s=40, zorder=5)
        if len(x) > 3:
            z  = np.polyfit(x, y, 1)
            xf = np.linspace(x.min(), x.max(), 100)
            ax.plot(xf, np.polyval(z, xf), color='black', lw=2,
                    label=f'slope={z[0]:.3f}')
        ax.axhline(4.0, color='grey', lw=1, ls='--',
                   alpha=0.6, label='Таргет 4%')
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('MPU')
        ax.set_ylabel(f'Инфляция(t+{horizon}M)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, f'inflation_scatter_h{horizon}.png')


def plot_ctrl_vs_base(all_results: dict) -> None:
    """
    Сравнение R2_oos базовой модели и модели с контролями.

    Показывает добавляет ли MPU информацию сверх
    лаговой инфляции и ключевой ставки.
    """
    names    = list(all_results.keys())
    n        = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        'R2_oos: базовая модель vs модель с контролями\n'
        'Контроли: Inflation(t) + Key_Rate(t)',
        fontsize=13, fontweight='bold'
    )

    for ax, name in zip(axes, names):
        res    = all_results[name]
        hs     = res['h'].values
        r2_b   = res['R2_oos_base'].values
        r2_c   = res['R2_oos_ctrl'].values
        color  = INDEX_COLORS.get(name, '#333')

        ax.plot(hs, r2_b, color=color, lw=2.5,
                marker='o', markersize=7, label='Базовая')
        ax.plot(hs, r2_c, color=color, lw=2, ls='--',
                marker='^', markersize=7, label='С контролями')
        ax.axhline(0, color='black', lw=1.2, ls='--')
        ax.fill_between(hs, 0,
                        np.maximum(r2_b, 0),
                        alpha=0.08, color=color)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Горизонт h (месяцев)')
        ax.set_ylabel('R2_oos')
        ax.set_xticks(hs)
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, 'inflation_base_vs_ctrl.png')


# ============================================================
# Точка входа
# ============================================================

if __name__ == '__main__':

    print("=" * 60)
    print("ТЕСТ: ПРОГНОЗИРОВАНИЕ ИНФЛЯЦИИ")
    print("Читает индексы из results/ (без пересчёта)")
    print("=" * 60)

    RESULT_DIR.mkdir(exist_ok=True)

    # 1. Загрузка индексов из CSV
    indices = load_indices()

    # 2. Загрузка инфляции
    inf_df = load_inflation()

    # 3. OOS тест
    summary, all_results = compare_inflation_forecast(indices, inf_df)

    # 4. Сохранение таблиц
    summary.to_csv(RESULT_DIR / 'inflation_r2oos.csv')
    for name, res in all_results.items():
        res.to_csv(RESULT_DIR / f'inflation_{name}.csv', index=False)
    print(f"\nТаблицы сохранены в: {RESULT_DIR}/")

    # 5. Графики
    print("\nГенерация графиков...")
    plot_inflation_overview(inf_df)
    plot_inflation_r2oos(summary)
    plot_inflation_beta_horizon(all_results)
    plot_ctrl_vs_base(all_results)

    for h in [3, 6, 12]:
        plot_inflation_forecast_ts(all_results, indices, inf_df, horizon=h)

    plot_inflation_scatter(indices, inf_df, horizon=6)

    print(f"\nГотово. Графики: {PLOT_DIR.resolve()}")