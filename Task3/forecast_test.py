# forecast_test.py
# Тест прогнозной силы MPU_pca: предсказывает ли рост MPU
# будущую реализованную волатильность (RV) ключевой ставки?
#
# Структура теста:
#   1. Расчёт RV ключевой ставки (скользящее окно)
#   2. Визуальный анализ (scatter + временной ряд)
#   3. Прогнозные регрессии OLS на горизонтах h = 1..6 месяцев
#   4. Out-of-sample тест (R2_oos)
#   5. Тест Грэнджера на причинность
#   6. Сводная таблица результатов

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from data_loading.case_2 import get_key_rate_dataframe

PLOT_DIR  = Path('results') / 'plots'
RESULT_DIR = Path('results')


# ============================================================
# ШАГ 1: Расчёт реализованной волатильности (RV)
# ============================================================

def compute_rv(key_rate_df: pd.DataFrame,
               window: int = 3) -> pd.DataFrame:
    """
    Считает реализованную волатильность ключевой ставки.

    RV — это фактическая историческая изменчивость ставки.
    Используется как целевая переменная для теста прогнозной силы MPU.

    Метод:
        RV_t = std(delta_r_{t-window+1}, ..., delta_r_t)
        где delta_r_t = r_t - r_{t-1} — ежемесячное изменение ставки.

    Параметры:
        key_rate_df — DataFrame с колонками Date, Key Rate
        window      — окно в месяцах (по умолчанию 3)

    Возвращает:
        DataFrame с колонками Date, Key_Rate, Delta, RV_{window}M
    """
    df = key_rate_df[['Date', 'Key Rate']].sort_values('Date').copy()
    df = df.rename(columns={'Key Rate': 'Key_Rate'})

    # Ежемесячное изменение ставки (в п.п.)
    df['Delta'] = df['Key_Rate'].diff()

    # RV = стандартное отклонение изменений за окно
    col = f'RV_{window}M'
    df[col] = df['Delta'].rolling(window=window, min_periods=window).std()

    print(f"\nRV ключевой ставки (окно {window}M):")
    print(df[['Date', 'Key_Rate', 'Delta', col]].dropna().describe().round(4))

    return df


# ============================================================
# ШАГ 2: Объединение MPU и RV
# ============================================================

def merge_mpu_rv(mpu_df: pd.DataFrame,
                 rv_df: pd.DataFrame,
                 rv_col: str = 'RV_3M') -> pd.DataFrame:
    """
    Объединяет MPU_pca и RV по дате.

    MPU и RV могут иметь разную периодичность или неполное пересечение.
    Используем merge_asof для ближайшей даты (tolerance = 15 дней).

    Возвращает:
        DataFrame с колонками Date, MPU_pca, MPU_pca_norm, RV, Key_Rate
    """
    mpu = mpu_df[['Date', 'MPU_pca', 'MPU_pca_norm']].copy()
    rv  = rv_df[['Date', 'Key_Rate', 'Delta', rv_col]].dropna().copy()

    mpu = mpu.sort_values('Date')
    rv  = rv.sort_values('Date')

    merged = pd.merge_asof(
        mpu, rv,
        on='Date',
        tolerance=pd.Timedelta('15D'),
        direction='nearest'
    ).dropna(subset=['MPU_pca', rv_col])

    merged = merged.rename(columns={rv_col: 'RV'})
    merged = merged.sort_values('Date').reset_index(drop=True)

    print(f"\nОбъединённая выборка: {len(merged)} наблюдений")
    print(f"Период: {merged['Date'].min().date()} -> "
          f"{merged['Date'].max().date()}")

    return merged


# ============================================================
# ШАГ 3: Прогнозные регрессии OLS
# ============================================================

def forecast_regression(merged: pd.DataFrame,
                        horizons: list[int] | None = None,
                        with_controls: bool = True
                        ) -> pd.DataFrame:
    """
    Тестирует прогнозную силу MPU_pca на горизонтах h месяцев вперёд.

    Модели:
        Базовая:    RV(t+h) = alpha + beta * MPU(t) + eps
        С контролями: RV(t+h) = alpha + beta * MPU(t)
                               + gamma * RV(t) + eps

    Гипотеза H0: beta = 0 (MPU не предсказывает RV).
    H0 отвергается при p-value < 0.05 -> MPU значим.

    Параметры:
        merged        — объединённый DataFrame
        horizons      — горизонты прогноза в месяцах
        with_controls — добавить лаговую RV как контрол

    Возвращает:
        DataFrame с результатами по всем горизонтам
    """
    if horizons is None:
        horizons = [1, 2, 3, 6]

    records = []

    for h in horizons:
        # RV через h месяцев (сдвиг назад по времени)
        rv_fwd = merged['RV'].shift(-h)

        # Убираем NaN
        mask = rv_fwd.notna() & merged['MPU_pca'].notna()
        y    = rv_fwd[mask].values
        x    = merged.loc[mask, 'MPU_pca'].values

        if len(y) < 10:
            print(f"  [SKIP] h={h}: мало наблюдений ({len(y)})")
            continue

        # ── Базовая регрессия ──────────────────────────────
        X_base  = sm.add_constant(x)
        res_base = sm.OLS(y, X_base).fit(
            cov_type='HAC', cov_kwds={'maxlags': h}
        )
        beta_b  = res_base.params[1]
        pval_b  = res_base.pvalues[1]
        r2_b    = res_base.rsquared
        tstat_b = res_base.tvalues[1]

        # ── Регрессия с контролями ─────────────────────────
        beta_c = pval_c = r2_c = tstat_c = np.nan
        if with_controls:
            rv_lag = merged['RV'].shift(1)
            x2     = rv_lag[mask].values
            if not np.all(np.isnan(x2)):
                valid = ~np.isnan(x2) & ~np.isnan(y)
                if valid.sum() >= 10:
                    X_ctrl  = sm.add_constant(
                        np.column_stack([x[valid], x2[valid]])
                    )
                    res_ctrl = sm.OLS(y[valid], X_ctrl).fit(
                        cov_type='HAC', cov_kwds={'maxlags': h}
                    )
                    beta_c  = res_ctrl.params[1]
                    pval_c  = res_ctrl.pvalues[1]
                    r2_c    = res_ctrl.rsquared
                    tstat_c = res_ctrl.tvalues[1]

        records.append({
            'Горизонт h':        h,
            'N':                 int(mask.sum()),
            # Базовая модель
            'beta (base)':       round(beta_b, 4),
            't-stat (base)':     round(tstat_b, 3),
            'p-value (base)':    round(pval_b, 4),
            'R2 (base)':         round(r2_b, 4),
            'Значим (base)':     'ДА' if pval_b < 0.05 else 'нет',
            # С контролями
            'beta (ctrl)':       round(beta_c, 4) if not np.isnan(beta_c) else '-',
            't-stat (ctrl)':     round(tstat_c, 3) if not np.isnan(tstat_c) else '-',
            'p-value (ctrl)':    round(pval_c, 4) if not np.isnan(pval_c) else '-',
            'R2 (ctrl)':         round(r2_c, 4) if not np.isnan(r2_c) else '-',
            'Значим (ctrl)':     ('ДА' if pval_c < 0.05 else 'нет')
            if not np.isnan(pval_c) else '-',
        })

    results = pd.DataFrame(records)

    print("\n" + "=" * 70)
    print("ПРОГНОЗНЫЕ РЕГРЕССИИ: MPU_pca -> RV(t+h)")
    print("=" * 70)
    print(results.to_string(index=False))

    return results


# ============================================================
# ШАГ 4: Out-of-sample тест
# ============================================================

def oos_test(merged: pd.DataFrame,
             horizons: list[int] | None = None,
             split_frac: float = 0.6) -> pd.DataFrame:
    """
    Out-of-sample тест прогнозной силы MPU_pca.

    Методология:
        - Выборка делится на in-sample (первые split_frac наблюдений)
          и out-of-sample (оставшиеся).
        - На in-sample обучаем: RV(t+h) = alpha + beta * MPU(t).
        - Прогнозируем RV на out-of-sample.
        - Считаем R2_oos = 1 - MSE_model / MSE_benchmark.
          Бенчмарк = среднее RV по in-sample (наивный прогноз).

    R2_oos > 0: модель лучше наивного прогноза -> MPU_pca полезен.
    R2_oos < 0: модель хуже наивного.

    Параметры:
        merged      — объединённый DataFrame
        horizons    — горизонты прогноза
        split_frac  — доля in-sample (по умолчанию 60%)
    """
    if horizons is None:
        horizons = [1, 2, 3, 6]

    n_split  = int(len(merged) * split_frac)
    records  = []

    print(f"\nOOS тест: in-sample = {n_split} obs, "
          f"out-of-sample = {len(merged) - n_split} obs")
    print(f"Split: до {merged['Date'].iloc[n_split-1].date()} | "
          f"после {merged['Date'].iloc[n_split].date()}")

    for h in horizons:
        rv_fwd = merged['RV'].shift(-h)
        mask   = rv_fwd.notna() & merged['MPU_pca'].notna()

        idx_is  = merged.index[mask & (merged.index < n_split)]
        idx_oos = merged.index[mask & (merged.index >= n_split)]

        if len(idx_is) < 8 or len(idx_oos) < 3:
            continue

        y_is  = rv_fwd.loc[idx_is].values
        x_is  = merged.loc[idx_is, 'MPU_pca'].values
        y_oos = rv_fwd.loc[idx_oos].values
        x_oos = merged.loc[idx_oos, 'MPU_pca'].values

        # In-sample обучение
        X_is   = sm.add_constant(x_is)
        model  = sm.OLS(y_is, X_is).fit()

        # OOS прогноз
        X_oos  = sm.add_constant(x_oos)
        y_pred = model.predict(X_oos)

        # Бенчмарк: среднее in-sample
        bench  = np.full_like(y_oos, y_is.mean())

        mse_model = np.mean((y_oos - y_pred) ** 2)
        mse_bench = np.mean((y_oos - bench) ** 2)
        r2_oos    = 1 - mse_model / mse_bench

        # RMSE
        rmse_model = np.sqrt(mse_model)
        rmse_bench = np.sqrt(mse_bench)

        records.append({
            'Горизонт h':    h,
            'N_oos':         len(idx_oos),
            'R2_oos':        round(r2_oos, 4),
            'RMSE_model':    round(rmse_model, 4),
            'RMSE_bench':    round(rmse_bench, 4),
            'Лучше бенчмарка': 'ДА' if r2_oos > 0 else 'нет',
        })

    oos_df = pd.DataFrame(records)
    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE ТЕСТ: MPU_pca -> RV(t+h)")
    print("=" * 60)
    print(oos_df.to_string(index=False))

    return oos_df


# ============================================================
# ШАГ 5: Тест Грэнджера
# ============================================================

def granger_test(merged: pd.DataFrame,
                 max_lag: int = 4) -> pd.DataFrame:
    """
    Тест Грэнджера: помогает ли история MPU_pca предсказать RV
    сверх собственной истории RV?

    H0: MPU_pca НЕ грэнджер-причинит RV (коэф. на лагах MPU = 0).
    H0 отвергается при p < 0.05 -> MPU_pca грэнджер-причинит RV.

    Важно: это статистическая предсказуемость, не экономическая
    причинность. Но для индикатора MPU это именно то, что нужно.

    Параметры:
        merged  — объединённый DataFrame
        max_lag — максимальный лаг (месяцев)
    """
    data = merged[['RV', 'MPU_pca']].dropna()

    print(f"\n{'='*60}")
    print("ТЕСТ ГРЭНДЖЕРА: MPU_pca -> RV")
    print(f"{'='*60}")
    print("H0: MPU_pca не грэнджер-причинит RV")
    print()

    results = grangercausalitytests(
        data[['RV', 'MPU_pca']],
        maxlag=max_lag,
        verbose=False
    )

    records = []
    for lag, res in results.items():
        # Берём F-тест (наиболее стандартный)
        f_stat = res[0]['ssr_ftest'][0]
        p_val  = res[0]['ssr_ftest'][1]
        records.append({
            'Лаг':          lag,
            'F-статистика': round(f_stat, 3),
            'p-value':      round(p_val, 4),
            'Отвергаем H0': 'ДА (MPU значим)' if p_val < 0.05 else 'нет',
        })
        print(f"  Лаг {lag}: F={f_stat:.3f}, p={p_val:.4f}  "
              f"{'-> MPU значим' if p_val < 0.05 else ''}")

    return pd.DataFrame(records)


# ============================================================
# ВИЗУАЛИЗАЦИИ
# ============================================================

def _save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_mpu_vs_rv(merged: pd.DataFrame) -> None:
    """
    График 1: MPU_pca и RV на одной временной оси.

    Визуальная проверка: предшествуют ли пики MPU пикам RV?
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # MPU_pca (нормированный)
    ax1.plot(merged['Date'], merged['MPU_pca_norm'],
             color='#6A1B9A', lw=2, label='MPU_pca (z-score)')
    ax1.axhline(0, color='grey', lw=0.8, ls=':')
    ax1.fill_between(merged['Date'], merged['MPU_pca_norm'],
                     where=merged['MPU_pca_norm'] > 0,
                     alpha=0.2, color='#6A1B9A',
                     label='Повышенная неопределённость')
    ax1.set_ylabel('MPU_pca (z-score)')
    ax1.set_title('MPU_pca и реализованная волатильность ключевой ставки',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')

    # RV ключевой ставки
    ax2.plot(merged['Date'], merged['RV'],
             color='#C62828', lw=2, label='RV ключевой ставки (3M)')
    ax2.fill_between(merged['Date'], merged['RV'],
                     alpha=0.2, color='#C62828')
    ax2.set_ylabel('RV, п.п.')
    ax2.set_xlabel('Дата')
    ax2.legend(loc='upper left')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    _save(fig, 'forecast_mpu_vs_rv.png')


def plot_scatter_mpu_rv(merged: pd.DataFrame,
                        horizons: list[int] | None = None) -> None:
    """
    График 2: Scatter MPU_pca(t) vs RV(t+h) для разных горизонтов.

    Положительный наклон линии тренда = MPU предсказывает будущую RV.
    """
    if horizons is None:
        horizons = [1, 2, 3, 6]

    n   = len(horizons)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        rv_fwd = merged['RV'].shift(-h)
        mask   = rv_fwd.notna() & merged['MPU_pca'].notna()
        x      = merged.loc[mask, 'MPU_pca'].values
        y      = rv_fwd[mask].values

        ax.scatter(x, y, alpha=0.6, color='#1565C0', s=40, zorder=5)

        # Линия тренда
        if len(x) > 3:
            z  = np.polyfit(x, y, 1)
            xf = np.linspace(x.min(), x.max(), 100)
            ax.plot(xf, np.polyval(z, xf),
                    color='#C62828', lw=2, label=f'Тренд (slope={z[0]:.3f})')

            # Корреляция
            rho, pval = pearsonr(x, y)
            ax.set_title(f'h = {h}M\nr = {rho:.3f}, p = {pval:.3f}',
                         fontweight='bold',
                         color='green' if pval < 0.05 else 'black')
        else:
            ax.set_title(f'h = {h}M (мало данных)')

        ax.set_xlabel('MPU_pca (текущий)')
        ax.set_ylabel(f'RV(t+{h}M)')
        ax.legend(fontsize=8)

    fig.suptitle('Прогнозная сила MPU_pca: scatter MPU(t) vs RV(t+h)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'forecast_scatter.png')


def plot_forecast_summary(reg_df: pd.DataFrame,
                          oos_df: pd.DataFrame) -> None:
    """
    График 3: Сводный график результатов тестирования.

    Левая панель:  beta и p-value по горизонтам (базовая модель).
    Правая панель: R2_oos по горизонтам.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Прогнозная сила MPU_pca для RV ключевой ставки',
                 fontsize=13, fontweight='bold')

    # -- Левая панель: beta и значимость ----------------------
    hs    = reg_df['Горизонт h'].values
    betas = pd.to_numeric(reg_df['beta (base)'], errors='coerce').values
    pvals = pd.to_numeric(reg_df['p-value (base)'], errors='coerce').values

    colors = ['#27ae60' if p < 0.05 else '#c0392b' for p in pvals]
    bars   = ax1.bar(hs, betas, color=colors, alpha=0.8,
                     edgecolor='white', width=0.6)
    ax1.axhline(0, color='grey', lw=1)

    for bar, b, p in zip(bars, betas, pvals):
        if not np.isnan(b):
            label = f'{b:.3f}\n{"*" if p < 0.05 else ""}'
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     b + (0.001 if b >= 0 else -0.002),
                     label, ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Горизонт h (месяцев)')
    ax1.set_ylabel('Коэффициент beta')
    ax1.set_title('Коэффициент beta (зелёный = p<0.05, красный = незначим)\n'
                  'Базовая регрессия: RV(t+h) = alpha + beta*MPU(t)')
    ax1.set_xticks(hs)

    # -- Правая панель: R2_oos --------------------------------
    if not oos_df.empty:
        hs_oos  = oos_df['Горизонт h'].values
        r2_oos  = oos_df['R2_oos'].values
        colors2 = ['#27ae60' if r > 0 else '#c0392b' for r in r2_oos]
        ax2.bar(hs_oos, r2_oos, color=colors2, alpha=0.8,
                edgecolor='white', width=0.6)
        ax2.axhline(0, color='grey', lw=1.5, ls='--',
                    label='R2_oos = 0 (бенчмарк)')
        for i, (h, r) in enumerate(zip(hs_oos, r2_oos)):
            ax2.text(h, r + (0.005 if r >= 0 else -0.01),
                     f'{r:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.set_xlabel('Горизонт h (месяцев)')
        ax2.set_ylabel('R2_oos')
        ax2.set_title('Out-of-sample R2\n'
                      'Зелёный = лучше бенчмарка, красный = хуже')
        ax2.set_xticks(hs_oos)
        ax2.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, 'forecast_summary.png')


def plot_oos_predictions(merged: pd.DataFrame,
                         horizon: int = 3,
                         split_frac: float = 0.6) -> None:
    """
    График 4: Фактическая RV vs прогноз модели (OOS период).

    Показывает насколько хорошо MPU_pca предсказывает RV
    на конкретном горизонте вне обучающей выборки.
    """
    rv_fwd  = merged['RV'].shift(-horizon)
    mask    = rv_fwd.notna() & merged['MPU_pca'].notna()
    n_split = int(mask.sum() * split_frac)

    idx_all = merged.index[mask]
    idx_is  = idx_all[:n_split]
    idx_oos = idx_all[n_split:]

    if len(idx_is) < 5 or len(idx_oos) < 3:
        print(f"  [SKIP] plot_oos_predictions: мало данных")
        return

    y_is  = rv_fwd.loc[idx_is].values
    x_is  = merged.loc[idx_is, 'MPU_pca'].values
    y_oos = rv_fwd.loc[idx_oos].values
    x_oos = merged.loc[idx_oos, 'MPU_pca'].values
    d_oos = merged.loc[idx_oos, 'Date'].values

    model  = sm.OLS(y_is, sm.add_constant(x_is)).fit()
    y_pred = model.predict(sm.add_constant(x_oos))
    bench  = np.full_like(y_oos, y_is.mean())

    mse_m  = np.mean((y_oos - y_pred) ** 2)
    mse_b  = np.mean((y_oos - bench) ** 2)
    r2_oos = 1 - mse_m / mse_b

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(d_oos, y_oos,  color='#1565C0', lw=2.5, label='RV (факт)')
    ax.plot(d_oos, y_pred, color='#C62828', lw=2,   ls='--',
            label=f'Прогноз MPU_pca (R2_oos={r2_oos:.3f})')
    ax.plot(d_oos, bench,  color='grey',    lw=1.5, ls=':',
            label=f'Бенчмарк (среднее IS = {bench[0]:.3f})')

    # Вертикальная граница IS/OOS
    split_date = merged.loc[idx_oos[0], 'Date']
    ax.axvline(split_date, color='black', lw=1.5, ls='-.',
               label=f'Начало OOS: {pd.Timestamp(split_date).date()}')

    ax.set_title(f'OOS-прогноз RV(t+{horizon}M) на основе MPU_pca(t)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Дата')
    ax.set_ylabel('RV, п.п.')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    _save(fig, f'forecast_oos_h{horizon}.png')


def plot_cross_correlogram(merged: pd.DataFrame,
                           max_lag: int = 8) -> None:
    """
    График 5: Кросс-корреляция MPU_pca(t) и RV(t+h).

    По оси X — лаг h (отрицательный = MPU опережает RV).
    Значимые корреляции выходят за пунктирные границы (+/- 2/sqrt(N)).

    Если корреляция при h > 0 выше, чем при h = 0 ->
    MPU является опережающим индикатором RV.
    """
    x = merged['MPU_pca'].dropna().values
    y = merged['RV'].dropna().values

    n     = min(len(x), len(y))
    x, y  = x[:n], y[:n]
    lags  = range(-max_lag, max_lag + 1)
    corrs = []

    for lag in lags:
        if lag < 0:
            r, _ = pearsonr(x[:n+lag], y[-lag:])
        elif lag > 0:
            r, _ = pearsonr(x[lag:], y[:n-lag])
        else:
            r, _ = pearsonr(x, y)
        corrs.append(r)

    sig_bound = 2 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = ['#27ae60' if abs(c) > sig_bound else '#aaa'
               for c in corrs]
    ax.bar(list(lags), corrs, color=colors, alpha=0.8, width=0.7)
    ax.axhline(0,           color='black', lw=0.8)
    ax.axhline(+sig_bound,  color='grey',  lw=1, ls='--',
               label=f'Граница значимости +/-{sig_bound:.2f}')
    ax.axhline(-sig_bound,  color='grey',  lw=1, ls='--')
    ax.axvline(0,           color='navy',  lw=1, ls=':', alpha=0.6)

    ax.set_xlabel('Лаг h (MPU опережает RV при h > 0)')
    ax.set_ylabel('Корреляция Пирсона')
    ax.set_title('Кросс-корреляция MPU_pca(t) и RV(t+h)\n'
                 'Зелёный = значим, серый = не значим',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(list(lags))
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, 'forecast_crosscorr.png')


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == '__main__':

    # ── 1. Загрузка MPU из файла ──────────────────────────────
    mpu_path = Path('results') / 'mpu_aggregated.csv'
    if not mpu_path.exists():
        raise FileNotFoundError(
            f"Файл {mpu_path} не найден. "
            f"Сначала запустите rnd_pipeline.py"
        )
    mpu_df = pd.read_csv(mpu_path, parse_dates=['Date'])
    print(f"MPU загружен: {len(mpu_df)} строк")
    print(mpu_df[['Date', 'MPU_pca', 'MPU_pca_norm']].head())

    # ── 2. Загрузка и расчёт RV ───────────────────────────────
    key_rate_df = get_key_rate_dataframe()
    rv_df       = compute_rv(key_rate_df, window=3)

    # ── 3. Объединение ────────────────────────────────────────
    merged = merge_mpu_rv(mpu_df, rv_df, rv_col='RV_3M')

    # ── 4. Прогнозные регрессии ───────────────────────────────
    HORIZONS = [1, 2, 3, 6]
    reg_df   = forecast_regression(merged, horizons=HORIZONS)

    # ── 5. OOS тест ───────────────────────────────────────────
    oos_df   = oos_test(merged, horizons=HORIZONS, split_frac=0.6)

    # ── 6. Тест Грэнджера ─────────────────────────────────────
    granger_df = granger_test(merged, max_lag=4)

    # ── 7. Сохранение таблиц ──────────────────────────────────
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)

    reg_df.to_csv(out_dir / 'forecast_regression.csv', index=False)
    oos_df.to_csv(out_dir / 'forecast_oos.csv', index=False)
    granger_df.to_csv(out_dir / 'forecast_granger.csv', index=False)
    print(f"\nТаблицы сохранены в {out_dir}/")

    # ── 8. Визуализация ───────────────────────────────────────
    print("\nГенерация графиков...")

    print("[1/5] MPU vs RV (временной ряд)...")
    plot_mpu_vs_rv(merged)

    print("[2/5] Scatter MPU(t) vs RV(t+h)...")
    plot_scatter_mpu_rv(merged, horizons=HORIZONS)

    print("[3/5] Сводный график (beta + R2_oos)...")
    plot_forecast_summary(reg_df, oos_df)

    print("[4/5] OOS прогноз (h=3)...")
    plot_oos_predictions(merged, horizon=3, split_frac=0.6)

    print("[5/5] Кросс-корреляция...")
    plot_cross_correlogram(merged, max_lag=8)

    print(f"\nГотово. Графики: {PLOT_DIR.resolve()}")