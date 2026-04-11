# mpu_tests.py
# Дополнительные тесты для сравнения MPU-индексов:
#   1. Событийный анализ (Event Study)
#   2. Тест на срочную структуру (Term Structure Test)
#   3. Rolling Window OOS
#
# Запуск через main.py или самостоятельно.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

PLOT_DIR   = Path('results') / 'plots'
RESULT_DIR = Path('results')

# Цвета для четырёх тестируемых индексов
INDEX_COLORS = {
    'MPU_decay':  '#E65100',
    'MPU_ext':    '#1565C0',
    'MPU_atm':    '#27AE60',
    'MPU_rv_std': '#8E44AD',
}

def _add_const(x: np.ndarray) -> np.ndarray:
    """
    Надёжная версия add_constant: работает корректно
    для массивов любой длины, включая одно наблюдение.
    """
    x = np.atleast_1d(x).reshape(-1)
    return np.column_stack([np.ones(len(x)), x])



def _save(fig: plt.Figure, name: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ============================================================
# ТЕСТ 1: Событийный анализ (Event Study)
# ============================================================

def detect_surprise_events(key_rate_df: pd.DataFrame,
                           threshold: float = 0.5
                           ) -> pd.DataFrame:
    """
    Определяет «неожиданные» решения ЦБ — заседания, на которых
    ставка изменилась на threshold п.п. или более.

    Логика: если рынок ожидал изменение (MPU высокий), то большое
    изменение ставки — ожидаемое. Нас интересуют случаи, когда
    ЦБ удивил рынок — это проверяется в самом событийном анализе
    (рос ли MPU ДО события).

    Параметры:
        key_rate_df — DataFrame с ['Date', 'Key Rate']
        threshold   — минимальное |изменение| для отбора события (п.п.)

    Возвращает:
        DataFrame с колонками Date, Key_Rate, Delta, event_type
    """
    df = (key_rate_df[['Date', 'Key Rate']]
          .sort_values('Date')
          .rename(columns={'Key Rate': 'Key_Rate'})
          .copy())
    df['Delta'] = df['Key_Rate'].diff()

    events = df[df['Delta'].abs() >= threshold].copy()
    # Убираем строки с NaT (могут появляться при diff на краях)
    events = events[events['Date'].notna()].copy()
    events['event_type'] = events['Delta'].apply(
        lambda d: 'hike' if d > 0 else 'cut'
    )

    print(f"\nОпределено событий (|Δ| >= {threshold} п.п.): {len(events)}")
    for _, row in events.iterrows():
        print(f"  {row['Date'].date()}  "
              f"{'▲' if row['event_type']=='hike' else '▼'} "
              f"{row['Delta']:+.2f} п.п.  "
              f"-> {row['Key_Rate']:.2f}%")

    return events.reset_index(drop=True)


def run_event_study(indices: dict[str, pd.Series],
                    key_rate_df: pd.DataFrame,
                    window_before: int = 4,
                    threshold: float = 0.5,
                    save: bool = True) -> pd.DataFrame:
    """
    Событийный анализ: динамика MPU вокруг решений ЦБ.

    Для каждого события (неожиданное изменение ставки) строим
    среднюю траекторию MPU в окне [-window_before, +2] месяцев.

    Гипотеза: если MPU — хороший опережающий индикатор,
    он должен расти ДО события и спадать ПОСЛЕ.

    Параметры:
        indices       — dict {name: pd.Series} с DatetimeIndex
        key_rate_df   — ключевая ставка
        window_before — окно наблюдений до события (месяцев)
        threshold     — порог изменения ставки для события
        save          — сохранять графики

    Возвращает:
        DataFrame со средними значениями MPU по окну для каждого индекса
    """
    print(f"\n{'='*60}")
    print("ТЕСТ 1: СОБЫТИЙНЫЙ АНАЛИЗ")
    print(f"Окно: [{-window_before}, +2] месяцев относительно события")
    print("=" * 60)

    events   = detect_surprise_events(key_rate_df, threshold=threshold)
    lags     = list(range(-window_before, 3))   # от -window_before до +2
    results  = {}

    for name, series in indices.items():
        series = series.sort_index().dropna()
        trajectories = []

        for _, event in events.iterrows():
            event_date = pd.Timestamp(event['Date'])
            traj = {}

            for lag in lags:
                # Ищем наблюдение MPU в lag месяцев от события
                target = event_date + pd.DateOffset(months=lag)

                # Ближайшее наблюдение в ±20 дней
                diffs = (series.index - target).days
                close = np.abs(diffs) <= 20
                if close.any():
                    traj[lag] = series[close].iloc[0]

            if len(traj) >= len(lags) // 2:   # не менее половины лагов
                trajectories.append(traj)

        if not trajectories:
            print(f"  [{name}] нет данных для событийного анализа")
            continue

        traj_df   = pd.DataFrame(trajectories)
        mean_traj = traj_df.mean()
        std_traj  = traj_df.std()
        n_events  = len(trajectories)

        results[name] = {
            'mean':     mean_traj,
            'std':      std_traj,
            'n_events': n_events,
        }

        print(f"  {name}: {n_events} событий, "
              f"MPU[-1]={mean_traj.get(-1, np.nan):.3f}, "
              f"MPU[0]={mean_traj.get(0, np.nan):.3f}, "
              f"MPU[+1]={mean_traj.get(1, np.nan):.3f}")

    # ── График ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Событийный анализ: MPU вокруг решений ЦБ РФ\n'
        f'(событие = изменение ставки ≥ {threshold} п.п., '
        f'n={len(events)})',
        fontsize=13, fontweight='bold'
    )

    # Левая панель: траектории всех индексов
    ax = axes[0]
    for name, res in results.items():
        color = INDEX_COLORS.get(name, '#333')
        mean  = res['mean']
        std   = res['std']
        n     = res['n_events']
        se    = std / np.sqrt(n)

        ax.plot(mean.index, mean.values,
                color=color, lw=2.5, marker='o',
                markersize=6, label=name)
        ax.fill_between(mean.index,
                        mean.values - 1.96 * se.values,
                        mean.values + 1.96 * se.values,
                        alpha=0.12, color=color)

    ax.axvline(0, color='black', lw=2, ls='--',
               label='Момент решения ЦБ')
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.set_xlabel('Месяцев до/после события')
    ax.set_ylabel('MPU (z-score)')
    ax.set_title('Средняя траектория MPU\n(полоса = 95% доверительный интервал)')
    ax.set_xticks(lags)
    ax.legend(fontsize=9)

    # Правая панель: только значения в ключевых точках [-2,-1,0,+1]
    ax2    = axes[1]
    key_lags = [-3, -2, -1, 0, 1, 2]
    x      = np.arange(len(key_lags))
    width  = 0.8 / len(results)

    for i, (name, res) in enumerate(results.items()):
        color  = INDEX_COLORS.get(name, '#333')
        vals   = [res['mean'].get(l, np.nan) for l in key_lags]
        offset = (i - len(results) / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width * 0.9,
                color=color, alpha=0.8, label=name)

    ax2.axhline(0, color='black', lw=1)
    ax2.axvline(key_lags.index(0) - 0.5 + 0.5,
                color='black', lw=1.5, ls='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f't{l:+d}' for l in key_lags])
    ax2.set_xlabel('Лаг (месяцев)')
    ax2.set_ylabel('Среднее MPU (z-score)')
    ax2.set_title('MPU по ключевым лагам\n(t=0: дата решения ЦБ)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save:
        _save(fig, 'test1_event_study.png')

    # Сводная таблица
    summary_rows = []
    for name, res in results.items():
        row = {'index': name, 'n_events': res['n_events']}
        for l in lags:
            row[f'lag_{l:+d}'] = round(res['mean'].get(l, np.nan), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULT_DIR / 'test1_event_study.csv', index=False)
    return summary_df


# ============================================================
# ТЕСТ 2: Срочная структура MPU
# ============================================================

def run_term_structure_test(iv_df: pd.DataFrame,
                            key_rate_df: pd.DataFrame,
                            save: bool = True) -> pd.DataFrame:
    """
    Тест на срочную структуру неопределённости.

    Проверяет две гипотезы:

    H1: Наклон кривой MPU(T) предсказывает направление изменения
        ставки — инверсия (коротко > длинно) сигнализирует о скором
        решении ЦБ.

    H2: Уровень краткосрочного MPU (1M) растёт перед заседаниями
        с изменением ставки (срочная структура как опережающий индикатор).

    Метод:
        - Для каждой даты строим IQR по срокам (1M, 3M, 6M, 1Y)
        - Считаем наклон: slope = IQR_1M - IQR_1Y (инверсия > 0)
        - Регрессия: |Δ_rate(t+h)| = α + β * slope(t) + ε

    Параметры:
        iv_df       — поверхность IV
        key_rate_df — ключевая ставка
        save        — сохранять графики

    Возвращает:
        DataFrame с результатами регрессий по горизонтам
    """
    from rnd_core import run_rnd_pipeline, check_quality

    print(f"\n{'='*60}")
    print("ТЕСТ 2: СРОЧНАЯ СТРУКТУРА MPU")
    print("=" * 60)

    maturities = ['1M', '3M', '6M', '1Y']

    # Строим IQR для всех сроков
    print("  Построение RND для срочной структуры...")
    stats_df, _ = run_rnd_pipeline(iv_df, maturities, verbose=False)
    stats_df    = check_quality(stats_df)
    clean_df    = stats_df[stats_df['quality_ok']].copy()

    pivot = (clean_df.pivot_table(
        values='IQR_9010', index='Date', columns='Maturity')
             .reindex(columns=maturities)
             .dropna())

    # Наклон = IQR_1M - IQR_1Y (инверсия если > 0)
    pivot['slope']     = pivot['1M'] - pivot['1Y']
    pivot['curvature'] = (pivot['1M'] + pivot['1Y']) / 2 - pivot['3M']

    # Объединяем с изменениями ставки
    kr = (key_rate_df[['Date', 'Key Rate']]
          .sort_values('Date')
          .rename(columns={'Key Rate': 'Key_Rate'})
          .set_index('Date'))
    kr['abs_change'] = kr['Key_Rate'].diff().abs()

    merged = pd.merge(
        pivot.reset_index(),
        kr.reset_index(),
        on='Date', how='inner'
    ).sort_values('Date').reset_index(drop=True)

    horizons = [1, 2, 3, 6, 12]
    records  = []

    print(f"\n  Регрессии slope -> |Δ_rate(t+h)|:")
    for h in horizons:
        y = merged['abs_change'].shift(-h).dropna()
        x = merged.loc[y.index, 'slope']
        mask = y.notna() & x.notna()
        y, x = y[mask].values, x[mask].values

        if len(y) < 8:
            continue

        res  = sm.OLS(y, sm.add_constant(x)).fit(
            cov_type='HAC', cov_kwds={'maxlags': h}
        )
        beta = res.params[1]
        pval = res.pvalues[1]
        r2   = res.rsquared

        records.append({
            'h':        h,
            'beta':     round(beta, 4),
            'p_value':  round(pval, 4),
            'R2':       round(r2,   4),
            'significant': pval < 0.05,
        })
        sig = '** ЗНАЧИМ **' if pval < 0.05 else ''
        print(f"    h={h:2d}M: beta={beta:.4f}  "
              f"p={pval:.4f}  R2={r2:.4f}  {sig}")

    results_df = pd.DataFrame(records)

    # ── Графики ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle('Тест 2: Срочная структура MPU (IQR_9010)',
                 fontsize=13, fontweight='bold')

    # 1. Срочная структура в динамике
    ax = axes[0]
    cmap   = plt.cm.plasma
    dates  = sorted(pivot.index)
    n_show = min(6, len(dates))
    idx    = np.linspace(0, len(dates) - 1, n_show, dtype=int)
    mat_order = ['1M', '3M', '6M', '1Y']
    mat_num   = [1/12, 3/12, 6/12, 1.0]

    for i in idx:
        d     = dates[i]
        color = cmap(i / max(len(dates) - 1, 1))
        vals  = [pivot.loc[d, m] for m in mat_order]
        ax.plot(mat_num, vals, color=color, lw=2,
                marker='o', markersize=6,
                label=pd.Timestamp(d).strftime('%Y-%m'))

    ax.set_xlabel('Срок (лет)')
    ax.set_ylabel('IQR Q90-Q10 (%)')
    ax.set_title('Срочная структура IQR\nв разные периоды')
    ax.set_xticks(mat_num)
    ax.set_xticklabels(mat_order)
    ax.legend(fontsize=7, title='Дата')

    # 2. Наклон кривой во времени + изменения ставки
    ax2 = axes[1]
    ax2.plot(merged['Date'], merged['slope'],
             color='#1565C0', lw=2, label='Наклон (1M - 1Y)')
    ax2.axhline(0, color='grey', lw=0.8, ls='--')
    ax2.fill_between(merged['Date'], merged['slope'],
                     where=merged['slope'] > 0,
                     alpha=0.2, color='#1565C0',
                     label='Инверсия (1M > 1Y)')

    ax2r = ax2.twinx()
    ax2r.bar(merged['Date'],
             merged['abs_change'].fillna(0),
             color='#C62828', alpha=0.4, width=20,
             label='|Δ ставки|')
    ax2r.set_ylabel('|Δ ставки|, п.п.', color='#C62828')
    ax2r.tick_params(axis='y', labelcolor='#C62828')

    ax2.set_title('Наклон кривой MPU и изменения ставки')
    ax2.set_ylabel('Наклон (п.п.)')
    ax2.set_xlabel('Дата')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.legend(fontsize=8, loc='upper left')

    # 3. beta наклона по горизонтам
    ax3 = axes[2]
    colors3 = ['#27ae60' if r['significant'] else '#c0392b'
               for _, r in results_df.iterrows()]
    bars = ax3.bar(results_df['h'], results_df['beta'],
                   color=colors3, alpha=0.8, edgecolor='white', width=0.8)
    ax3.axhline(0, color='black', lw=1)

    for bar, row in zip(bars, results_df.itertuples()):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 row.beta + 0.002 * np.sign(row.beta),
                 f'p={row.p_value:.2f}',
                 ha='center', va='bottom' if row.beta >= 0 else 'top',
                 fontsize=8)

    ax3.set_xlabel('Горизонт h (месяцев)')
    ax3.set_ylabel('Beta')
    ax3.set_title('Beta: slope -> |Δ_rate(t+h)|\n'
                  'Зелёный = p<0.05, красный = незначимо')
    ax3.set_xticks(horizons)

    plt.tight_layout()
    if save:
        _save(fig, 'test2_term_structure.png')

    results_df.to_csv(RESULT_DIR / 'test2_term_structure.csv', index=False)
    return results_df


# ============================================================
# ТЕСТ 3: Rolling Window OOS
# ============================================================

def run_rolling_oos(indices: dict[str, pd.Series],
                    key_rate_df: pd.DataFrame,
                    horizon: int = 3,
                    min_is_obs: int = 24,
                    horizons_all: list[int] | None = None,
                    save: bool = True) -> pd.DataFrame:
    """
    Rolling Window OOS: R2_oos на каждом возможном разбиении выборки.

    Вместо одного split 60/40 прогоняем OOS на каждом разбиении,
    начиная с min_is_obs наблюдений in-sample.

    Показывает:
        - Стабильна ли прогнозная сила во времени
        - Когда каждый индекс начинает работать
        - Нет ли концентрации результата в одном периоде (напр. 2022)

    Параметры:
        indices      — dict {name: pd.Series}
        key_rate_df  — ключевая ставка
        horizon      — основной горизонт для детального анализа
        min_is_obs   — минимальный размер in-sample (наблюдений)
        horizons_all — все горизонты для сводной таблицы
        save         — сохранять графики

    Возвращает:
        DataFrame с rolling R2_oos по датам для каждого индекса
    """
    from mpu_evaluator import compute_rv

    if horizons_all is None:
        horizons_all = [1, 2, 3, 6, 12]

    print(f"\n{'='*60}")
    print(f"ТЕСТ 3: ROLLING WINDOW OOS")
    print(f"Основной горизонт: h={horizon}M  |  min IS: {min_is_obs} obs")
    print("=" * 60)

    rv = compute_rv(key_rate_df, window=3)

    def build_merged(series: pd.Series) -> pd.DataFrame:
        mpu_df = pd.DataFrame({
            'Date': pd.to_datetime(series.index),
            'MPU':  series.values,
        }).dropna()
        rv_df = pd.DataFrame({
            'Date': pd.to_datetime(rv.index),
            'RV':   rv.values,
        }).dropna()
        return pd.merge_asof(
            mpu_df.sort_values('Date'),
            rv_df.sort_values('Date'),
            on='Date',
            tolerance=pd.Timedelta('15D'),
            direction='nearest'
        ).dropna().reset_index(drop=True)

    # ── Rolling R2_oos для основного горизонта ────────────────
    rolling_results = {}

    for name, series in indices.items():
        merged   = build_merged(series)
        rv_fwd   = merged['RV'].shift(-horizon)
        mask     = rv_fwd.notna() & merged['MPU'].notna()
        merged_h = merged[mask].copy().reset_index(drop=True)
        rv_fwd_h = rv_fwd[mask].reset_index(drop=True)

        n_total  = len(merged_h)
        r2_roll  = []
        dates_oos = []

        for split in range(min_is_obs, n_total - 3):
            y_is  = rv_fwd_h.iloc[:split].values
            x_is  = merged_h['MPU'].iloc[:split].values
            y_oos = rv_fwd_h.iloc[split:split+1].values
            x_oos = merged_h['MPU'].iloc[split:split+1].values

            model  = sm.OLS(y_is, _add_const(x_is)).fit()
            y_pred = model.predict(_add_const(x_oos))
            bench  = y_is.mean()

            mse_m  = (y_oos[0] - y_pred[0]) ** 2
            mse_b  = (y_oos[0] - bench)     ** 2
            r2_oos = 1.0 - mse_m / max(mse_b, 1e-10)

            r2_roll.append(r2_oos)
            dates_oos.append(merged_h['Date'].iloc[split])

        rolling_results[name] = pd.Series(r2_roll, index=dates_oos)
        # Сглаженное среднее
        smooth = pd.Series(r2_roll, index=dates_oos).rolling(
            6, min_periods=3
        ).mean()
        cumulative_r2 = np.mean(r2_roll)
        print(f"  {name}: cumulative R2_oos={cumulative_r2:+.3f}  "
              f"(h={horizon}M, n_oos={len(r2_roll)})")

    # ── Сводная таблица по всем горизонтам ────────────────────
    summary_records = []
    for name, series in indices.items():
        merged   = build_merged(series)
        row = {'index': name}

        for h in horizons_all:
            rv_fwd   = merged['RV'].shift(-h)
            mask     = rv_fwd.notna() & merged['MPU'].notna()
            merged_h = merged[mask].copy().reset_index(drop=True)
            rv_fwd_h = rv_fwd[mask].reset_index(drop=True)
            n_total  = len(merged_h)

            r2_vals = []
            for split in range(min_is_obs, n_total - 3):
                y_is  = rv_fwd_h.iloc[:split].values
                x_is  = merged_h['MPU'].iloc[:split].values
                y_oos = rv_fwd_h.iloc[split:split+1].values
                x_oos = merged_h['MPU'].iloc[split:split+1].values
                m     = sm.OLS(y_is, _add_const(x_is)).fit()
                y_p   = m.predict(_add_const(x_oos))
                mse_m = (y_oos[0] - y_p[0]) ** 2
                mse_b = (y_oos[0] - y_is.mean()) ** 2
                r2_vals.append(1.0 - mse_m / max(mse_b, 1e-10))

            row[f'h{h}M'] = round(np.mean(r2_vals), 4) if r2_vals else np.nan

        summary_records.append(row)

    summary_df = pd.DataFrame(summary_records)
    print(f"\n  Сводная таблица cumulative R2_oos по горизонтам:")
    print(f"  {'Индекс':<14}", end='')
    for h in horizons_all:
        print(f"  h={h:2d}M", end='')
    print()
    for _, row in summary_df.iterrows():
        print(f"  {row['index']:<14}", end='')
        for h in horizons_all:
            val = row[f'h{h}M']
            print(f"  {val:+.3f}", end='')
        print()

    # ── Графики ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Тест 3: Rolling Window OOS  (h={horizon}M)\n'
        f'Каждая точка = R2_oos для данного момента разбиения',
        fontsize=13, fontweight='bold'
    )

    # Левая: сырые rolling R2_oos
    ax = axes[0]
    for name, series in rolling_results.items():
        color  = INDEX_COLORS.get(name, '#333')
        smooth = series.rolling(6, min_periods=3).mean()
        ax.plot(series.index, series.values,
                color=color, alpha=0.2, lw=1)
        ax.plot(smooth.index, smooth.values,
                color=color, lw=2.5, label=f'{name}')

    ax.axhline(0, color='black', lw=1.5, ls='--',
               label='Бенчмарк (R2=0)')
    ax.fill_between(
        [series.index.min() for series in rolling_results.values()][0:1] +
        [series.index.max() for series in rolling_results.values()][0:1],
        [0, 0], [10, 10],
        alpha=0.04, color='green'
    )
    ax.set_xlabel('Дата (начало OOS периода)')
    ax.set_ylabel('Rolling R2_oos (сглажено, окно=6)')
    ax.set_title('Стабильность прогнозной силы во времени\n'
                 'Тонкая линия = raw, жирная = сглаженная')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Правая: сводная таблица по горизонтам (тепловая карта)
    ax2   = axes[1]
    names = summary_df['index'].tolist()
    h_labels = [f'h={h}M' for h in horizons_all]
    data  = summary_df[[f'h{h}M' for h in horizons_all]].values

    # Нормируем для цвета: зелёный > 0, красный < 0
    vmax = max(abs(data[~np.isnan(data)]).max(), 0.01)
    im   = ax2.imshow(data, cmap='RdYlGn',
                      vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax2, label='Cumulative R2_oos')

    ax2.set_xticks(range(len(h_labels)))
    ax2.set_yticks(range(len(names)))
    ax2.set_xticklabels(h_labels, fontsize=10)
    ax2.set_yticklabels(names, fontsize=10)

    for i in range(len(names)):
        for j in range(len(h_labels)):
            val   = data[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax2.text(j, i, f'{val:+.3f}',
                     ha='center', va='center',
                     fontsize=9, fontweight='bold', color=color)

    ax2.set_title('Cumulative Rolling R2_oos\nпо горизонтам')

    plt.tight_layout()
    if save:
        _save(fig, f'test3_rolling_oos_h{horizon}.png')

    summary_df.to_csv(RESULT_DIR / 'test3_rolling_oos.csv', index=False)
    return summary_df


# ============================================================
# Запуск всех трёх тестов
# ============================================================

def run_all_tests(indices_subset: dict[str, pd.Series],
                  iv_df: pd.DataFrame,
                  key_rate_df: pd.DataFrame,
                  horizons_all: list[int] | None = None,
                  save: bool = True) -> None:
    """
    Запускает все три теста для переданного набора индексов.

    Параметры:
        indices_subset — dict с отобранными индексами для тестирования
        iv_df          — поверхность IV (нужна для теста 2)
        key_rate_df    — ключевая ставка
        horizons_all   — горизонты для rolling OOS
        save           — сохранять графики
    """
    if horizons_all is None:
        horizons_all = [1, 2, 3, 6, 12]

    print(f"\n{'#'*60}")
    print("ЗАПУСК ДОПОЛНИТЕЛЬНЫХ ТЕСТОВ")
    print(f"Индексы: {list(indices_subset.keys())}")
    print(f"Горизонты: {horizons_all}")
    print('#' * 60)

    RESULT_DIR.mkdir(exist_ok=True)

    # Тест 1: Событийный анализ
    event_df = run_event_study(
        indices_subset, key_rate_df,
        window_before=4, threshold=0.5, save=save
    )

    # Тест 2: Срочная структура
    ts_df = run_term_structure_test(
        iv_df, key_rate_df, save=save
    )

    # Тест 3: Rolling OOS
    rolling_df = run_rolling_oos(
        indices_subset, key_rate_df,
        horizon=3,
        min_is_obs=24,
        horizons_all=horizons_all,
        save=save
    )

    print(f"\n{'='*60}")
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print(f"Результаты: {RESULT_DIR.resolve()}")
    print("=" * 60)

    return event_df, ts_df, rolling_df