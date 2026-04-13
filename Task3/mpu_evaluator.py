# mpu_evaluator.py
# Универсальный модуль оценки прогнозной силы MPU-индексов.
#
# Использование:
#   from mpu_evaluator import MPUEvaluator
#
#   evaluator = MPUEvaluator(key_rate_df)
#
#   # Оценить один индекс
#   result = evaluator.evaluate(mpu_series, name='MPU_pca')
#
#   # Сравнить несколько индексов
#   summary = evaluator.compare({
#       'MPU_pca':   series_pca,
#       'MPU_decay': series_decay,
#       'ATM_IV':    series_atm,
#   })
#
#   # Сгенерировать все графики
#   evaluator.report(horizon=3)
#
# Критерий сравнения: R2_oos = 1 - MSE_model / MSE_benchmark
#   > 0 : индекс лучше наивного прогноза (среднего in-sample RV)
#   Лучший индекс = максимальный R2_oos

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

PLOT_DIR   = Path('results') / 'plots'
RESULT_DIR = Path('results')


# ============================================================
# РАСЧЁТ РЕАЛИЗОВАННОЙ ВОЛАТИЛЬНОСТИ
# ============================================================

def compute_rv(key_rate_df: pd.DataFrame,
               window: int = 3) -> pd.Series:
    """
    Реализованная волатильность ключевой ставки.

    RV_t = std(delta_r_{t-w+1}, ..., delta_r_t)
    где delta_r_t = r_t - r_{t-1} — ежемесячное изменение ставки.

    Параметры:
        key_rate_df — DataFrame с колонками ['Date', 'Key Rate']
        window      — скользящее окно (месяцев)

    Возвращает:
        pd.Series с DatetimeIndex
    """
    df = (key_rate_df[['Date', 'Key Rate']]
          .sort_values('Date')
          .rename(columns={'Key Rate': 'Key_Rate'})
          .set_index('Date'))

    delta    = df['Key_Rate'].diff()
    rv       = delta.rolling(window=window, min_periods=window).std()
    rv.name  = f'RV_{window}M'
    return rv


# ============================================================
# КЛАСС-ОЦЕНЩИК
# ============================================================

class MPUEvaluator:
    """
    Универсальный оценщик прогнозной силы MPU-индексов.

    Единственный критерий сравнения: R2_oos.

    R2_oos = 1 - MSE_model / MSE_benchmark
        > 0 : индекс лучше наивного прогноза
        = 1 : идеальный прогноз
        < 0 : хуже среднего — индекс не работает

    Параметры:
        key_rate_df — DataFrame с ключевой ставкой ЦБ РФ
        rv_window   — окно расчёта RV (месяцев, по умолчанию 3)
        horizons    — горизонты прогноза в месяцах
        split_frac  — доля in-sample наблюдений (по умолчанию 0.6)
    """

    def __init__(self,
                 key_rate_df: pd.DataFrame,
                 rv_window:   int       = 3,
                 horizons:    list[int] = None,
                 split_frac:  float     = 0.6):

        self.rv_window  = rv_window
        self.horizons   = horizons or [1, 2, 3, 6]
        self.split_frac = split_frac
        self._results   = {}
        self._summary   = None

        # RV считается один раз для всех индексов
        self.rv = compute_rv(key_rate_df, window=rv_window)

        n = self.rv.dropna().shape[0]
        print(f"MPUEvaluator готов:")
        print(f"  RV окно:   {rv_window}M")
        print(f"  Горизонты: {self.horizons}")
        print(f"  Split:     {split_frac:.0%} IS / "
              f"{1 - split_frac:.0%} OOS")
        print(f"  RV obs:    {n}  |  "
              f"mean={self.rv.mean():.3f}  "
              f"std={self.rv.std():.3f}  "
              f"max={self.rv.max():.3f}")

    # ----------------------------------------------------------
    # evaluate(): оценить один MPU
    # ----------------------------------------------------------

    def evaluate(self,
                 mpu:  pd.Series,
                 name: str = 'MPU') -> dict[str, Any] | None:
        """
        Оценивает прогнозную силу одного MPU-индекса.

        Алгоритм для каждого горизонта h:
            1. Формируем пары (MPU(t), RV(t+h))
            2. Делим на IS / OOS по split_frac
            3. OLS: RV(t+h) = alpha + beta * MPU(t)  [обучение на IS]
            4. Прогнозируем RV на OOS
            5. R2_oos = 1 - MSE_model / MSE_benchmark
               Бенчмарк = среднее RV по IS

        Параметры:
            mpu  — pd.Series с DatetimeIndex
            name — название индекса

        Возвращает:
            dict с ключами: name, scores, merged, models, split_idx
        """
        mpu    = self._to_datetime_series(mpu, name)
        merged = self._merge(mpu)

        if len(merged) < 15:
            print(f"  [{name}] слишком мало данных: {len(merged)}")
            return None

        n_split = int(len(merged) * self.split_frac)
        records = []
        models  = {}

        for h in self.horizons:
            rv_fwd = merged['RV'].shift(-h)
            mask   = rv_fwd.notna() & merged['MPU'].notna()

            idx_all = merged.index[mask]
            idx_is  = idx_all[idx_all < n_split]
            idx_oos = idx_all[idx_all >= n_split]

            if len(idx_is) < 8 or len(idx_oos) < 3:
                continue

            y_is  = rv_fwd.loc[idx_is].values
            x_is  = merged.loc[idx_is,  'MPU'].values
            y_oos = rv_fwd.loc[idx_oos].values
            x_oos = merged.loc[idx_oos, 'MPU'].values

            # Обучение на IS
            X_is  = sm.add_constant(x_is)
            model = sm.OLS(y_is, X_is).fit(
                cov_type='HAC', cov_kwds={'maxlags': h}
            )
            models[h] = model

            # Прогноз на OOS
            y_pred = model.predict(sm.add_constant(x_oos))
            bench  = np.full_like(y_oos, y_is.mean())

            mse_m  = np.mean((y_oos - y_pred) ** 2)
            mse_b  = np.mean((y_oos - bench)  ** 2)
            r2_oos = 1.0 - mse_m / mse_b

            records.append({
                'name':       name,
                'h':          h,
                'N_is':       len(idx_is),
                'N_oos':      len(idx_oos),
                'beta':       round(model.params[1], 4),
                'p_value':    round(model.pvalues[1], 4),
                'R2_is':      round(model.rsquared,  4),
                'R2_oos':     round(r2_oos,           4),
                'RMSE_model': round(np.sqrt(mse_m),   4),
                'RMSE_bench': round(np.sqrt(mse_b),   4),
                'beats_bench': r2_oos > 0,
            })

        if not records:
            print(f"  [{name}] не удалось рассчитать ни одного горизонта")
            return None

        scores = pd.DataFrame(records)
        print(f"  {name}:  " +
              "  ".join(f"h={r['h']}M R2_oos={r['R2_oos']:+.3f}"
                        for _, r in scores.iterrows()))

        return {
            'name':      name,
            'scores':    scores,
            'merged':    merged,
            'models':    models,
            'split_idx': n_split,
        }

    # ----------------------------------------------------------
    # compare(): сравнить несколько MPU
    # ----------------------------------------------------------

    def compare(self,
                mpu_dict:  dict[str, pd.Series],
                save_csv:  bool = True) -> pd.DataFrame:
        """
        Оценивает и сравнивает несколько MPU-индексов.

        Параметры:
            mpu_dict — {название: pd.Series}
            save_csv — сохранить сводную таблицу в CSV

        Возвращает:
            pd.DataFrame: строки = горизонты h,
                          столбцы = R2_oos каждого индекса
        """
        print(f"\n{'='*60}")
        print(f"СРАВНЕНИЕ {len(mpu_dict)} ИНДЕКСОВ MPU")
        print("=" * 60)

        results = {}
        for name, series in mpu_dict.items():
            res = self.evaluate(series, name=name)
            if res is not None:
                results[name] = res

        if not results:
            raise ValueError("Ни один индекс не оценён успешно.")

        # Сводная таблица R2_oos
        r2_frames = []
        for name, res in results.items():
            sub = (res['scores'][['h', 'R2_oos', 'beats_bench']]
                   .rename(columns={
                'R2_oos':     f'R2_oos_{name}',
                'beats_bench': f'beats_{name}',
            })
                   .set_index('h'))
            r2_frames.append(sub)

        summary = pd.concat(r2_frames, axis=1)

        r2_cols = [c for c in summary.columns if c.startswith('R2_oos_')]
        summary['best'] = (summary[r2_cols]
                           .idxmax(axis=1)
                           .str.replace('R2_oos_', '', regex=False))

        print(f"\n{'='*60}")
        print("СВОДНАЯ ТАБЛИЦА R2_oos")
        print("=" * 60)
        print(summary.to_string())
        print("\nЛучший индекс на каждом горизонте:")
        for h, row in summary.iterrows():
            best = row['best']
            val  = row[f'R2_oos_{best}']
            print(f"  h={h}M  ->  {best}  R2_oos={val:+.4f}")

        if save_csv:
            RESULT_DIR.mkdir(exist_ok=True)
            path = RESULT_DIR / 'mpu_comparison.csv'
            summary.to_csv(path)
            print(f"\nСохранено: {path}")

        self._results = results
        self._summary = summary
        return summary

    # ----------------------------------------------------------
    # Визуализации
    # ----------------------------------------------------------

    def plot_r2_oos_comparison(self,
                               results:  dict = None,
                               save:     bool = True) -> None:
        """
        Столбчатая диаграмма R2_oos по горизонтам для всех индексов.

        Каждая панель = один горизонт h.
        Зелёный = R2_oos > 0, полупрозрачный = R2_oos < 0.
        """
        results  = results or self._results
        horizons = self.horizons
        names    = list(results.keys())
        n_h      = len(horizons)
        n_idx    = len(names)

        fig, axes = plt.subplots(1, n_h, figsize=(4.5 * n_h, 6),
                                 sharey=False)
        if n_h == 1:
            axes = [axes]

        fig.suptitle(
            f'Сравнение MPU: R2_oos по горизонтам\n'
            f'OOS = последние {100*(1-self.split_frac):.0f}% '
            f'выборки  |  RV окно {self.rv_window}M',
            fontsize=13, fontweight='bold'
        )

        cmap   = plt.cm.tab10
        colors = [cmap(i) for i in range(n_idx)]

        for ax, h in zip(axes, horizons):
            r2_vals = []
            for name in names:
                sc  = results[name]['scores']
                row = sc[sc['h'] == h]
                r2_vals.append(
                    row['R2_oos'].values[0] if not row.empty else np.nan
                )

            bar_c = []
            for r, c in zip(r2_vals, colors):
                if np.isnan(r):
                    bar_c.append((*c[:3], 0.2))
                elif r > 0:
                    bar_c.append(c)
                else:
                    bar_c.append((*c[:3], 0.35))

            bars = ax.bar(range(n_idx), r2_vals,
                          color=bar_c, edgecolor='white', width=0.6)
            ax.axhline(0, color='black', lw=1.5, ls='--',
                       label='Бенчмарк')

            for bar, val in zip(bars, r2_vals):
                if not np.isnan(val):
                    offset = 0.005 if val >= 0 else -0.005
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + offset,
                        f'{val:+.3f}',
                        ha='center',
                        va='bottom' if val >= 0 else 'top',
                        fontsize=9, fontweight='bold'
                    )

            ax.set_xticks(range(n_idx))
            ax.set_xticklabels(names, rotation=25,
                               ha='right', fontsize=9)
            ax.set_title(f'h = {h}M', fontweight='bold')
            if h == horizons[0]:
                ax.set_ylabel('R2_oos')
                ax.legend(fontsize=8)

        plt.tight_layout()
        self._save(fig, 'mpu_r2oos_comparison.png', save)

    def plot_r2_oos_by_horizon(self,
                               results: dict = None,
                               save:    bool = True) -> None:
        """
        Линейный график R2_oos как функции горизонта h.

        Пересечения линий показывают смену лидера по горизонтам.
        """
        results = results or self._results

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap    = plt.cm.tab10
        colors  = [cmap(i) for i in range(len(results))]

        for (name, res), color in zip(results.items(), colors):
            sc = res['scores'].sort_values('h')
            ax.plot(sc['h'], sc['R2_oos'],
                    color=color, lw=2.5, marker='o',
                    markersize=8, label=name)
            for _, row in sc.iterrows():
                ax.annotate(f"{row['R2_oos']:+.3f}",
                            (row['h'], row['R2_oos']),
                            textcoords='offset points',
                            xytext=(6, 4), fontsize=8, color=color)

        ax.axhline(0, color='black', lw=1.5, ls='--',
                   label='Бенчмарк (R2_oos = 0)')

        # Закрашиваем зону выше бенчмарка
        y_top = max(
            res['scores']['R2_oos'].max()
            for res in results.values()
        ) * 1.15
        if y_top > 0:
            ax.axhspan(0, y_top, alpha=0.04, color='green')

        ax.set_xlabel('Горизонт h (месяцев)')
        ax.set_ylabel('R2_oos')
        ax.set_xticks(self.horizons)
        ax.set_title(
            'R2_oos как функция горизонта прогноза\n'
            'Лучший индекс = максимальный R2_oos',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=10)
        plt.tight_layout()
        self._save(fig, 'mpu_r2oos_by_horizon.png', save)

    def plot_oos_timeseries(self,
                            results: dict = None,
                            horizon: int  = 3,
                            save:    bool = True) -> None:
        """
        Временной ряд: факт RV vs прогнозы всех MPU (OOS период).

        Показывает когда каждый индекс ошибается,
        а когда точно попадает в направление движения RV.
        """
        results = results or self._results

        fig, ax = plt.subplots(figsize=(14, 6))
        cmap    = plt.cm.tab10
        colors  = [cmap(i) for i in range(len(results))]

        # Факт RV в OOS периоде
        first_res = next(iter(results.values()))
        merged    = first_res['merged']
        n_split   = first_res['split_idx']
        rv_fwd    = merged['RV'].shift(-horizon)
        mask_oos  = rv_fwd.notna() & (merged.index >= n_split)

        ax.plot(merged.loc[mask_oos, 'Date'].values,
                rv_fwd[mask_oos].values,
                color='black', lw=2.5, zorder=10, label='RV (факт)')

        # Прогнозы каждого индекса
        for (name, res), color in zip(results.items(), colors):
            sc  = res['scores']
            row = sc[sc['h'] == horizon]
            if row.empty or horizon not in res['models']:
                continue

            m      = res['merged']
            rv_f   = m['RV'].shift(-horizon)
            m_oos  = rv_f.notna() & (m.index >= res['split_idx'])
            x_oos  = m.loc[m_oos, 'MPU'].values
            y_pred = res['models'][horizon].predict(
                sm.add_constant(x_oos)
            )
            r2 = row['R2_oos'].values[0]

            ax.plot(m.loc[m_oos, 'Date'].values,
                    y_pred,
                    color=color, lw=2, ls='--',
                    label=f'{name}  R2_oos={r2:+.3f}')

        # Граница IS/OOS
        split_idx  = min(n_split, len(merged) - 1)
        split_date = merged.iloc[split_idx]['Date']
        ax.axvline(pd.Timestamp(split_date),
                   color='grey', lw=1.5, ls='-.',
                   label=f'Начало OOS: {pd.Timestamp(split_date).date()}')

        ax.set_title(
            f'OOS прогноз RV(t+{horizon}M): факт vs MPU-индексы',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('Дата')
        ax.set_ylabel('RV, п.п.')
        ax.legend(fontsize=9, loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        self._save(fig, f'mpu_oos_timeseries_h{horizon}.png', save)

    def plot_scatter_grid(self,
                          results: dict = None,
                          horizon: int  = 3,
                          save:    bool = True) -> None:
        """
        Scatter MPU(t) vs RV(t+h) для каждого индекса.

        Треугольники = OOS наблюдения, кружки = IS.
        Наклон и R2_oos в заголовке каждой панели.
        """
        results = results or self._results

        names = list(results.keys())
        n     = len(names)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        cmap   = plt.cm.tab10
        colors = [cmap(i) for i in range(n)]

        fig.suptitle(
            f'Scatter MPU(t) vs RV(t+{horizon}M) — все индексы\n'
            f'Треугольник = OOS, кружок = IS',
            fontsize=13, fontweight='bold'
        )

        for ax, (name, res), color in zip(axes, results.items(), colors):
            merged  = res['merged']
            rv_fwd  = merged['RV'].shift(-horizon)
            mask    = rv_fwd.notna() & merged['MPU'].notna()
            n_split = res['split_idx']

            idx_is  = merged.index[mask & (merged.index < n_split)]
            idx_oos = merged.index[mask & (merged.index >= n_split)]

            ax.scatter(merged.loc[idx_is,  'MPU'],
                       rv_fwd[idx_is],
                       alpha=0.45, color=color, s=35,
                       marker='o', label='IS')
            ax.scatter(merged.loc[idx_oos, 'MPU'],
                       rv_fwd[idx_oos],
                       alpha=0.85, color=color, s=55,
                       marker='^', label='OOS')

            # Линия тренда по всей выборке
            x_all = merged.loc[mask, 'MPU'].values
            y_all = rv_fwd[mask].values
            if len(x_all) > 3:
                z  = np.polyfit(x_all, y_all, 1)
                xf = np.linspace(x_all.min(), x_all.max(), 100)
                ax.plot(xf, np.polyval(z, xf),
                        color='black', lw=2,
                        label=f'slope={z[0]:.3f}')

            sc  = res['scores']
            row = sc[sc['h'] == horizon]
            r2  = row['R2_oos'].values[0] if not row.empty else np.nan

            ax.set_title(
                f'{name}\nR2_oos = {r2:+.3f}',
                fontweight='bold',
                color='#27ae60' if (not np.isnan(r2) and r2 > 0)
                else '#c0392b'
            )
            ax.set_xlabel('MPU')
            ax.set_ylabel(f'RV(t+{horizon}M)')
            ax.legend(fontsize=7)

        plt.tight_layout()
        self._save(fig, f'mpu_scatter_h{horizon}.png', save)

    # ----------------------------------------------------------
    # report(): всё сразу
    # ----------------------------------------------------------

    def report(self,
               results: dict = None,
               horizon: int  = 3,
               save:    bool = True) -> None:
        """
        Генерирует все 4 графика сравнения одной командой.

        Параметры:
            results — dict из compare() (если None — берёт последние)
            horizon — основной горизонт для детальных графиков
            save    — сохранять в файл
        """
        results = results or self._results
        if not results:
            print("Нет результатов. Вызовите compare() сначала.")
            return

        print("\nГенерация отчёта...")
        print("[1/4] Столбцы R2_oos по индексам...")
        self.plot_r2_oos_comparison(results, save=save)

        print("[2/4] Линии R2_oos по горизонтам...")
        self.plot_r2_oos_by_horizon(results, save=save)

        print(f"[3/4] OOS временной ряд (h={horizon}M)...")
        self.plot_oos_timeseries(results, horizon=horizon, save=save)

        print(f"[4/4] Scatter MPU vs RV (h={horizon}M)...")
        self.plot_scatter_grid(results, horizon=horizon, save=save)

        print(f"\nГотово. Графики: {PLOT_DIR.resolve()}")

    # ----------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------

    @staticmethod
    def _to_datetime_series(mpu: pd.Series, name: str) -> pd.Series:
        """Приводит MPU к pd.Series с DatetimeIndex."""
        s = mpu.copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        s.name = 'MPU'
        return s.sort_index()

    def _merge(self, mpu: pd.Series) -> pd.DataFrame:
        """Объединяет MPU и RV по ближайшей дате (+-15 дней)."""
        # MPU: reset_index даёт две колонки [дата, значение]
        mpu_reset = mpu.reset_index()
        mpu_reset.columns = ['Date', 'MPU']
        mpu_reset['Date'] = pd.to_datetime(mpu_reset['Date'])
        mpu_reset = mpu_reset.dropna(subset=['Date'])

        # RV: аналогично
        rv_reset = self.rv.reset_index()
        rv_reset.columns = ['Date', 'RV']
        rv_reset['Date'] = pd.to_datetime(rv_reset['Date'])
        rv_reset = rv_reset.dropna(subset=['Date', 'RV'])

        merged = pd.merge_asof(
            mpu_reset.sort_values('Date'),
            rv_reset.sort_values('Date'),
            on='Date',
            tolerance=pd.Timedelta('15D'),
            direction='nearest'
        ).dropna(subset=['MPU', 'RV'])

        return merged.reset_index(drop=True)

    @staticmethod
    def _save(fig: plt.Figure, name: str, save: bool) -> None:
        if save:
            PLOT_DIR.mkdir(parents=True, exist_ok=True)
            path = PLOT_DIR / name
            fig.savefig(path, bbox_inches='tight', dpi=150)
            print(f"  saved: {path}")
            plt.close(fig)
        else:
            plt.show()


# ============================================================
# ДЕМОНСТРАЦИЯ
# ============================================================

if __name__ == '__main__':
    from data_loading.case_2 import get_key_rate_dataframe

    key_rate_df = get_key_rate_dataframe()

    mpu_path = Path('results') / 'mpu_aggregated.csv'
    if not mpu_path.exists():
        raise FileNotFoundError(
            f"{mpu_path} не найден. Сначала запустите rnd_pipeline.py"
        )

    mpu_df = pd.read_csv(mpu_path, parse_dates=['Date'])

    def to_series(df, col):
        return df.set_index('Date')[col].sort_index()

    # Инициализация
    evaluator = MPUEvaluator(
        key_rate_df = key_rate_df,
        rv_window   = 3,
        horizons    = [1, 2, 3, 6],
        split_frac  = 0.6,
    )

    # Сравнение двух индексов из rnd_pipeline.py
    summary = evaluator.compare({
        'MPU_pca':   to_series(mpu_df, 'MPU_pca'),
        'MPU_decay': to_series(mpu_df, 'MPU_decay'),
    })

    # Все графики
    evaluator.report(horizon=3)