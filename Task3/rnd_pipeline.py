# rnd_pipeline.py
# Пайплайн извлечения RND и расчёта статистик для индекса MPU
# Кейс 2: Индекс неопределённости ДКП на основе поверхности вменённой волатильности
#
# Принцип: форвард F извлекается ТОЛЬКО из поверхности IV
#          (страйк с минимальной волатильностью = ATM-point улыбки).
#          Ключевая ставка ЦБ РФ используется ТОЛЬКО для
#          справочных линий на графиках — не для расчёта RND/MPU.

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Совместимость np.trapz → np.trapezoid (NumPy >= 2.0)
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# Импорт функций загрузки данных
from data_loading.case_2 import get_case_2_IV, get_key_rate_dataframe


# ============================================================
# НАСТРОЙКИ ГРАФИКОВ
# ============================================================

PLOT_DIR = Path('results') / 'plots'

COLORS = {
    '1M':        '#2196F3',
    '3M':        '#4CAF50',
    '6M':        '#FF9800',
    '1Y':        '#E91E63',
    'key_rate':  '#212121',
    'inflation': '#F44336',
    'iqr':       '#1565C0',
    'quantile':  '#6A1B9A',
}

plt.rcParams.update({
    'figure.dpi':      150,
    'font.size':       10,
    'axes.titlesize':  12,
    'axes.labelsize':  10,
    'axes.grid':       True,
    'grid.alpha':      0.3,
    'legend.fontsize': 9,
    'lines.linewidth': 1.8,
})


# ============================================================
# ШАГ 0: Загрузка данных
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загрузка исходных данных.

    Возвращает:
        iv_df       — поверхность вменённой волатильности
        key_rate_df — ключевая ставка и инфляция (только для графиков)
    """
    iv_df       = get_case_2_IV()
    key_rate_df = get_key_rate_dataframe()

    print("=" * 60)
    print("IV данные:")
    print(iv_df.head())
    print(f"Период: {iv_df['Date'].min().date()} → {iv_df['Date'].max().date()}")
    print(f"Дат наблюдений: {iv_df['Date'].nunique()}")
    print(f"Сроки экспирации: {sorted(iv_df['Maturity'].unique())}")
    print(f"Страйки: {sorted(iv_df['Strike'].unique())}")

    print("\nКлючевая ставка (только для графиков, не для расчёта RND):")
    print(key_rate_df.head(10))
    print(f"Период: {key_rate_df['Date'].min().date()} → {key_rate_df['Date'].max().date()}")
    print("=" * 60)

    return iv_df, key_rate_df


# ============================================================
# ШАГ 1: Форвард из поверхности IV
# ============================================================

def get_atm_forward(strikes: np.ndarray, ivols: np.ndarray) -> float:
    """
    Извлекает форвардную ставку F из улыбки волатильности.

    Метод: ATM-point = страйк с минимальной вменённой волатильностью.

    Обоснование:
        По теории безарбитражного ценообразования форвард соответствует
        минимуму улыбки IV — при этом страйке колл и пут стоят одинаково
        (put-call parity). Это стандартный подход для извлечения F
        из рыночных данных без использования внешних источников.

    Параметры:
        strikes — массив страйков (%)
        ivols   — массив вменённых волатильностей

    Возвращает:
        float — форвард F (%)
    """
    return float(strikes[np.argmin(ivols)])


# ============================================================
# ШАГ 2: Модель Башелье (нормальная модель)
# ============================================================

def bachelier_call(F: float, K: float,
                   T: float, sigma_n: float) -> float:
    """
    Цена колл-опциона по нормальной модели Башелье.

    Используется вместо Black-Scholes т.к. волатильность
    задана в абсолютных единицах (нормальная IV).

    Параметры:
        F       — форвард / ATM ставка (%)
        K       — страйк (%)
        T       — время до экспирации (лет)
        sigma_n — нормальная волатильность (% годовых)
    """
    if T <= 1e-10:
        return max(F - K, 0.0)
    denom = sigma_n * np.sqrt(T)
    if denom < 1e-10:
        return max(F - K, 0.0)
    d = (F - K) / denom
    return denom * (d * norm.cdf(d) + norm.pdf(d))


def bachelier_put(F: float, K: float,
                  T: float, sigma_n: float) -> float:
    """Цена пут-опциона через пут-колл паритет (модель Башелье)."""
    return bachelier_call(F, K, T, sigma_n) - (F - K)


# ============================================================
# ШАГ 3: Извлечение RND для одной даты и одного срока
# ============================================================

def extract_rnd(date: pd.Timestamp,
                iv_df: pd.DataFrame,
                maturity: str = '3M',
                n_points: int = 1000,
                verbose: bool = False) -> dict | None:
    """
    Извлекает RND методом Breeden-Litzenberger и считает статистики.

    Форвард F извлекается исключительно из улыбки IV (ATM-point).
    Внешние данные (ключевая ставка ЦБ) не используются.

    Алгоритм:
        1. Срез IV(K) для фиксированной даты и срока T
        2. F = страйк с минимальной IV (ATM-point улыбки)
        3. Интерполяция улыбки кубическим сплайном
        4. Цены коллов по Башелье на плотной сетке
        5. Вторая производная по страйку (Breeden-Litzenberger) → RND
        6. Моменты и квантили

    Параметры:
        date     — дата наблюдения
        iv_df    — датафрейм с поверхностью IV
        maturity — срок экспирации ('1M', '3M', '6M', '1Y', ...)
        n_points — число точек в плотной сетке страйков
        verbose  — печатать предупреждения

    Возвращает:
        dict со статистиками или None если данных недостаточно
    """
    mask     = (iv_df['Date'] == date) & (iv_df['Maturity'] == maturity)
    slice_df = iv_df[mask].sort_values('Strike').copy()

    if len(slice_df) < 5:
        if verbose:
            print(f"  [SKIP] Мало точек для {date.date()} {maturity}: "
                  f"{len(slice_df)}")
        return None

    strikes = slice_df['Strike'].values
    ivols   = slice_df['Volatility'].values
    T_val   = slice_df['Maturity (year fraction)'].iloc[0]

    # --- Форвард: ATM-point улыбки IV ---
    # Используем только данные поверхности IV, без внешних источников
    F = get_atm_forward(strikes, ivols)

    # --- Интерполяция улыбки кубическим сплайном ---
    cs     = CubicSpline(strikes, ivols, extrapolate=False)
    K_grid = np.linspace(strikes.min(), strikes.max(), n_points)
    dK     = K_grid[1] - K_grid[0]

    iv_grid = cs(K_grid)
    # Flat extrapolation на хвостах (NaN → граничные значения)
    iv_grid = np.where(
        np.isnan(iv_grid),
        np.where(K_grid < strikes[0], ivols[0], ivols[-1]),
        iv_grid
    )
    iv_grid = np.maximum(iv_grid, 0.01)     # IV строго положительна

    # --- Цены коллов по Башелье ---
    C_grid = np.array([
        bachelier_call(F, K, T_val, iv)
        for K, iv in zip(K_grid, iv_grid)
    ])

    # --- Breeden-Litzenberger: ∂²C/∂K² = RND ---
    RND   = np.gradient(np.gradient(C_grid, dK), dK)
    RND   = np.maximum(RND, 0.0)
    total = np.trapz(RND, K_grid)

    if total < 1e-10:
        if verbose:
            print(f"  [SKIP] Вырожденная RND для {date.date()} {maturity}")
        return None
    RND /= total

    # --- Моменты ---
    mean_rnd = np.trapz(K_grid * RND, K_grid)
    var_rnd  = np.trapz((K_grid - mean_rnd) ** 2 * RND, K_grid)
    std_rnd  = np.sqrt(max(var_rnd, 1e-10))
    skew     = (np.trapz((K_grid - mean_rnd) ** 3 * RND, K_grid)
                / std_rnd ** 3)
    kurt     = (np.trapz((K_grid - mean_rnd) ** 4 * RND, K_grid)
                / std_rnd ** 4)

    # --- Квантили ---
    cdf = np.cumsum(RND) * dK
    cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)

    def quantile(p: float) -> float:
        idx = np.searchsorted(cdf, p)
        return float(K_grid[np.clip(idx, 0, len(K_grid) - 1)])

    Q01 = quantile(0.01)
    Q10 = quantile(0.10)
    Q25 = quantile(0.25)
    Q50 = quantile(0.50)
    Q75 = quantile(0.75)
    Q90 = quantile(0.90)
    Q99 = quantile(0.99)

    return {
        'Date':       date,
        'Maturity':   maturity,
        'F':          F,           # ATM-форвард из улыбки IV
        'Mean':       mean_rnd,
        'Std':        std_rnd,
        'Skew':       skew,
        'Kurt':       kurt,
        'Q01':        Q01,
        'Q10':        Q10,
        'Q25':        Q25,
        'Q50':        Q50,
        'Q75':        Q75,
        'Q90':        Q90,
        'Q99':        Q99,
        'IQR_9010':   Q90 - Q10,
        'IQR_7525':   Q75 - Q25,
        'Tail_right': Q99 - Q90,
        'Tail_left':  Q10 - Q01,
        'Tail_total': (Q99 - Q90) + (Q10 - Q01),
    }


# ============================================================
# ШАГ 4: Прогон по всем датам и срокам
# ============================================================

def run_pipeline(iv_df: pd.DataFrame,
                 maturities: list[str] | None = None,
                 verbose: bool = False) -> pd.DataFrame:
    """
    Прогоняет extract_rnd по всем датам и заданным срокам.

    Параметры:
        iv_df      — датафрейм с поверхностью IV
        maturities — список сроков (по умолчанию ['1M','3M','6M','1Y'])
        verbose    — печатать предупреждения

    Возвращает:
        pd.DataFrame со статистиками для всех дат и сроков
    """
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']

    dates       = sorted(iv_df['Date'].unique())
    all_results = []

    for maturity in maturities:
        print(f"Обрабатываем срок: {maturity} ({len(dates)} дат)...")
        ok, skip = 0, 0

        for date in dates:
            result = extract_rnd(date, iv_df,
                                 maturity=maturity, verbose=verbose)
            if result is not None:
                all_results.append(result)
                ok += 1
            else:
                skip += 1

        print(f"  ✅ Успешно: {ok}  ⚠️  Пропущено: {skip}")

    results_df = pd.DataFrame(all_results)
    print(f"\nИтого наблюдений: {len(results_df)}")
    return results_df


# ============================================================
# ШАГ 5: Проверка качества RND
# ============================================================

def check_rnd_quality(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет флаги качества к каждому наблюдению.

    Критерии:
        1. |Mean - F| < 2%    — среднее близко к ATM-форварду
        2. Std ∈ [0.1%, 15%]  — разумный разброс
        3. IQR_9010 > 0        — Q90 > Q10
        4. Квантили монотонны  — Q10 < Q25 < Q50 < Q75 < Q90
    """
    df = results_df.copy()

    df['flag_mean_ok'] = (df['Mean'] - df['F']).abs() < 2.0
    df['flag_std_ok']  = df['Std'].between(0.1, 15.0)
    df['flag_iqr_ok']  = df['IQR_9010'] > 0
    df['flag_mono_ok'] = (
            (df['Q10'] < df['Q25']) &
            (df['Q25'] < df['Q50']) &
            (df['Q50'] < df['Q75']) &
            (df['Q75'] < df['Q90'])
    )

    df['quality_ok'] = (
            df['flag_mean_ok'] & df['flag_std_ok'] &
            df['flag_iqr_ok']  & df['flag_mono_ok']
    )

    n_bad = (~df['quality_ok']).sum()
    if n_bad > 0:
        print(f"⚠️  Проблемных наблюдений: {n_bad}")
        print(df[~df['quality_ok']][
                  ['Date', 'Maturity', 'F', 'Mean', 'Std', 'IQR_9010',
                   'flag_mean_ok', 'flag_std_ok', 'flag_iqr_ok', 'flag_mono_ok']
              ].to_string())
    else:
        print("✅ Все наблюдения прошли проверку качества")

    return df


# ============================================================
# ШАГ 6: Описательная статистика
# ============================================================

def print_summary(results_df: pd.DataFrame) -> None:
    """Описательная статистика по ключевым метрикам."""

    cols = ['IQR_9010', 'IQR_7525', 'Std', 'Skew', 'Kurt',
            'Tail_right', 'Tail_left', 'Tail_total']

    print("\n" + "=" * 60)
    print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА ПО СРОКАМ")
    print("=" * 60)

    for maturity, group in results_df.groupby('Maturity'):
        print(f"\n--- Срок: {maturity} (n={len(group)}) ---")
        print(group[cols].describe().round(4).to_string())

    print("\n" + "=" * 60)
    print("IQR_9010 (базовый MPU) по срокам:")
    pivot = results_df.pivot_table(
        values='IQR_9010', index='Date', columns='Maturity'
    )
    print(pivot.describe().round(4).to_string())


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ГРАФИКОВ
# ============================================================

def _save_or_show(fig: plt.Figure, save: bool, filename: str) -> None:
    """Сохраняет график в файл или показывает интерактивно."""
    if save:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOT_DIR / filename
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"  📊 Сохранён: {path}")
        plt.close(fig)
    else:
        plt.show()


def _compute_rnd_arrays(date: pd.Timestamp,
                        iv_df: pd.DataFrame,
                        maturity: str,
                        n_points: int = 1000):
    """
    Внутренняя утилита: возвращает (K_grid, RND, F, result).

    Форвард берётся из ATM-point улыбки IV.
    Используется визуальными функциями для избежания дублирования.
    """
    mask     = (iv_df['Date'] == date) & (iv_df['Maturity'] == maturity)
    slice_df = iv_df[mask].sort_values('Strike')
    if slice_df.empty:
        return None

    result = extract_rnd(date, iv_df, maturity=maturity)
    if result is None:
        return None

    strikes = slice_df['Strike'].values
    ivols   = slice_df['Volatility'].values
    T_val   = slice_df['Maturity (year fraction)'].iloc[0]
    F       = result['F']       # ATM из улыбки IV

    cs      = CubicSpline(strikes, ivols, extrapolate=False)
    K_grid  = np.linspace(strikes.min(), strikes.max(), n_points)
    dK      = K_grid[1] - K_grid[0]
    iv_grid = cs(K_grid)
    iv_grid = np.where(np.isnan(iv_grid),
                       np.interp(K_grid, strikes, ivols), iv_grid)
    iv_grid = np.maximum(iv_grid, 0.01)

    C_grid = np.array([bachelier_call(F, K, T_val, iv)
                       for K, iv in zip(K_grid, iv_grid)])
    RND    = np.gradient(np.gradient(C_grid, dK), dK)
    RND    = np.maximum(RND, 0.0)
    total  = np.trapz(RND, K_grid)
    if total > 1e-10:
        RND /= total

    return K_grid, RND, F, result


# ============================================================
# ВИЗУАЛИЗАЦИЯ 1: Улыбка волатильности
# ============================================================

def plot_iv_smile(date: pd.Timestamp,
                  iv_df: pd.DataFrame,
                  key_rate_df: pd.DataFrame | None = None,
                  maturities: list[str] | None = None,
                  save: bool = True) -> None:
    """
    Улыбка IV(K) для заданной даты.

    Рыночные точки + кубический сплайн + вертикаль ATM.

    Параметры:
        key_rate_df — опционально, только для подписи в заголовке.
                      Не используется в расчётах.

    Что читать:
        - Минимум улыбки = ATM (форвард F, извлечённый из IV)
        - Наклон вправо → дороже страховка от роста ставки
        - Кривизна (butterfly) → толщина хвостов
    """
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']

    n_mat = len(maturities)
    fig, axes = plt.subplots(1, n_mat, figsize=(5 * n_mat, 4))
    if n_mat == 1:
        axes = [axes]

    # Заголовок: если есть ключевая ставка — показываем справочно
    if key_rate_df is not None:
        past = key_rate_df[key_rate_df['Date'] <= date]
        kr_label = (f"  |  КС (справ.): "
                    f"{past.iloc[-1]['Key Rate']:.2f}%"
                    if not past.empty else "")
    else:
        kr_label = ""

    fig.suptitle(
        f'Улыбка вменённой волатильности  |  {date.date()}{kr_label}',
        fontsize=13, fontweight='bold'
    )

    for ax, mat in zip(axes, maturities):
        mask     = (iv_df['Date'] == date) & (iv_df['Maturity'] == mat)
        slice_df = iv_df[mask].sort_values('Strike')
        if slice_df.empty:
            ax.set_title(f'{mat} — нет данных')
            continue

        strikes = slice_df['Strike'].values
        ivols   = slice_df['Volatility'].values
        color   = COLORS.get(mat, '#333333')

        # ATM из улыбки IV
        F_atm = get_atm_forward(strikes, ivols)

        cs      = CubicSpline(strikes, ivols, extrapolate=False)
        K_fine  = np.linspace(strikes.min(), strikes.max(), 500)
        iv_fine = cs(K_fine)
        iv_fine = np.where(np.isnan(iv_fine),
                           np.interp(K_fine, strikes, ivols), iv_fine)

        ax.plot(K_fine, iv_fine, color=color, lw=2, label='Сплайн')
        ax.scatter(strikes, ivols, color=color, s=40, zorder=5,
                   label='Рынок', edgecolors='white', linewidths=0.5)
        ax.axvline(F_atm, color='black', lw=1.2, ls='--', alpha=0.7,
                   label=f'ATM={F_atm:.1f}%')

        ax.set_title(f'Срок: {mat}')
        ax.set_xlabel('Страйк (%)')
        ax.set_ylabel('Волатильность')
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save, f'iv_smile_{date.date()}.png')


# ============================================================
# ВИЗУАЛИЗАЦИЯ 2: RND для одной даты
# ============================================================

def plot_rnd(date: pd.Timestamp,
             iv_df: pd.DataFrame,
             key_rate_df: pd.DataFrame | None = None,
             maturities: list[str] | None = None,
             save: bool = True) -> None:
    """
    Плотность вероятности будущей ставки (RND) с квантилями.

    Вертикальная линия ATM — форвард из улыбки IV (не ключевая ставка).
    key_rate_df опционален — только для справочной подписи.

    Что читать:
        - Ширина кривой = неопределённость (MPU)
        - Закрашенная область [Q10, Q90] = 80% вероятности
        - Асимметрия вправо → рынок боится роста ставки
        - Толстые хвосты → риск экстремального сценария
    """
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']

    n_mat = len(maturities)
    fig, axes = plt.subplots(1, n_mat, figsize=(5 * n_mat, 4))
    if n_mat == 1:
        axes = [axes]

    if key_rate_df is not None:
        past = key_rate_df[key_rate_df['Date'] <= date]
        kr_label = (f"  |  КС (справ.): "
                    f"{past.iloc[-1]['Key Rate']:.2f}%"
                    if not past.empty else "")
    else:
        kr_label = ""

    fig.suptitle(
        f'Risk-Neutral Distribution (RND)  |  {date.date()}{kr_label}',
        fontsize=13, fontweight='bold'
    )

    for ax, mat in zip(axes, maturities):
        out = _compute_rnd_arrays(date, iv_df, mat)
        if out is None:
            ax.set_title(f'{mat} — нет данных')
            continue

        K_grid, RND, F_atm, result = out
        color         = COLORS.get(mat, '#1565C0')
        Q10, Q50, Q90 = result['Q10'], result['Q50'], result['Q90']

        mask_fill = (K_grid >= Q10) & (K_grid <= Q90)
        ax.fill_between(K_grid[mask_fill], RND[mask_fill],
                        alpha=0.25, color=color, label='Q10–Q90 (80%)')
        ax.plot(K_grid, RND, color=color, lw=2)

        y_top = RND.max()
        for q, lbl, ls in [(Q10, 'Q10', ':'), (Q50, 'Q50', '--'),
                           (Q90, 'Q90', ':')]:
            ax.axvline(q, color=color, lw=1.2, ls=ls, alpha=0.8)
            ax.text(q, y_top * 0.05, f'{lbl}\n{q:.1f}%',
                    ha='center', fontsize=7, color=color)

        # ATM из улыбки IV (не ключевая ставка)
        ax.axvline(F_atm, color='black', lw=1.5, ls='-', alpha=0.6,
                   label=f'ATM={F_atm:.1f}%')
        ax.set_title(
            f'{mat}  |  IQR={result["IQR_9010"]:.2f}%'
            f'  |  Skew={result["Skew"]:.2f}'
        )
        ax.set_xlabel('Ставка (%)')
        ax.set_ylabel('Плотность вероятности')
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save, f'rnd_{date.date()}.png')


# ============================================================
# ВИЗУАЛИЗАЦИЯ 3: Fan chart — эволюция RND во времени
# ============================================================

def plot_rnd_fan(clean_df: pd.DataFrame,
                 maturity: str = '3M',
                 key_rate_df: pd.DataFrame | None = None,
                 save: bool = True) -> None:
    """
    Fan chart: квантильные полосы RND во времени.

    Квантили рассчитаны из RND — только данные поверхности IV.
    Линия ключевой ставки добавляется опционально как справочная.

    Что читать:
        - Тёмная полоса [Q25, Q75] = 50% вероятности
        - Светлая полоса [Q10, Q90] = 80% вероятности
        - Расширение полос = рост MPU
        - Линия ATM (F) = медианный форвард из улыбки IV
        - Пунктир КС = фактическая ставка (справочно, не из RND)
    """
    df = clean_df[clean_df['Maturity'] == maturity].sort_values('Date')
    if df.empty:
        print(f"Нет данных для {maturity}")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(df['Date'], df['Q10'], df['Q90'],
                    alpha=0.20, color=COLORS['iqr'],
                    label='Q10–Q90 (80% вероятности)')
    ax.fill_between(df['Date'], df['Q25'], df['Q75'],
                    alpha=0.35, color=COLORS['iqr'],
                    label='Q25–Q75 (50% вероятности)')
    ax.plot(df['Date'], df['Q50'],
            color=COLORS['iqr'], lw=2, label='Q50 (медиана RND)')
    ax.plot(df['Date'], df['F'],
            color='navy', lw=1.5, ls=':', alpha=0.8,
            label='ATM-форвард из IV')

    # Ключевая ставка — только справочная линия, не участвует в расчёте
    if key_rate_df is not None:
        kr = key_rate_df[
            (key_rate_df['Date'] >= df['Date'].min()) &
            (key_rate_df['Date'] <= df['Date'].max())
            ]
        ax.plot(kr['Date'], kr['Key Rate'],
                color=COLORS['key_rate'], lw=2.5, ls='--',
                label='Ключевая ставка (справочно)')

    ax.set_title(
        f'Эволюция RND во времени (Fan Chart)  |  Срок: {maturity}\n'
        f'Квантили из RND (поверхность IV). '
        f'КС — справочно, не используется в расчёте.',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('Дата')
    ax.set_ylabel('Ставка (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.legend(loc='upper left')

    plt.tight_layout()
    _save_or_show(fig, save, f'rnd_fan_{maturity}.png')


# ============================================================
# ВИЗУАЛИЗАЦИЯ 4: MPU во времени
# ============================================================

def plot_mpu_timeseries(clean_df: pd.DataFrame,
                        key_rate_df: pd.DataFrame | None = None,
                        maturities: list[str] | None = None,
                        save: bool = True) -> None:
    """
    Временной ряд MPU = IQR(Q90−Q10) по всем срокам.

    Нижняя панель: ключевая ставка и инфляция (справочно).
    MPU рассчитан исключительно из поверхности IV.

    Что читать:
        - Пики MPU = периоды высокой неопределённости ДКП
        - Синхронный рост по всем срокам = системная неопределённость
        - Расхождение сроков = неопределённость локализована по горизонту
    """
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']

    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 1, figure=fig, hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    for mat in maturities:
        sub = clean_df[clean_df['Maturity'] == mat].sort_values('Date')
        if sub.empty:
            continue
        ax1.plot(sub['Date'], sub['IQR_9010'],
                 color=COLORS.get(mat, '#333'), lw=2,
                 label=f'MPU {mat}')

    ax1.set_title(
        'Индекс неопределённости ДКП  (MPU = Q90 − Q10)\n'
        'Рассчитан из поверхности IV без использования данных ЦБ',
        fontsize=12, fontweight='bold'
    )
    ax1.set_ylabel('IQR, п.п.')
    ax1.legend(ncol=len(maturities))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2 = fig.add_subplot(gs[1])
    if key_rate_df is not None:
        ref = clean_df[clean_df['Maturity'] == maturities[0]].sort_values('Date')
        if not ref.empty:
            kr = key_rate_df[
                (key_rate_df['Date'] >= ref['Date'].min()) &
                (key_rate_df['Date'] <= ref['Date'].max())
                ]
            ax2.plot(kr['Date'], kr['Key Rate'],
                     color=COLORS['key_rate'], lw=2,
                     label='Ключевая ставка (справочно)')
            if 'Inflation' in kr.columns:
                ax2.plot(kr['Date'], kr['Inflation'],
                         color=COLORS['inflation'], lw=1.5, ls='--',
                         label='Инфляция г/г (справочно)')

    ax2.set_title('Ключевая ставка и инфляция (справочно, не используются в MPU)',
                  fontsize=10)
    ax2.set_ylabel('%')
    ax2.set_xlabel('Дата')
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle('Индекс неопределённости ДКП | Обзор',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save, 'mpu_timeseries.png')


# ============================================================
# ВИЗУАЛИЗАЦИЯ 5: Моменты RND во времени
# ============================================================

def plot_moments_timeseries(clean_df: pd.DataFrame,
                            maturity: str = '3M',
                            save: bool = True) -> None:
    """
    Временные ряды моментов RND: Std, Skew, Kurt, Tail.

    Все метрики рассчитаны из RND (поверхность IV).

    Что читать:
        - Std    = общая ширина (общая неопределённость)
        - Skew>0 = рынок боится роста ставки
        - Kurt>3 = риск экстремального сценария
        - Tail   = хвостовой риск (компонент MPU_extended)
    """
    df = clean_df[clean_df['Maturity'] == maturity].sort_values('Date')
    if df.empty:
        print(f"Нет данных для {maturity}")
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f'Моменты RND во времени  |  Срок: {maturity}',
                 fontsize=14, fontweight='bold')

    panels = [
        ('Std',        'Стандартное отклонение (ширина неопределённости)', '#1565C0', None),
        ('Skew',       'Асимметрия  [Skew > 0 → риск роста ставки]',      '#6A1B9A', 0.0),
        ('Kurt',       'Эксцесс  [Kurt > 3 → толстые хвосты]',            '#E65100', 3.0),
        ('Tail_total', 'Хвостовой риск  (Q99−Q90) + (Q10−Q01)',           '#B71C1C', None),
    ]

    for ax, (col, title, color, hline) in zip(axes, panels):
        ax.plot(df['Date'], df[col], color=color, lw=2)
        ax.fill_between(df['Date'], df[col], alpha=0.15, color=color)
        if hline is not None:
            ax.axhline(hline, color='grey', lw=1, ls='--', alpha=0.6,
                       label=f'Граница = {hline}')
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(col)

    axes[-1].set_xlabel('Дата')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    _save_or_show(fig, save, f'moments_{maturity}.png')


# ============================================================
# ВИЗУАЛИЗАЦИЯ 6: Срочная структура неопределённости
# ============================================================

def plot_term_structure(clean_df: pd.DataFrame,
                        dates_highlight: list | None = None,
                        save: bool = True) -> None:
    """
    IQR_9010 как функция срока экспирации.

    Все данные из поверхности IV.

    Что читать:
        - Нормальная форма: неопределённость растёт с горизонтом
        - Инверсия: краткосрочная > долгосрочной → ждут скорого решения ЦБ
        - Горбатая: максимум на среднем горизонте
    """
    maturity_order = ['1M', '2M', '3M', '6M', '9M',
                      '1Y', '2Y', '3Y', '4Y', '5Y']
    available = [m for m in maturity_order
                 if m in clean_df['Maturity'].unique()]

    avg_iqr = (clean_df.groupby('Maturity')['IQR_9010']
               .mean().reindex(available))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Срочная структура неопределённости ДКП  '
                 '(из поверхности IV)',
                 fontsize=13, fontweight='bold')

    ax1.bar(available, avg_iqr.values,
            color=COLORS['iqr'], alpha=0.8, edgecolor='white', width=0.6)
    ax1.set_title('Средний MPU (IQR Q90−Q10) по срокам')
    ax1.set_xlabel('Срок экспирации')
    ax1.set_ylabel('IQR, п.п.')
    for i, (mat, val) in enumerate(zip(available, avg_iqr.values)):
        if not np.isnan(val):
            ax1.text(i, val + 0.05, f'{val:.2f}', ha='center', fontsize=8)

    if dates_highlight is None:
        all_dates = sorted(clean_df['Date'].unique())
        n = len(all_dates)
        dates_highlight = [all_dates[0], all_dates[n // 2], all_dates[-1]]

    cmap   = plt.cm.plasma
    colors = [cmap(i / max(len(dates_highlight) - 1, 1))
              for i in range(len(dates_highlight))]

    for date, color in zip(dates_highlight, colors):
        sub = (clean_df[clean_df['Date'] == date]
               .set_index('Maturity').reindex(available))
        ax2.plot(available, sub['IQR_9010'].values,
                 marker='o', color=color, lw=2,
                 label=pd.Timestamp(date).strftime('%Y-%m'))

    ax2.set_title('Срочная структура MPU в отдельные даты')
    ax2.set_xlabel('Срок экспирации')
    ax2.set_ylabel('IQR, п.п.')
    ax2.legend(title='Дата', fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save, 'term_structure.png')


# ============================================================
# ВИЗУАЛИЗАЦИЯ 7: Корреляционная матрица компонент MPU
# ============================================================

def plot_correlation_matrix(clean_df: pd.DataFrame,
                            maturity: str = '3M',
                            save: bool = True) -> None:
    """
    Корреляционная матрица компонент MPU.

    Все компоненты из RND (поверхность IV).

    Нужна для обоснования PCA-агрегации:
        - Высокая корреляция → PCA даст один доминирующий фактор
        - Низкая корреляция → компоненты несут уникальную информацию
    """
    df   = clean_df[clean_df['Maturity'] == maturity].copy()
    cols = ['IQR_9010', 'IQR_7525', 'Std', 'Skew',
            'Kurt', 'Tail_right', 'Tail_left', 'Tail_total']
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Корреляция Пирсона')

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val   = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.7 else 'black'
            ax.text(j, i, f'{val:.2f}',
                    ha='center', va='center', fontsize=8, color=color)

    ax.set_title(f'Корреляция компонент MPU  |  Срок: {maturity}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save, f'correlation_{maturity}.png')


# ============================================================
# ВИЗУАЛИЗАЦИЯ 8: Сравнение RND в разные даты
# ============================================================

def plot_rnd_comparison(dates_compare: list[pd.Timestamp],
                        iv_df: pd.DataFrame,
                        maturity: str = '3M',
                        save: bool = True) -> None:
    """
    Наложение RND нескольких дат на один график.

    Вертикальные линии — ATM-форвард из улыбки IV.
    Ключевая ставка ЦБ не используется.

    Что читать:
        - Широкая кривая = высокая неопределённость
        - Сдвиг вправо = рынок ожидает роста ставки
        - Асимметрия = направленный риск
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    cmap   = plt.cm.viridis
    colors = [cmap(i / max(len(dates_compare) - 1, 1))
              for i in range(len(dates_compare))]

    for date, color in zip(dates_compare, colors):
        out = _compute_rnd_arrays(date, iv_df, maturity)
        if out is None:
            continue

        K_grid, RND, F_atm, result = out
        label = (f"{pd.Timestamp(date).strftime('%Y-%m')}  "
                 f"IQR={result['IQR_9010']:.1f}%  "
                 f"ATM={F_atm:.1f}%")
        ax.plot(K_grid, RND, color=color, lw=2, label=label)
        ax.axvline(F_atm, color=color, lw=0.8, ls=':', alpha=0.5)

    ax.set_title(
        f'Сравнение RND в разные даты  |  Срок: {maturity}\n'
        f'Пунктир — ATM-форвард из улыбки IV',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel('Ставка (%)')
    ax.set_ylabel('Плотность вероятности')
    ax.legend(fontsize=9, title='Дата | IQR | ATM')

    plt.tight_layout()
    _save_or_show(fig, save, f'rnd_comparison_{maturity}.png')


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == '__main__':

    # 1. Загрузка данных
    iv_df, key_rate_df = load_data()

    # 2. Прогон пайплайна (только iv_df — ключевая ставка не нужна)
    results_df = run_pipeline(
        iv_df,
        maturities=['1M', '3M', '6M', '1Y'],
        verbose=True
    )

    # 3. Проверка качества
    results_df = check_rnd_quality(results_df)

    # 4. Фильтрация
    clean_df = results_df[results_df['quality_ok']].copy()
    print(f"\nПосле фильтрации: {len(clean_df)} наблюдений")

    # 5. Описательная статистика
    print_summary(clean_df)

    # 6. Сохранение CSV
    output_path = Path('results') / 'rnd_statistics.csv'
    output_path.parent.mkdir(exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    print(f"\nРезультаты сохранены: {output_path}")

    # ============================================================
    # 7. Визуализация
    # ============================================================
    print("\nГенерация графиков...")

    all_dates  = sorted(iv_df['Date'].unique())
    date_first = all_dates[0]
    date_mid   = all_dates[len(all_dates) // 2]
    date_last  = all_dates[-1]

    # Граф 1: Улыбка IV (ключевая ставка — только для подписи)
    print("\n[1/8] Улыбка волатильности...")
    plot_iv_smile(date_last, iv_df, key_rate_df=key_rate_df,
                  maturities=['1M', '3M', '6M', '1Y'])

    # Граф 2: RND на последнюю дату
    print("[2/8] RND на последнюю дату...")
    plot_rnd(date_last, iv_df, key_rate_df=key_rate_df,
             maturities=['1M', '3M', '6M', '1Y'])

    # Граф 3: Fan chart (3M)
    print("[3/8] Fan chart (3M)...")
    plot_rnd_fan(clean_df, maturity='3M', key_rate_df=key_rate_df)

    # Граф 4: MPU во времени
    print("[4/8] MPU временной ряд...")
    plot_mpu_timeseries(clean_df, key_rate_df=key_rate_df,
                        maturities=['1M', '3M', '6M', '1Y'])

    # Граф 5: Моменты (3M)
    print("[5/8] Моменты RND (3M)...")
    plot_moments_timeseries(clean_df, maturity='3M')

    # Граф 6: Срочная структура
    print("[6/8] Срочная структура...")
    plot_term_structure(clean_df,
                        dates_highlight=[date_first, date_mid, date_last])

    # Граф 7: Корреляционная матрица (3M)
    print("[7/8] Корреляционная матрица...")
    plot_correlation_matrix(clean_df, maturity='3M')

    # Граф 8: Сравнение RND (начало / середина / конец)
    print("[8/8] Сравнение RND...")
    plot_rnd_comparison(
        [date_first, date_mid, date_last],
        iv_df, maturity='3M'
    )

    print(f"\n✅ Все графики сохранены в: {PLOT_DIR.resolve()}")