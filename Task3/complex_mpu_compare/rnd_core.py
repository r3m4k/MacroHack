# rnd_core.py
# Базовые функции для извлечения RND из поверхности вменённой волатильности.
# Используется как зависимость в mpu_rnd.py и mpu_rwd.py.
#
# Принцип: форвард F извлекается ТОЛЬКО из поверхности IV (ATM-point).
# Внешние данные (ключевая ставка ЦБ) не используются в расчётах.

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

# Совместимость np.trapz -> np.trapezoid (NumPy >= 2.0)
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid


# ============================================================
# ATM-форвард из улыбки IV
# ============================================================

def get_atm_forward(strikes: np.ndarray, ivols: np.ndarray) -> float:
    """
    Форвард F = страйк с минимальной вменённой волатильностью.

    По теории безарбитражного ценообразования форвард соответствует
    минимуму улыбки IV: при этом страйке колл и пут равны по цене
    (put-call parity). Внешние данные не нужны.
    """
    return float(strikes[np.argmin(ivols)])


# ============================================================
# Модель Башелье
# ============================================================

def bachelier_call(F: float, K: float,
                   T: float, sigma_n: float) -> float:
    """
    Цена колл-опциона по нормальной модели Башелье.

    Используется вместо Black-Scholes: волатильность в данных
    задана в абсолютных единицах (нормальная IV, не логнормальная).

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


# ============================================================
# Извлечение RND методом Breeden-Litzenberger
# ============================================================

def extract_rnd(date: pd.Timestamp,
                iv_df: pd.DataFrame,
                maturity: str = '3M',
                n_points: int = 1000,
                verbose: bool = False) -> dict | None:
    """
    Извлекает RND методом Breeden-Litzenberger для одной даты и срока.

    Алгоритм:
        1. Срез IV(K) для даты и срока T
        2. F = ATM-point улыбки (min IV)
        3. Кубический сплайн по улыбке IV
        4. Цены коллов по Башелье на плотной сетке K
        5. d2C/dK2 -> RND (Breeden-Litzenberger 1978)
        6. Моменты и квантили распределения

    Параметры:
        date     — дата наблюдения
        iv_df    — DataFrame с колонками Date, Maturity, Strike, Volatility,
                   Maturity (year fraction)
        maturity — срок экспирации ('1M', '3M', '6M', '1Y', ...)
        n_points — число точек плотной сетки страйков
        verbose  — печатать предупреждения

    Возвращает:
        dict со статистиками или None если данных недостаточно.
        Ключи: Date, Maturity, F, Mean, Std, Skew, Kurt,
               Q01, Q10, Q25, Q50, Q75, Q90, Q99,
               IQR_9010, IQR_7525, Tail_right, Tail_left, Tail_total
    """
    mask     = (iv_df['Date'] == date) & (iv_df['Maturity'] == maturity)
    slice_df = iv_df[mask].sort_values('Strike').copy()

    if len(slice_df) < 5:
        if verbose:
            print(f"  [SKIP] {date.date()} {maturity}: "
                  f"мало точек ({len(slice_df)})")
        return None

    strikes = slice_df['Strike'].values
    ivols   = slice_df['Volatility'].values
    T_val   = slice_df['Maturity (year fraction)'].iloc[0]

    F = get_atm_forward(strikes, ivols)

    # Интерполяция улыбки кубическим сплайном
    cs      = CubicSpline(strikes, ivols, extrapolate=False)
    K_grid  = np.linspace(strikes.min(), strikes.max(), n_points)
    dK      = K_grid[1] - K_grid[0]
    iv_grid = cs(K_grid)

    # Flat extrapolation на хвостах
    iv_grid = np.where(
        np.isnan(iv_grid),
        np.where(K_grid < strikes[0], ivols[0], ivols[-1]),
        iv_grid
    )
    iv_grid = np.maximum(iv_grid, 0.01)

    # Цены коллов по Башелье
    C_grid = np.array([bachelier_call(F, K, T_val, iv)
                       for K, iv in zip(K_grid, iv_grid)])

    # Breeden-Litzenberger: RND = d2C/dK2
    RND   = np.gradient(np.gradient(C_grid, dK), dK)
    RND   = np.maximum(RND, 0.0)
    total = np.trapz(RND, K_grid)

    if total < 1e-10:
        if verbose:
            print(f"  [SKIP] {date.date()} {maturity}: вырожденная RND")
        return None
    RND /= total

    # Моменты
    mean_rnd = np.trapz(K_grid * RND, K_grid)
    var_rnd  = np.trapz((K_grid - mean_rnd) ** 2 * RND, K_grid)
    std_rnd  = np.sqrt(max(var_rnd, 1e-10))
    skew     = (np.trapz((K_grid - mean_rnd) ** 3 * RND, K_grid)
                / std_rnd ** 3)
    kurt     = (np.trapz((K_grid - mean_rnd) ** 4 * RND, K_grid)
                / std_rnd ** 4)

    # Квантили через CDF
    cdf = np.clip(np.cumsum(RND) * dK, 0.0, 1.0)
    cdf = cdf / cdf[-1]

    def q(p: float) -> float:
        idx = np.searchsorted(cdf, p)
        return float(K_grid[np.clip(idx, 0, len(K_grid) - 1)])

    Q01, Q10, Q25 = q(0.01), q(0.10), q(0.25)
    Q50, Q75, Q90 = q(0.50), q(0.75), q(0.90)
    Q99           = q(0.99)

    return {
        'Date':       date,      'Maturity':   maturity,
        'F':          F,         'Mean':       mean_rnd,
        'Std':        std_rnd,   'Skew':       skew,
        'Kurt':       kurt,
        'Q01':        Q01,       'Q10':        Q10,
        'Q25':        Q25,       'Q50':        Q50,
        'Q75':        Q75,       'Q90':        Q90,
        'Q99':        Q99,
        'IQR_9010':   Q90 - Q10,
        'IQR_7525':   Q75 - Q25,
        'Tail_right': Q99 - Q90,
        'Tail_left':  Q10 - Q01,
        'Tail_total': (Q99 - Q90) + (Q10 - Q01),
        # Сырые массивы для RWD и других вычислений
        '_K_grid':    K_grid,
        '_RND':       RND,
        '_iv_grid':   iv_grid,
    }


# ============================================================
# Прогон по всем датам и срокам
# ============================================================

def run_rnd_pipeline(iv_df: pd.DataFrame,
                     maturities: list[str] | None = None,
                     n_points: int = 1000,
                     verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Прогоняет extract_rnd по всем датам и срокам.

    Возвращает:
        stats_df  — DataFrame со статистиками (без сырых массивов)
        rnd_store — dict {(date, maturity): {'K_grid', 'RND', 'iv_grid'}}
    """
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']

    dates       = sorted(iv_df['Date'].unique())
    all_stats   = []
    rnd_store   = {}

    for maturity in maturities:
        print(f"  {maturity} ({len(dates)} дат)...", end=' ')
        ok = skip = 0

        for date in dates:
            result = extract_rnd(date, iv_df,
                                 maturity=maturity,
                                 n_points=n_points,
                                 verbose=verbose)
            if result is not None:
                key = (date, maturity)
                rnd_store[key] = {
                    'K_grid':  result.pop('_K_grid'),
                    'RND':     result.pop('_RND'),
                    'iv_grid': result.pop('_iv_grid'),
                }
                all_stats.append(result)
                ok += 1
            else:
                skip += 1

        print(f"OK={ok} skip={skip}")

    stats_df = pd.DataFrame(all_stats)
    return stats_df, rnd_store


# ============================================================
# Проверка качества RND
# ============================================================

def check_quality(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет флаги качества. Критерии:
        1. |Mean - F| < 2%   — среднее близко к ATM
        2. Std in [0.1, 15]% — разумный разброс
        3. IQR_9010 > 0
        4. Квантили монотонны
    """
    df = stats_df.copy()
    df['flag_mean'] = (df['Mean'] - df['F']).abs() < 2.0
    df['flag_std']  = df['Std'].between(0.1, 15.0)
    df['flag_iqr']  = df['IQR_9010'] > 0
    df['flag_mono'] = (
        (df['Q10'] < df['Q25']) & (df['Q25'] < df['Q50']) &
        (df['Q50'] < df['Q75']) & (df['Q75'] < df['Q90'])
    )
    df['quality_ok'] = (df['flag_mean'] & df['flag_std'] &
                        df['flag_iqr']  & df['flag_mono'])

    n_bad = (~df['quality_ok']).sum()
    if n_bad > 0:
        print(f"  Проблемных наблюдений: {n_bad}")
    else:
        print("  Все наблюдения прошли контроль качества")
    return df
