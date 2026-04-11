# mpu_rwd.py
# MPU на основе Real World Distribution (RWD).
#
# RND (risk-neutral) включает премию за риск, которую инвесторы
# требуют за неопределённость. RWD — «реальное» распределение
# без этой премии, отражающее истинные вероятности исходов.
#
# Метод перехода RND -> RWD: параметрическая корректировка
# через Esscher transform (сдвиг среднего на оценку премии за риск).
#
# Variance Risk Premium (VRP) = Var(RND) - Var(RWD) > 0 означает,
# что рынок переплачивает за страховку — это сама по себе мера
# неопределённости сверх «объективных» ожиданий.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from rnd_core import run_rnd_pipeline, check_quality


# ============================================================
# Переход RND -> RWD через Esscher transform
# ============================================================

def rnd_to_rwd(K_grid: np.ndarray,
               RND: np.ndarray,
               lambda_: float = 0.0) -> np.ndarray:
    """
    Конвертирует RND в RWD через Esscher transform.

    Esscher transform: q_rw(x) = q_rn(x) * exp(lambda * x) / Z
    где Z = integral q_rn(x) * exp(lambda * x) dx — нормировочная константа.

    lambda = 0: RWD = RND (нет поправки на риск)
    lambda < 0: RWD сдвинута влево относительно RND
                (реальное среднее ниже риск-нейтрального —
                 рынок требует премию за риск роста ставки)

    На практике lambda оценивается из исторических данных.
    Здесь используем lambda = -0.1 как консервативную оценку
    для процентных ставок (типичный диапазон: [-0.3, 0.0]).

    Параметры:
        K_grid  — сетка страйков
        RND     — risk-neutral плотность
        lambda_ — параметр Esscher (отрицательный = сдвиг влево)

    Возвращает:
        RWD — real world плотность (нормированная)
    """
    dK  = K_grid[1] - K_grid[0]
    w   = np.exp(lambda_ * K_grid)
    rwd = RND * w
    Z   = np.trapz(rwd, K_grid)
    if Z < 1e-10:
        return RND.copy()
    return rwd / Z


def compute_rwd_stats(K_grid: np.ndarray,
                      RWD: np.ndarray) -> dict:
    """Считает моменты и квантили RWD."""
    dK       = K_grid[1] - K_grid[0]
    mean_rwd = np.trapz(K_grid * RWD, K_grid)
    var_rwd  = np.trapz((K_grid - mean_rwd) ** 2 * RWD, K_grid)
    std_rwd  = np.sqrt(max(var_rwd, 1e-10))
    skew     = (np.trapz((K_grid - mean_rwd) ** 3 * RWD, K_grid)
                / std_rwd ** 3)
    kurt     = (np.trapz((K_grid - mean_rwd) ** 4 * RWD, K_grid)
                / std_rwd ** 4)

    cdf = np.clip(np.cumsum(RWD) * dK, 0.0, 1.0)
    cdf = cdf / cdf[-1]

    def q(p):
        idx = np.searchsorted(cdf, p)
        return float(K_grid[np.clip(idx, 0, len(K_grid) - 1)])

    return {
        'Mean_rwd':      mean_rwd,
        'Std_rwd':       std_rwd,
        'Skew_rwd':      skew,
        'Kurt_rwd':      kurt,
        'Q10_rwd':       q(0.10),
        'Q90_rwd':       q(0.90),
        'IQR_rwd':       q(0.90) - q(0.10),
        'Tail_right_rwd': q(0.99) - q(0.90),
        'Tail_left_rwd':  q(0.10) - q(0.01),
    }


# ============================================================
# Основная функция: MPU на основе RWD
# ============================================================

def build_mpu_rwd(iv_df: pd.DataFrame,
                  maturities: list[str] | None = None,
                  lambda_: float = -0.1,
                  verbose: bool = False) -> pd.DataFrame:
    """
    Строит MPU_rwd из Real World Distribution.

    Шаги:
        1. Извлечь RND для каждой даты и срока
        2. Применить Esscher transform: RND -> RWD
        3. Рассчитать IQR_rwd (Q90_rwd - Q10_rwd)
        4. Рассчитать VRP = Var(RND) - Var(RWD) (variance risk premium)
        5. Агрегировать IQR_rwd по срокам через PCA -> MPU_rwd
        6. Построить MPU_vrp из VRP

    Два итоговых индекса:
        MPU_rwd — неопределённость по «реальному» распределению
        MPU_vrp — превышение ожидаемой неопределённости над реальной
                  (рыночная премия за риск неопределённости)

    Параметры:
        iv_df      — поверхность IV
        maturities — сроки экспирации
        lambda_    — параметр Esscher (по умолчанию -0.1)
        verbose    — подробный вывод

    Возвращает:
        DataFrame: Date, MPU_rwd, MPU_vrp,
                   MPU_rwd_norm, MPU_vrp_norm
    """
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']

    print(f"RND пайплайн (MPU_rwd, lambda={lambda_}):")
    stats_df, rnd_store = run_rnd_pipeline(
        iv_df, maturities, verbose=verbose
    )
    stats_df = check_quality(stats_df)
    clean_df = stats_df[stats_df['quality_ok']].copy()

    # Считаем RWD-статистики для каждого наблюдения
    rwd_records = []
    for _, row in clean_df.iterrows():
        key = (row['Date'], row['Maturity'])
        if key not in rnd_store:
            continue

        K_grid = rnd_store[key]['K_grid']
        RND    = rnd_store[key]['RND']

        RWD       = rnd_to_rwd(K_grid, RND, lambda_=lambda_)
        rwd_stats = compute_rwd_stats(K_grid, RWD)

        # VRP = дисперсия RND - дисперсия RWD
        vrp = row['Std'] ** 2 - rwd_stats['Std_rwd'] ** 2

        rwd_records.append({
            'Date':     row['Date'],
            'Maturity': row['Maturity'],
            'VRP':      vrp,
            **rwd_stats,
        })

    if not rwd_records:
        raise ValueError("Не удалось рассчитать RWD ни для одного наблюдения")

    rwd_df = pd.DataFrame(rwd_records)
    print(f"  RWD рассчитан: {len(rwd_df)} наблюдений")

    # PCA по IQR_rwd нескольких сроков -> MPU_rwd
    pivot_rwd = (
        rwd_df[rwd_df['Maturity'].isin(maturities)]
        .pivot_table(values='IQR_rwd', index='Date', columns='Maturity')
        .reindex(columns=maturities)
        .dropna()
    )

    # PCA по VRP нескольких сроков -> MPU_vrp
    pivot_vrp = (
        rwd_df[rwd_df['Maturity'].isin(maturities)]
        .pivot_table(values='VRP', index='Date', columns='Maturity')
        .reindex(columns=maturities)
        .dropna()
    )

    def pca_first_component(pivot: pd.DataFrame,
                             ref_series: pd.Series | None = None
                             ) -> pd.Series:
        """PCA, первая компонента с правильной ориентацией."""
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(pivot.values)
        pca_fit = PCA(n_components=1)
        scores  = pca_fit.fit_transform(X_sc).ravel()
        pc1     = pd.Series(scores, index=pivot.index)
        if ref_series is not None and pc1.corr(ref_series) < 0:
            pc1 = -pc1
        ev = pca_fit.explained_variance_ratio_[0]
        return pc1, ev

    common_dates = pivot_rwd.index.intersection(pivot_vrp.index)
    pivot_rwd    = pivot_rwd.loc[common_dates]
    pivot_vrp    = pivot_vrp.loc[common_dates]

    mpu_rwd_raw, ev_rwd = pca_first_component(pivot_rwd)
    mpu_vrp_raw, ev_vrp = pca_first_component(
        pivot_vrp, ref_series=mpu_rwd_raw
    )

    def zscore(s):
        return (s - s.mean()) / s.std()

    result = pd.DataFrame({
        'Date':         common_dates,
        'MPU_rwd':      mpu_rwd_raw.values,
        'MPU_vrp':      mpu_vrp_raw.values,
        'MPU_rwd_norm': zscore(mpu_rwd_raw).values,
        'MPU_vrp_norm': zscore(mpu_vrp_raw).values,
    }).reset_index(drop=True)

    print(f"  MPU_rwd: PC1 объясняет {ev_rwd:.1%}")
    print(f"  MPU_vrp: PC1 объясняет {ev_vrp:.1%}")
    print(f"  Корреляция MPU_rwd <-> MPU_vrp: "
          f"{result['MPU_rwd'].corr(result['MPU_vrp']):.3f}")

    return result
