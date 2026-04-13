# mpu_iv_surface.py
# MPU напрямую из поверхности вменённой волатильности.
#
# Идея: зачем восстанавливать RND, если информация уже есть
# в самой улыбке IV? Три стандартных рыночных показателя:
#
#   ATM IV      — вменённая волатильность при текущей ставке.
#                 Прямая цена неопределённости на рынке опционов.
#
#   Risk Reversal (RR) — наклон улыбки: IV(правый хвост) - IV(левый хвост).
#                 RR > 0: рынок больше боится роста ставки.
#                 RR < 0: рынок больше боится снижения.
#
#   Butterfly (BF) — кривизна улыбки: среднее хвостов - ATM IV.
#                 Мера толщины хвостов без пересчёта в RND.
#
# Преимущество над RND-методами: нет модельного риска (не нужна
# формула Башелье и предположения о форме распределения).
# Ответ на вопрос задания: «а нужно ли вообще считать RND?»

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from rnd_core import get_atm_forward


# ============================================================
# Расчёт ATM IV, Risk Reversal, Butterfly для одной даты/срока
# ============================================================

def compute_iv_surface_metrics(date: pd.Timestamp,
                                iv_df: pd.DataFrame,
                                maturity: str,
                                delta_wing: float = 0.25
                                ) -> dict | None:
    """
    Рассчитывает ATM IV, Risk Reversal и Butterfly из улыбки IV.

    Параметры:
        date       — дата наблюдения
        iv_df      — поверхность IV
        maturity   — срок экспирации
        delta_wing — отступ крыльев от ATM в долях диапазона страйков
                     (0.25 = 25% от расстояния ATM до края)

    Возвращает:
        dict с полями: Date, Maturity, F, ATM_IV, RR, BF,
                       IV_left_wing, IV_right_wing
    """
    mask     = (iv_df['Date'] == date) & (iv_df['Maturity'] == maturity)
    slice_df = iv_df[mask].sort_values('Strike')

    if len(slice_df) < 5:
        return None

    strikes = slice_df['Strike'].values
    ivols   = slice_df['Volatility'].values

    F = get_atm_forward(strikes, ivols)

    # Интерполяция улыбки
    cs = CubicSpline(strikes, ivols, extrapolate=False)

    def safe_iv(k: float) -> float:
        """IV в точке k, с fallback на линейную интерполяцию."""
        v = float(cs(k))
        if np.isnan(v):
            v = float(np.interp(k, strikes, ivols))
        return max(v, 0.01)

    atm_iv = safe_iv(F)

    # Крылья: отступ delta_wing от ATM в обе стороны
    left_range  = F - strikes.min()
    right_range = strikes.max() - F

    K_left  = F - delta_wing * left_range
    K_right = F + delta_wing * right_range

    K_left  = max(K_left,  strikes.min())
    K_right = min(K_right, strikes.max())

    iv_left  = safe_iv(K_left)
    iv_right = safe_iv(K_right)

    # Risk Reversal: наклон улыбки
    rr = iv_right - iv_left

    # Butterfly: кривизна (хвосты дороже ATM при BF > 0)
    bf = 0.5 * (iv_left + iv_right) - atm_iv

    return {
        'Date':          date,
        'Maturity':      maturity,
        'F':             F,
        'ATM_IV':        atm_iv,
        'IV_left_wing':  iv_left,
        'IV_right_wing': iv_right,
        'RR':            rr,
        'BF':            bf,
    }


# ============================================================
# Основная функция: MPU из поверхности IV
# ============================================================

def build_mpu_iv_surface(iv_df: pd.DataFrame,
                          maturities: list[str] | None = None,
                          delta_wing: float = 0.25,
                          verbose: bool = False) -> pd.DataFrame:
    """
    Строит MPU напрямую из поверхности IV без пересчёта в RND.

    Три итоговых индекса (агрегация через PCA по срокам):
        MPU_atm — из ATM IV (уровень неопределённости)
        MPU_rr  — из Risk Reversal (асимметрия рисков)
        MPU_bf  — из Butterfly (хвостовой риск из IV)

    Параметры:
        iv_df      — поверхность IV
        maturities — сроки для агрегации
        delta_wing — ширина крыльев для RR и BF
        verbose    — подробный вывод

    Возвращает:
        DataFrame: Date, ATM_IV_{mat}, RR_{mat}, BF_{mat} (по каждому сроку),
                   MPU_atm, MPU_rr, MPU_bf,
                   MPU_atm_norm, MPU_rr_norm, MPU_bf_norm
    """
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']

    dates = sorted(iv_df['Date'].unique())
    print(f"IV Surface пайплайн ({len(dates)} дат, "
          f"сроки={maturities}):")

    # Рассчитываем метрики для каждой даты и срока
    records = []
    for mat in maturities:
        ok = skip = 0
        for date in dates:
            res = compute_iv_surface_metrics(
                date, iv_df, mat, delta_wing=delta_wing
            )
            if res is not None:
                records.append(res)
                ok += 1
            else:
                skip += 1
        print(f"  {mat}: OK={ok} skip={skip}")

    if not records:
        raise ValueError("Не удалось рассчитать метрики IV Surface")

    raw_df = pd.DataFrame(records)

    # Pivot для каждой метрики: строки = даты, столбцы = сроки
    def pivot_metric(metric: str) -> pd.DataFrame:
        return (
            raw_df.pivot_table(values=metric,
                                index='Date', columns='Maturity')
            .reindex(columns=maturities)
            .dropna()
        )

    pivot_atm = pivot_metric('ATM_IV')
    pivot_rr  = pivot_metric('RR')
    pivot_bf  = pivot_metric('BF')

    # Переименуем колонки с суффиксом срока для финального DataFrame
    def rename_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return df.rename(columns={m: f'{prefix}_{m}' for m in maturities})

    # PCA по каждой метрике -> один индекс
    def pca_index(pivot: pd.DataFrame,
                  ref_positive: bool = True) -> tuple[pd.Series, float]:
        """
        Первая главная компонента.
        ref_positive=True: PC1 ориентирована положительно (коррел. со средним).
        """
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(pivot.values)
        pca_fit = PCA(n_components=1)
        scores  = pca_fit.fit_transform(X_sc).ravel()
        pc1     = pd.Series(scores, index=pivot.index)
        ev      = pca_fit.explained_variance_ratio_[0]

        if ref_positive:
            mean_series = pivot.mean(axis=1)
            if pc1.corr(mean_series) < 0:
                pc1 = -pc1

        return pc1, ev

    common = (pivot_atm.index
              .intersection(pivot_rr.index)
              .intersection(pivot_bf.index))

    pivot_atm = pivot_atm.loc[common]
    pivot_rr  = pivot_rr.loc[common]
    pivot_bf  = pivot_bf.loc[common]

    mpu_atm, ev_atm = pca_index(pivot_atm)
    mpu_rr,  ev_rr  = pca_index(pivot_rr)
    mpu_bf,  ev_bf  = pca_index(pivot_bf)

    def zscore(s):
        return (s - s.mean()) / s.std()

    # Финальный DataFrame
    result = pd.DataFrame({'Date': common})

    # Добавляем сырые метрики по каждому сроку
    for mat in maturities:
        result[f'ATM_IV_{mat}'] = pivot_atm[mat].values
        result[f'RR_{mat}']     = pivot_rr[mat].values
        result[f'BF_{mat}']     = pivot_bf[mat].values

    result['MPU_atm']      = mpu_atm.values
    result['MPU_rr']       = mpu_rr.values
    result['MPU_bf']       = mpu_bf.values
    result['MPU_atm_norm'] = zscore(mpu_atm).values
    result['MPU_rr_norm']  = zscore(mpu_rr).values
    result['MPU_bf_norm']  = zscore(mpu_bf).values

    print(f"\n  ATM IV PCA: PC1={ev_atm:.1%}")
    print(f"  RR PCA:    PC1={ev_rr:.1%}")
    print(f"  BF PCA:    PC1={ev_bf:.1%}")
    print(f"  Итого дат: {len(result)}")

    return result
