# mpu_rv.py
# MPU на основе реализованной волатильности (RV) ключевой ставки.
#
# Логика: если рынок ожидал высокую неопределённость в прошлом
# и она реализовалась, это информативно для текущего MPU.
# Реализованная волатильность — самый простой и прозрачный
# индикатор фактической неопределённости ДКП.
#
# Три спецификации:
#   RV_std  — стандартное отклонение изменений ставки (скользящее окно)
#   RV_abs  — среднее абсолютное изменение (менее чувствительна к выбросам)
#   HAR_RV  — Heterogeneous Autoregressive модель (Corsi 2009):
#             компоненты дневная + недельная + месячная волатильность
#             Стандарт литературы по RV.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Расчёт RV
# ============================================================

def compute_rv_metrics(key_rate_df: pd.DataFrame,
                        windows: list[int] | None = None
                        ) -> pd.DataFrame:
    """
    Рассчитывает несколько мер реализованной волатильности.

    Параметры:
        key_rate_df — DataFrame с колонками ['Date', 'Key Rate']
        windows     — скользящие окна в месяцах

    Возвращает:
        DataFrame с колонками: Date, Key_Rate, Delta,
                               RV_std_{w}M, RV_abs_{w}M
                               для каждого окна w
    """
    if windows is None:
        windows = [1, 3, 6]

    df = (key_rate_df[['Date', 'Key Rate']]
          .sort_values('Date')
          .rename(columns={'Key Rate': 'Key_Rate'})
          .copy())

    df['Delta'] = df['Key_Rate'].diff()

    for w in windows:
        # Стандартное отклонение изменений
        df[f'RV_std_{w}M'] = (df['Delta']
                               .rolling(window=w, min_periods=w)
                               .std())
        # Среднее абсолютное изменение
        df[f'RV_abs_{w}M'] = (df['Delta'].abs()
                               .rolling(window=w, min_periods=w)
                               .mean())

    return df.reset_index(drop=True)


# ============================================================
# HAR-RV модель (Corsi 2009)
# ============================================================

def compute_har_rv(key_rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Heterogeneous Autoregressive RV (HAR-RV, Corsi 2009).

    Модель: RV_t = alpha + beta_d * RV_{t-1}
                         + beta_w * RV^w_{t-1}
                         + beta_m * RV^m_{t-1} + eps

    где:
        RV_{t-1}   = RV за прошлый месяц (дневная компонента)
        RV^w_{t-1} = среднее RV за последние 3 месяца (недельная)
        RV^m_{t-1} = среднее RV за последние 6 месяцев (месячная)

    Экономический смысл: волатильность имеет долгую память —
    дневные шоки затухают быстро, месячные — медленно.
    HAR_RV — это прогнозное значение RV из этой модели,
    используемое как индикатор «ожидаемой» волатильности.

    Параметры:
        key_rate_df — DataFrame с ключевой ставкой

    Возвращает:
        DataFrame: Date, RV_daily, RV_weekly, RV_monthly, HAR_RV_fitted
    """
    df = (key_rate_df[['Date', 'Key Rate']]
          .sort_values('Date')
          .rename(columns={'Key Rate': 'Key_Rate'})
          .copy())

    df['Delta']      = df['Key_Rate'].diff()
    df['RV_daily']   = df['Delta'].abs()
    df['RV_weekly']  = df['RV_daily'].rolling(3, min_periods=3).mean()
    df['RV_monthly'] = df['RV_daily'].rolling(6, min_periods=6).mean()

    # Лаговые компоненты
    df['RV_d_lag'] = df['RV_daily'].shift(1)
    df['RV_w_lag'] = df['RV_weekly'].shift(1)
    df['RV_m_lag'] = df['RV_monthly'].shift(1)

    # Целевая переменная: RV следующего месяца
    df['RV_next']  = df['RV_daily'].shift(-1)

    clean = df.dropna(subset=['RV_d_lag', 'RV_w_lag',
                               'RV_m_lag', 'RV_next']).copy()

    if len(clean) < 10:
        df['HAR_RV_fitted'] = np.nan
        return df[['Date', 'RV_daily', 'RV_weekly',
                   'RV_monthly', 'HAR_RV_fitted']]

    X = sm.add_constant(clean[['RV_d_lag', 'RV_w_lag', 'RV_m_lag']])
    model = sm.OLS(clean['RV_next'], X).fit()

    # Fitted values — ожидаемая RV по HAR
    df.loc[clean.index, 'HAR_RV_fitted'] = model.fittedvalues.values

    print(f"  HAR-RV: R2={model.rsquared:.3f}  "
          f"beta_d={model.params['RV_d_lag']:.3f}  "
          f"beta_w={model.params['RV_w_lag']:.3f}  "
          f"beta_m={model.params['RV_m_lag']:.3f}")

    return df[['Date', 'RV_daily', 'RV_weekly',
               'RV_monthly', 'HAR_RV_fitted']]


# ============================================================
# Основная функция: MPU на основе RV
# ============================================================

def build_mpu_rv(key_rate_df: pd.DataFrame,
                  verbose: bool = False) -> pd.DataFrame:
    """
    Строит MPU-индексы на основе реализованной волатильности.

    Два итоговых индекса:
        MPU_rv_std — нормированная RV_std (стандартная)
        MPU_rv_har — нормированный fitted HAR-RV (прогностическая)

    Концептуальное отличие от других MPU:
        Все остальные MPU — это ex ante (из ожиданий рынка).
        MPU_rv — это ex post (фактически реализованная турбулентность).
        Используется как бенчмарк: насколько рыночные ожидания
        соответствуют тому, что реально происходило с ЦБ.

    Параметры:
        key_rate_df — DataFrame с ключевой ставкой
        verbose     — подробный вывод

    Возвращает:
        DataFrame: Date, RV_std_3M, RV_abs_3M, HAR_RV_fitted,
                   MPU_rv_std, MPU_rv_har,
                   MPU_rv_std_norm, MPU_rv_har_norm
    """
    print("RV пайплайн (MPU_rv):")

    rv_df  = compute_rv_metrics(key_rate_df, windows=[1, 3, 6])
    har_df = compute_har_rv(key_rate_df)

    # Объединяем
    result = pd.merge(
        rv_df[['Date', 'Key_Rate', 'RV_std_3M', 'RV_abs_3M']],
        har_df[['Date', 'HAR_RV_fitted']],
        on='Date', how='inner'
    ).dropna(subset=['RV_std_3M', 'HAR_RV_fitted'])

    def zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std()

    result['MPU_rv_std']      = result['RV_std_3M']
    result['MPU_rv_har']      = result['HAR_RV_fitted']
    result['MPU_rv_std_norm'] = zscore(result['RV_std_3M'])
    result['MPU_rv_har_norm'] = zscore(result['HAR_RV_fitted'])

    print(f"  MPU_rv построен: {len(result)} дат")
    print(f"  RV_std_3M: mean={result['RV_std_3M'].mean():.3f}  "
          f"max={result['RV_std_3M'].max():.3f}")
    print(f"  Корреляция RV_std <-> HAR_RV: "
          f"{result['RV_std_3M'].corr(result['HAR_RV_fitted']):.3f}")

    return result.reset_index(drop=True)
