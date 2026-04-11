# mpu_rnd.py
# Индексы MPU на основе RND (Risk-Neutral Distribution).
#
# Реализует три спецификации:
#   MPU_pca    — PCA по IQR_9010 нескольких сроков экспирации
#   MPU_decay  — взвешенное среднее IQR_9010 (короткие сроки важнее)
#   MPU_ext    — PCA по трём компонентам одного срока:
#                IQR_9010 + Δ(IQR_9010) + Tail_total

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from rnd_core import run_rnd_pipeline, check_quality


# ============================================================
# MPU_pca и MPU_decay: агрегация по срокам
# ============================================================

def build_mpu_rnd(iv_df: pd.DataFrame,
                  maturities: list[str] | None = None,
                  metric: str = 'IQR_9010',
                  verbose: bool = False
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Строит MPU_pca и MPU_decay из RND поверхности IV.

    Оба метода агрегируют метрику неопределённости (IQR_9010)
    по нескольким срокам экспирации в единый временной ряд.

    MPU_decay: w_i = exp(-ln2 * i) / sum — короткие сроки важнее.
    MPU_pca:   первая главная компонента IQR по всем срокам.

    Параметры:
        iv_df      — поверхность IV
        maturities — сроки для агрегации
        metric     — агрегируемая метрика
        verbose    — подробный вывод

    Возвращает:
        mpu_df    — DataFrame: Date, MPU_pca, MPU_decay
                               MPU_pca_norm, MPU_decay_norm
        stats_df  — полная таблица статистик RND
        meta      — словарь с весами и нагрузками PCA
    """
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']

    print("RND пайплайн (MPU_pca / MPU_decay):")
    stats_df, _ = run_rnd_pipeline(iv_df, maturities, verbose=verbose)
    stats_df    = check_quality(stats_df)
    clean_df    = stats_df[stats_df['quality_ok']].copy()

    # Pivot: строки = даты, столбцы = сроки
    pivot = (
        clean_df[clean_df['Maturity'].isin(maturities)]
        .pivot_table(values=metric, index='Date', columns='Maturity')
        .reindex(columns=maturities)
        .dropna()
    )

    if pivot.empty:
        raise ValueError(f"Нет данных для агрегации по {maturities}")

    n_mat = pivot.shape[1]

    # Decay-веса
    lam           = np.log(2)
    raw_w         = np.array([np.exp(-lam * i) for i in range(n_mat)])
    decay_weights = raw_w / raw_w.sum()
    mpu_decay     = pd.Series(pivot.values @ decay_weights,
                               index=pivot.index)

    # PCA
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(pivot.values)
    pca_fit  = PCA(n_components=n_mat)
    scores   = pca_fit.fit_transform(X_scaled)
    loadings = pca_fit.components_[0].copy()
    explained = pca_fit.explained_variance_ratio_

    pc1 = pd.Series(scores[:, 0], index=pivot.index)
    if pc1.corr(mpu_decay) < 0:
        pc1      = -pc1
        loadings = -loadings

    def zscore(s):
        return (s - s.mean()) / s.std()

    mpu_df = pd.DataFrame({
        'Date':           pivot.index,
        'MPU_pca':        pc1.values,
        'MPU_decay':      mpu_decay.values,
        'MPU_pca_norm':   zscore(pc1).values,
        'MPU_decay_norm': zscore(mpu_decay).values,
    }).reset_index(drop=True)

    meta = {
        'maturities':    maturities,
        'metric':        metric,
        'decay_weights': dict(zip(maturities, decay_weights)),
        'loadings_pca':  dict(zip(maturities, loadings)),
        'explained_pca': {f'PC{i+1}': v
                          for i, v in enumerate(explained)},
        'corr':          float(pc1.corr(mpu_decay)),
    }

    print(f"  MPU_pca / MPU_decay построены: {len(mpu_df)} дат")
    print(f"  PC1 объясняет {explained[0]:.1%} дисперсии")
    print(f"  Корреляция PCA <-> Decay: {meta['corr']:.3f}")

    return mpu_df, clean_df, meta


# ============================================================
# MPU_ext: IQR + динамика + хвост (PCA по трём компонентам)
# ============================================================

def build_mpu_extended(iv_df: pd.DataFrame,
                        maturity: str = '3M',
                        verbose: bool = False) -> pd.DataFrame:
    """
    Расширенный MPU из трёх компонент одного срока экспирации.

    Компоненты:
        C1 = IQR_9010     — ширина центрального коридора (уровень)
        C2 = Δ(IQR_9010)  — ежемесячное изменение IQR (динамика)
        C3 = Tail_total   — суммарный хвостовой риск

    Агрегация: первая главная компонента (PCA).

    Экономический смысл:
        C1 — насколько широк разброс ожиданий сейчас
        C2 — нарастает или спадает неопределённость
        C3 — риск экстремального сценария (толстые хвосты)

    Параметры:
        iv_df    — поверхность IV
        maturity — срок экспирации для расчёта компонент
        verbose  — подробный вывод

    Возвращает:
        DataFrame: Date, MPU_ext, MPU_ext_norm,
                   C1_IQR, C2_delta_IQR, C3_tail
    """
    print(f"RND пайплайн (MPU_ext, срок={maturity}):")
    stats_df, _ = run_rnd_pipeline(iv_df, [maturity], verbose=verbose)
    stats_df    = check_quality(stats_df)
    clean_df    = (stats_df[stats_df['quality_ok'] &
                            (stats_df['Maturity'] == maturity)]
                  .sort_values('Date').copy())

    if len(clean_df) < 10:
        raise ValueError(f"Мало данных для MPU_ext (срок {maturity})")

    # Три компоненты
    C1 = clean_df['IQR_9010'].values
    C2 = np.diff(C1, prepend=C1[0])      # Δ(IQR), первое значение = 0
    C3 = clean_df['Tail_total'].values

    X = np.column_stack([C1, C2, C3])

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    pca_fit = PCA(n_components=3)
    scores  = pca_fit.fit_transform(X_sc)
    ev      = pca_fit.explained_variance_ratio_

    # PC1 должна коррелировать с C1 (уровнем неопределённости)
    pc1 = scores[:, 0]
    if np.corrcoef(pc1, C1)[0, 1] < 0:
        pc1 = -pc1

    def zscore(arr):
        return (arr - arr.mean()) / arr.std()

    result = pd.DataFrame({
        'Date':         clean_df['Date'].values,
        'MPU_ext':      pc1,
        'MPU_ext_norm': zscore(pc1),
        'C1_IQR':       C1,
        'C2_delta_IQR': C2,
        'C3_tail':      C3,
    })

    print(f"  MPU_ext построен: {len(result)} дат")
    print(f"  PC1 объясняет {ev[0]:.1%} | "
          f"PC2 {ev[1]:.1%} | PC3 {ev[2]:.1%}")

    return result
