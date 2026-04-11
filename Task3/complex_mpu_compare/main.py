# main.py
# Точка входа: построение всех MPU-индексов и их сравнение.
#
# Архитектура:
#   rnd_core.py        — Bachelier, ATM-форвард, extract_rnd, пайплайн RND
#   mpu_rnd.py         — MPU_pca, MPU_decay, MPU_ext
#   mpu_rwd.py         — MPU_rwd, MPU_vrp
#   mpu_iv_surface.py  — MPU_atm, MPU_rr, MPU_bf
#   mpu_rv.py          — MPU_rv_std, MPU_rv_har
#   mpu_evaluator.py   — универсальный OOS-оценщик
#   main.py            — эта точка входа
#
# Запуск: python main.py

from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_loading.case_2 import get_case_2_IV, get_key_rate_dataframe

from mpu_rnd       import build_mpu_rnd, build_mpu_extended
from mpu_rwd       import build_mpu_rwd
from mpu_iv_surface import build_mpu_iv_surface
from mpu_rv        import build_mpu_rv
from mpu_evaluator import MPUEvaluator
from mpu_tests     import run_all_tests

RESULT_DIR  = Path('results')
MATURITIES  = ['1M', '3M', '6M', '1Y']
MAIN_MAT    = '3M'       # основной срок для MPU_ext и RWD
SPLIT_FRAC  = 0.6        # доля in-sample для OOS теста
HORIZONS    = [1, 2, 3, 6, 12]   # добавлен h=12M
TEST_INDICES = ['MPU_decay', 'MPU_ext', 'MPU_atm', 'MPU_rv_std']


# ============================================================
# Утилита: Series с DatetimeIndex из DataFrame
# ============================================================

def to_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Извлекает колонку col как pd.Series с DatetimeIndex."""
    s = df.set_index('Date')[col].sort_index()
    s.index = pd.to_datetime(s.index)
    return s


# ============================================================
# Шаг 1: Загрузка данных
# ============================================================

def load_data():
    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)
    iv_df       = get_case_2_IV()
    key_rate_df = get_key_rate_dataframe()
    print(f"IV:        {len(iv_df)} строк  "
          f"({iv_df['Date'].nunique()} дат, "
          f"{iv_df['Maturity'].nunique()} сроков)")
    print(f"Key Rate:  {len(key_rate_df)} строк  "
          f"({key_rate_df['Date'].min().date()} — "
          f"{key_rate_df['Date'].max().date()})")
    return iv_df, key_rate_df


# ============================================================
# Шаг 2: Построение всех индексов
# ============================================================

def build_all_indices(iv_df: pd.DataFrame,
                      key_rate_df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Строит все MPU-индексы и возвращает словарь {название: pd.Series}.

    Каждая Series имеет DatetimeIndex и нормирована (z-score)
    для корректного сравнения через OOS-оценщик.
    """
    indices = {}

    # ── 1. MPU_pca и MPU_decay из RND ────────────────────────
    print(f"\n{'='*60}")
    print("1. MPU_pca / MPU_decay  (RND, агрегация по срокам)")
    print("=" * 60)
    mpu_rnd_df, _, _ = build_mpu_rnd(
        iv_df, maturities=MATURITIES
    )
    indices['MPU_pca']   = to_series(mpu_rnd_df, 'MPU_pca_norm')
    indices['MPU_decay'] = to_series(mpu_rnd_df, 'MPU_decay_norm')

    # Сохраняем для дальнейшего использования
    RESULT_DIR.mkdir(exist_ok=True)
    mpu_rnd_df.to_csv(RESULT_DIR / 'mpu_aggregated.csv', index=False)

    # ── 2. MPU_ext: IQR + динамика + хвост (PCA) ─────────────
    print(f"\n{'='*60}")
    print(f"2. MPU_ext  (RND: IQR + ΔIQR + Tail, срок={MAIN_MAT})")
    print("=" * 60)
    mpu_ext_df = build_mpu_extended(iv_df, maturity=MAIN_MAT)
    indices['MPU_ext'] = to_series(mpu_ext_df, 'MPU_ext_norm')
    mpu_ext_df.to_csv(RESULT_DIR / 'mpu_extended.csv', index=False)

    # ── 3. MPU_rwd и MPU_vrp из RWD ──────────────────────────
    print(f"\n{'='*60}")
    print("3. MPU_rwd / MPU_vrp  (Real World Distribution)")
    print("=" * 60)
    mpu_rwd_df = build_mpu_rwd(iv_df, maturities=MATURITIES)
    indices['MPU_rwd'] = to_series(mpu_rwd_df, 'MPU_rwd_norm')
    indices['MPU_vrp'] = to_series(mpu_rwd_df, 'MPU_vrp_norm')
    mpu_rwd_df.to_csv(RESULT_DIR / 'mpu_rwd.csv', index=False)

    # ── 4. MPU из поверхности IV напрямую ────────────────────
    print(f"\n{'='*60}")
    print("4. MPU_atm / MPU_rr / MPU_bf  (IV Surface напрямую)")
    print("=" * 60)
    mpu_iv_df = build_mpu_iv_surface(iv_df, maturities=MATURITIES)
    indices['MPU_atm'] = to_series(mpu_iv_df, 'MPU_atm_norm')
    indices['MPU_rr']  = to_series(mpu_iv_df, 'MPU_rr_norm')
    indices['MPU_bf']  = to_series(mpu_iv_df, 'MPU_bf_norm')
    mpu_iv_df.to_csv(RESULT_DIR / 'mpu_iv_surface.csv', index=False)

    # ── 5. MPU на основе реализованной волатильности ──────────
    print(f"\n{'='*60}")
    print("5. MPU_rv_std / MPU_rv_har  (Realized Volatility)")
    print("=" * 60)
    mpu_rv_df = build_mpu_rv(key_rate_df)
    indices['MPU_rv_std'] = to_series(mpu_rv_df, 'MPU_rv_std_norm')
    indices['MPU_rv_har'] = to_series(mpu_rv_df, 'MPU_rv_har_norm')
    mpu_rv_df.to_csv(RESULT_DIR / 'mpu_rv.csv', index=False)

    print(f"\n{'='*60}")
    print(f"ПОСТРОЕНО ИНДЕКСОВ: {len(indices)}")
    print("=" * 60)
    for name, series in indices.items():
        print(f"  {name:<15}  N={len(series)}  "
              f"[{series.index.min().date()} — "
              f"{series.index.max().date()}]")

    return indices


# ============================================================
# Шаг 3: OOS-сравнение всех индексов
# ============================================================

def evaluate_all(indices: dict[str, pd.Series],
                 key_rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Прогоняет все индексы через MPUEvaluator и возвращает
    сводную таблицу R2_oos.
    """
    print(f"\n{'='*60}")
    print("OOS СРАВНЕНИЕ ВСЕХ ИНДЕКСОВ")
    print("=" * 60)

    evaluator = MPUEvaluator(
        key_rate_df = key_rate_df,
        rv_window   = 3,
        horizons    = HORIZONS,
        split_frac  = SPLIT_FRAC,
    )

    summary = evaluator.compare(indices, save_csv=True)

    # Отчёт с графиками
    evaluator.report(horizon=3, save=True)

    return summary, evaluator


# ============================================================
# Шаг 4: Итоговая сводка
# ============================================================

def print_final_summary(summary: pd.DataFrame) -> None:
    """Печатает итоговую таблицу рейтинга индексов."""
    print(f"\n{'='*60}")
    print("ИТОГОВЫЙ РЕЙТИНГ MPU-ИНДЕКСОВ")
    print("(по среднему R2_oos по горизонтам h=2,3,6)")
    print("=" * 60)

    r2_cols = [c for c in summary.columns if c.startswith('R2_oos_')]

    # Среднее R2_oos по горизонтам h=2,3,6 (исключаем h=1 — шумный)
    h_subset = summary.index[summary.index >= 2]
    if len(h_subset) == 0:
        h_subset = summary.index

    avg_r2 = (summary.loc[h_subset, r2_cols]
              .mean()
              .rename(lambda c: c.replace('R2_oos_', ''))
              .sort_values(ascending=False))

    print(f"\n{'Индекс':<16} {'Средний R2_oos (h=2,3,6)':>25}")
    print("-" * 44)
    for name, val in avg_r2.items():
        marker = ' <- ЛУЧШИЙ' if name == avg_r2.index[0] else ''
        color  = '+' if val > 0 else ' '
        print(f"  {name:<14}  {color}{val:>8.4f}{marker}")

    print()
    best = avg_r2.index[0]
    best_val = avg_r2.iloc[0]
    print(f"Вывод: лучший индекс по OOS — {best} "
          f"(R2_oos={best_val:+.4f})")


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == '__main__':

    # 1. Данные
    iv_df, key_rate_df = load_data()

    # 2. Строим все индексы
    indices = build_all_indices(iv_df, key_rate_df)

    # 3. OOS-сравнение
    summary, evaluator = evaluate_all(indices, key_rate_df)

    # 4. Итоговая сводка
    print_final_summary(summary)

    # 5. Дополнительные тесты для четырёх индексов
    test_subset = {k: v for k, v in indices.items()
                   if k in TEST_INDICES}

    run_all_tests(
        indices_subset = test_subset,
        iv_df          = iv_df,
        key_rate_df    = key_rate_df,
        horizons_all   = [1, 2, 3, 6, 12],
        save           = True,
    )

    print(f"\nВсе результаты сохранены в: {RESULT_DIR.resolve()}")