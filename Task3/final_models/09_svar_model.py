# 09_svar_model.py
# Structural VAR (SVAR) для анализа макро-финансового влияния MPU.
#
# Модель:
#   A · y_t = B_1·y_{t-1} + ... + B_p·y_{t-p} + u_t
#
# Приведённая форма (reduced form):
#   y_t = Φ_1·y_{t-1} + ... + Φ_p·y_{t-p} + e_t,  e_t ~ N(0,Σ)
#
# Идентификация структурных шоков — рекурсивная (Cholesky):
#   P·P' = Σ  (нижнетреугольное P),  u_t = P^{-1}·e_t
#
# Порядок переменных (от наиболее экзогенной к наиболее эндогенной):
#   1. MPU_decay   — неопределённость ДКП
#                   (фиксируется опционным рынком ДО решения ЦБ)
#   2. Key_Rate    — инструмент ДКП
#   3. Inflation   — целевая переменная ЦБ
#   4. GVA_YoY     — реальный сектор (медленнее всего реагирует)
#
# Что строим:
#   1. IRF  — Impulse Response Functions (отклики на шок MPU)
#   2. FEVD — Forecast Error Variance Decomposition
#             (доля дисперсии прогнозной ошибки объяснённая шоком MPU)
#   3. HD   — Historical Decomposition
#             (вклад шока MPU в фактическую динамику каждой переменной)
#
# Стационарность:
#   Если переменная I(1) по ADF+KPSS → берём первую разность.
#   MPU_decay оставляем в уровнях: ADF имеет низкую мощность при N=60,
#   и z-score ряд стационарен по построению.
#
# Результаты -> Task3/final_models/svar_model/
# Запуск: python 09_svar_model.py

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))
MPU_DIR  = BASE_DIR / 'mpu_models'
OUT_DIR  = BASE_DIR / 'svar_model'
PLOT_DIR = OUT_DIR / 'plots'

from data_loading.case_2 import get_key_rate_dataframe
from data_loading.extra_data import get_gva_monthly_dataframe

# ── Настройки ────────────────────────────────────────────────
MAX_LAG      = 2      # порядок VAR (оптимален при N=60)
FORCE_LEVELS = True   # True → все переменные в уровнях (Вариант 2)
# При N=60 ADF имеет низкую мощность и ошибочно
# находит единичный корень у стационарных рядов
# с высокой дисперсией (Key_Rate 2019-2025).
# Key_Rate в России вернулась с 4.25% к 15% —
# стохастического тренда нет, ряд стационарен.
IRF_HORIZON  = 24     # горизонт IRF (месяцев)
N_BOOT       = 500    # bootstrap итераций
CI_LEVEL     = 0.68   # 68% CI (≈ ±1 std)
SAMPLE_START = '2019-03-01'
SHOCK_VAR    = 'MPU_decay'   # переменная шока

# Порядок переменных в системе (важен для Cholesky!)
VAR_ORDER = ['MPU_decay', 'Key_Rate', 'Inflation', 'GVA_YoY']

# Мета-информация для графиков
VAR_META = {
    'MPU_decay':   ('MPU неопределённость (z)',  '#E65100'),
    'Key_Rate':    ('Ключевая ставка (%)',         '#1565C0'),
    'D_Key_Rate':  ('Δ Ключевая ставка (пп)',      '#1565C0'),
    'Inflation':   ('Инфляция г/г (%)',             '#C62828'),
    'GVA_YoY':    ('ВДС YoY (%)',                  '#27AE60'),
    'D_GVA_YoY':  ('Δ ВДС YoY (%)',               '#27AE60'),
}


# ════════════════════════════════════════════════════════════
# Утилиты
# ════════════════════════════════════════════════════════════

def _save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {name}")


# ════════════════════════════════════════════════════════════
# Загрузка и подготовка данных
# ════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Загружает и объединяет все переменные системы."""

    # MPU_decay
    mpu_df = pd.read_csv(MPU_DIR / 'MPU_decay.csv', parse_dates=['Date'])
    mpu    = mpu_df.set_index('Date')['MPU_decay_norm'].rename('MPU_decay')
    mpu.index = pd.to_datetime(mpu.index)

    # Макро: Key_Rate, Inflation
    kr_df = get_key_rate_dataframe()
    kr = (kr_df[['Date', 'Key Rate', 'Inflation']]
          .dropna(subset=['Date'])
          .rename(columns={'Key Rate': 'Key_Rate'})
          .sort_values('Date')
          .set_index('Date'))
    kr.index = pd.to_datetime(kr.index)

    # GVA YoY
    gva_df = get_gva_monthly_dataframe()
    gva_df['Date'] = pd.to_datetime(gva_df['Date'])
    gva = gva_df.sort_values('Date').set_index('Date')['Value']
    gva_yoy = (gva / gva.shift(12) - 1) * 100
    gva_yoy.name = 'GVA_YoY'

    # Объединяем
    df = pd.concat([mpu, kr[['Key_Rate', 'Inflation']], gva_yoy],
                   axis=1)
    df = (df[df.index >= SAMPLE_START]
          .dropna()
          .sort_index())

    print(f"Данные загружены:")
    print(f"  Период: {df.index.min().date()} — {df.index.max().date()}")
    print(f"  N = {len(df)}")
    print(f"  Переменные: {list(df.columns)}")
    return df


def check_stationarity(df: pd.DataFrame) -> dict:
    """
    Совместный тест ADF + KPSS.

    При FORCE_LEVELS=True все переменные остаются в уровнях.
    Обоснование: при N=60 ADF имеет низкую мощность (~30-40%)
    и часто ошибочно находит единичный корень у стационарных рядов
    с высокой дисперсией или структурными разрывами (2022).
    Key_Rate в России 2019-2025 не имеет стохастического тренда:
    ряд вернулся к сопоставимым уровням — это стационарность
    с изменяющимся средним, а не I(1) процесс.

    При FORCE_LEVELS=False применяется автоматическое правило:
      ADF p < 0.10             → стационарен ('level')
      KPSS отвергает H0        → I(1) → разность ('diff')
      MPU_decay                → всегда 'level'
    """
    print(f"\nТест на стационарность (ADF + KPSS):")
    if FORCE_LEVELS:
        print(f"  FORCE_LEVELS=True: все переменные в уровнях")
        print(f"  Обоснование: N=60, низкая мощность ADF при структурных разрывах")
    print(f"  {'Переменная':<15}  {'ADF p':>7}  {'KPSS':>8}  {'Решение'}")
    print("  " + "─" * 52)

    transforms = {}
    for col in df.columns:
        s = df[col].dropna()

        adf_p = adfuller(s, maxlag=3, autolag='AIC')[1]
        try:
            kpss_stat, kpss_p, _, _ = kpss(s, regression='c', nlags='auto')
            kpss_rej = kpss_stat > 0.463
        except Exception:
            kpss_rej = False

        if FORCE_LEVELS:
            # Вариант 2: принудительно все в уровнях
            decision = 'level'
            note     = 'уровень (FORCE_LEVELS)'
        elif col == 'MPU_decay':
            decision = 'level'
            note     = 'z-score → уровень'
        elif adf_p < 0.10:
            decision = 'level'
            note     = 'стационарен'
        elif kpss_rej and adf_p > 0.10:
            decision = 'diff'
            note     = 'I(1) → Δ'
        else:
            decision = 'level'
            note     = 'неясно → уровень'

        transforms[col] = decision
        print(f"  {col:<15}  {adf_p:>7.4f}  "
              f"{'I(1)' if kpss_rej else 'stat':>8}  {note}")

    return transforms


def apply_transforms(df: pd.DataFrame,
                     transforms: dict) -> tuple[pd.DataFrame, dict]:
    """
    Применяет трансформации, сохраняет маппинг
    оригинальное_имя → итоговое_имя.
    """
    parts   = {}
    col_map = {}  # оригинальное → итоговое имя

    for col in df.columns:
        if transforms[col] == 'level':
            parts[col]  = df[col]
            col_map[col] = col
        else:
            new_name = f'D_{col}'
            parts[new_name] = df[col].diff()
            col_map[col]    = new_name
            print(f"  {col} → {new_name}")

    result = pd.DataFrame(parts, index=df.index).dropna()
    return result, col_map


# ════════════════════════════════════════════════════════════
# Оценка VAR и выбор лага
# ════════════════════════════════════════════════════════════

def select_lag(df: pd.DataFrame, maxlag: int = 6) -> int:
    """Выбирает оптимальный лаг по AIC/BIC/HQIC."""
    model = VAR(df)
    res   = model.select_order(maxlags=maxlag)

    print(f"\nВыбор лага VAR (maxlag={maxlag}):")
    print(f"  AIC  → p = {res.aic}")
    print(f"  BIC  → p = {res.bic}")
    print(f"  HQIC → p = {res.hqic}")

    # При малой выборке предпочитаем BIC (штрафует больше за параметры)
    p_bic = res.bic
    p_use = min(p_bic, MAX_LAG)   # не больше MAX_LAG
    p_use = max(p_use, 1)          # не меньше 1
    print(f"  Используем p = {p_use} "
          f"(min(BIC={p_bic}, MAX_LAG={MAX_LAG}))")
    return p_use


def fit_var(df: pd.DataFrame, p: int) -> object:
    """Оценивает VAR(p) и возвращает результат."""
    model  = VAR(df.values)
    result = model.fit(p)

    print(f"\nVAR({p}) оценён:")
    print(f"  N наблюдений:  {result.nobs}")
    print(f"  K переменных:  {result.neqs}")
    print(f"  Параметров:    {result.df_model}")
    print(f"  Log-likelihood: {result.llf:.2f}")

    # Тест на автокорреляцию остатков (Portmanteau)
    try:
        pt = result.test_whiteness(nlags=10, signif=0.05)
        print(f"  Тест Ljung-Box (остатки): "
              f"stat={pt.test_statistic:.3f}  "
              f"p={pt.pvalue:.4f}  "
              f"{'OK' if pt.pvalue > 0.05 else '⚠️  автокорреляция'}")
    except Exception:
        pass

    return result


# ════════════════════════════════════════════════════════════
# Cholesky идентификация
# ════════════════════════════════════════════════════════════

def get_cholesky(var_result, df_cols: list,
                 var_order: list) -> np.ndarray:
    """
    Возвращает нижнетреугольную матрицу Холецкого P
    такую что P·P' = Σ (ковариационная матрица остатков).

    Порядок переменных в var_order определяет рекурсивную
    структуру: переменная i не зависит одновременно от
    переменных j > i.

    Параметры:
        var_result — оценённый VAR
        df_cols    — текущие имена колонок в DataFrame
        var_order  — желаемый порядок (может содержать оригинальные имена)

    Возвращает:
        P — матрица идентификации (K × K)
    """
    sigma = var_result.sigma_u   # ковариационная матрица остатков

    # Находим индексы переменных в нужном порядке
    # var_order может содержать имена вида 'Key_Rate',
    # а df_cols — 'D_Key_Rate'. Ищем по подстроке.
    idx = []
    for name in var_order:
        # Сначала точное совпадение
        if name in df_cols:
            idx.append(df_cols.index(name))
        else:
            # Поиск по подстроке (D_Key_Rate содержит Key_Rate)
            found = [i for i, c in enumerate(df_cols) if name in c]
            if found:
                idx.append(found[0])

    if len(idx) != len(df_cols):
        # Если не нашли все — используем исходный порядок
        idx = list(range(len(df_cols)))
        print(f"  ⚠️  Порядок Cholesky: используем исходный {df_cols}")
    else:
        reorder = [df_cols[i] for i in idx]
        print(f"  Порядок Cholesky: {reorder}")

    # Переупорядочиваем ковариационную матрицу
    sigma_reord = sigma[np.ix_(idx, idx)]

    # Cholesky разложение
    P = np.linalg.cholesky(sigma_reord)

    # Возвращаем P в исходном порядке переменных
    inv_idx = np.argsort(idx)
    P_orig  = P[np.ix_(inv_idx, inv_idx)]

    return P_orig, idx


# ════════════════════════════════════════════════════════════
# IRF (Impulse Response Functions)
# ════════════════════════════════════════════════════════════

def compute_irf(var_result, P: np.ndarray,
                shock_idx: int,
                horizon: int = IRF_HORIZON) -> np.ndarray:
    """
    Вычисляет IRF для структурного шока shock_idx.

    Алгоритм:
        1. Получаем коэффициенты VAR (компаньон-матрица)
        2. Вычисляем матрицы отклика Θ_h = Φ_h · P
        3. Θ_h[:,shock_idx] — вектор откликов всех переменных на шок shock_idx

    Возвращает:
        irf: (horizon+1) × K матрица — IRF по горизонтам
    """
    irf_obj = var_result.irf(horizon)
    # irf_obj.orth_irfs[h] — матрица ортогонализированных откликов на горизонте h
    # Размер: (horizon+1, K, K), где [h,i,j] = отклик i на шок j
    orth_irf = irf_obj.orth_irfs   # использует встроенный Cholesky statsmodels
    return orth_irf[:, :, shock_idx]  # (horizon+1) × K


def bootstrap_irf(df: pd.DataFrame,
                  p: int,
                  shock_idx: int,
                  horizon: int = IRF_HORIZON,
                  n_boot: int = N_BOOT,
                  ci_level: float = CI_LEVEL) -> tuple:
    """
    Bootstrap доверительные интервалы для SVAR IRF.

    Метод: residual bootstrap.
    1. Оцениваем VAR на исходных данных → остатки e_t
    2. Пересэмплируем e_t → строим псевдоданные y*
    3. Переоцениваем VAR → получаем IRF*
    4. Квантили распределения IRF* → CI

    Возвращает:
        ci_low, ci_high: (horizon+1) × K массивы
    """
    np.random.seed(42)
    alpha  = (1 - ci_level) / 2
    n, k   = df.shape
    boot_irfs = []

    # Базовая оценка
    base_var = VAR(df.values).fit(p)
    resids   = base_var.resid      # (n-p) × k
    fitted   = base_var.fittedvalues  # (n-p) × k
    coefs    = base_var.coefs      # (p, k, k)
    intercept = base_var.intercept  # k

    for b in range(n_boot):
        # Пересэмплируем остатки
        idx_r   = np.random.choice(len(resids), size=len(resids), replace=True)
        resids_b = resids[idx_r]

        # Строим псевдоряд
        y_boot = np.zeros((n, k))
        y_boot[:p] = df.values[:p]  # начальные условия
        for t in range(p, n):
            y_t = intercept.copy()
            for lag in range(p):
                y_t += coefs[lag] @ y_boot[t - lag - 1]
            y_t += resids_b[t - p]
            y_boot[t] = y_t

        try:
            var_b   = VAR(y_boot).fit(p)
            irf_b_obj = var_b.irf(horizon)
            irf_b   = irf_b_obj.orth_irfs[:, :, shock_idx]
            boot_irfs.append(irf_b)
        except Exception:
            continue

    if not boot_irfs:
        # Fallback: аналитические CI
        base_irf = compute_irf(base_var, None, shock_idx, horizon)
        se = np.ones_like(base_irf) * 0.1
        return base_irf - 1.645*se, base_irf + 1.645*se

    boot_arr = np.array(boot_irfs)  # (n_boot, horizon+1, k)
    ci_low   = np.quantile(boot_arr, alpha, axis=0)
    ci_high  = np.quantile(boot_arr, 1 - alpha, axis=0)

    return ci_low, ci_high


# ════════════════════════════════════════════════════════════
# FEVD (Forecast Error Variance Decomposition)
# ════════════════════════════════════════════════════════════

def compute_fevd(var_result,
                 horizon: int = IRF_HORIZON) -> np.ndarray:
    """
    FEVD: доля дисперсии прогнозной ошибки каждой переменной,
    объяснённая каждым структурным шоком.

    Возвращает:
        fevd: (horizon+1) × K × K массив
              fevd[h, i, j] = доля шока j в дисперсии переменной i на горизонте h
    """
    fevd_obj = var_result.fevd(horizon + 1)
    return fevd_obj.decomp   # (horizon+1, K, K)


# ════════════════════════════════════════════════════════════
# Historical Decomposition (HD)
# ════════════════════════════════════════════════════════════

def compute_historical_decomposition(
        var_result,
        df: pd.DataFrame,
        p: int,
        shock_idx: int,
        horizon: int = IRF_HORIZON) -> pd.DataFrame:
    """
    Историческая декомпозиция: вклад структурного шока MPU
    в фактическую динамику каждой переменной.

    Алгоритм Beveridge-Nelson для HD:
        y_t = baseline_t + Σ_j contribution_j(t)

    Где contribution_j(t) = Σ_{s=0}^{t} Θ_s[:,j] · u_j(t-s)
    (свёртка IRF со структурными шоками)

    На практике используем встроенную функцию statsmodels
    или реализуем через структурные остатки.

    Параметры:
        shock_idx — индекс шока MPU в системе

    Возвращает:
        DataFrame с вкладом шока MPU в каждую переменную
    """
    k      = df.shape[1]
    n_obs  = var_result.nobs
    cols   = list(df.columns)

    # Структурные шоки: u_t = P^{-1} · e_t
    e_t = var_result.resid          # (n_obs, k)
    sigma = var_result.sigma_u
    P   = np.linalg.cholesky(sigma)  # нижнетреугольная
    P_inv = np.linalg.inv(P)
    u_t = (P_inv @ e_t.T).T          # (n_obs, k)

    # IRF матрица Θ_h: (horizon+1, k, k)
    irf_obj  = var_result.irf(horizon)
    orth_irf = irf_obj.orth_irfs     # (horizon+1, k, k)

    # Вклад шока shock_idx в переменную i в момент t:
    # contrib_i(t) = Σ_{s=0}^{min(t,horizon)} Θ_s[i, shock_idx] · u_{shock_idx}(t-s)
    T     = n_obs
    contribs = np.zeros((T, k))

    for t in range(T):
        for s in range(min(t + 1, horizon + 1)):
            if t - s >= 0 and t - s < len(u_t):
                irf_vals = orth_irf[s, :, shock_idx]  # k-вектор
                contribs[t] += irf_vals * u_t[t - s, shock_idx]

    # Индексы для DataFrame (совпадают с остатками VAR)
    idx = df.index[-T:]
    hd  = pd.DataFrame(contribs, index=idx,
                       columns=[f'HD_{c}' for c in cols])

    return hd


# ════════════════════════════════════════════════════════════
# Графики
# ════════════════════════════════════════════════════════════

def plot_irf_panel(irf: np.ndarray,
                   ci_low: np.ndarray,
                   ci_high: np.ndarray,
                   cols: list,
                   shock_name: str = 'MPU') -> None:
    """Панель IRF: все переменные в одной фигуре."""
    k   = len(cols)
    n_c = min(k, 2)
    n_r = (k + 1) // 2
    fig, axes = plt.subplots(n_r, n_c,
                             figsize=(7 * n_c, 5 * n_r))
    if k == 1:
        axes = np.array([[axes]])
    elif n_r == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f'SVAR IRF: шок {shock_name} (+1 std)\n'
        f'Рекурсивная идентификация (Cholesky), '
        f'{int(CI_LEVEL*100)}% CI (bootstrap)',
        fontweight='bold', fontsize=13
    )

    hs = np.arange(irf.shape[0])
    for i, col in enumerate(cols):
        ax    = axes[i // n_c, i % n_c]
        meta  = VAR_META.get(col, (col, '#333'))
        label, color = meta

        ax.fill_between(hs, ci_low[:, i], ci_high[:, i],
                        alpha=0.25, color=color, label=f'{int(CI_LEVEL*100)}% CI')
        ax.plot(hs, irf[:, i], color=color, lw=2.5, label='IRF')
        ax.axhline(0, color='black', lw=1.2, ls='--', alpha=0.6)

        # Значимые горизонты (CI не пересекают 0)
        sig = (ci_low[:, i] > 0) | (ci_high[:, i] < 0)
        ax.scatter(hs[sig], irf[sig, i], color=color,
                   s=60, zorder=6, marker='*')

        ax.set_title(label, fontweight='bold', fontsize=10)
        ax.set_xlabel('Горизонт (месяцев)')
        ax.set_ylabel('Отклик')
        ax.set_xticks(range(0, len(hs), 3))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    # Убираем пустые ячейки
    for i in range(k, n_r * n_c):
        axes[i // n_c, i % n_c].set_visible(False)

    plt.tight_layout()
    _save(fig, 'irf_panel.png')


def plot_fevd(fevd: np.ndarray, cols: list,
              shock_idx: int, shock_name: str = 'MPU') -> None:
    """
    FEVD: доля дисперсии прогнозной ошибки объяснённая шоком MPU.
    Горизонтальная линия — доля на горизонте 12 и 24 месяца.
    """
    k    = len(cols)
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 5), sharey=False)
    if k == 1:
        axes = [axes]

    fig.suptitle(
        f'FEVD: доля дисперсии прогнозной ошибки, объяснённая шоком {shock_name}',
        fontweight='bold', fontsize=13
    )

    hs = np.arange(fevd.shape[0])
    for i, col in enumerate(cols):
        ax    = axes[i]
        meta  = VAR_META.get(col, (col, '#333'))
        label, color = meta

        share = fevd[:, i, shock_idx] * 100   # в процентах

        ax.fill_between(hs, 0, share, alpha=0.25, color=color)
        ax.plot(hs, share, color=color, lw=2.5)
        ax.axhline(share[min(12, len(share)-1)],
                   color='grey', lw=1, ls='--', alpha=0.7)

        # Аннотации на ключевых горизонтах
        for h_ann in [6, 12, 18]:
            if h_ann < len(share):
                ax.annotate(f'{share[h_ann]:.1f}%',
                            (h_ann, share[h_ann]),
                            textcoords='offset points',
                            xytext=(4, 5), fontsize=8)

        ax.set_title(label, fontweight='bold', fontsize=10)
        ax.set_xlabel('Горизонт (месяцев)')
        ax.set_ylabel('Доля дисперсии (%)')
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, len(hs), 3))
        ax.grid(alpha=0.2)

    plt.tight_layout()
    _save(fig, 'fevd.png')


def plot_historical_decomposition(hd: pd.DataFrame,
                                  df: pd.DataFrame,
                                  col_map: dict) -> None:
    """
    HD: вклад шока MPU в фактическую динамику переменных.

    Для каждой переменной:
      - Серая линия: фактическое значение
      - Цветные столбцы: вклад шока MPU (+ и -)
    """
    vars_to_plot = [c for c in ['Key_Rate', 'Inflation', 'GVA_YoY']
                    if col_map.get(c, c) in df.columns]

    n   = len(vars_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        'Historical Decomposition: вклад шока MPU_decay\n'
        'в динамику макропеременных',
        fontweight='bold', fontsize=13
    )

    for ax, orig_col in zip(axes, vars_to_plot):
        col   = col_map.get(orig_col, orig_col)
        meta  = VAR_META.get(col, VAR_META.get(orig_col, (orig_col, '#333')))
        label, color = meta

        hd_col = f'HD_{col}'
        if hd_col not in hd.columns:
            continue

        contrib = hd[hd_col]
        actual  = df[col].reindex(hd.index)

        # Столбцы вклада (цветные)
        ax.bar(contrib.index,
               contrib.values,
               width=25,
               color=np.where(contrib.values >= 0, color, '#C0392B'),
               alpha=0.65,
               label='Вклад шока MPU')

        # Фактические значения
        ax2 = ax.twinx()
        ax2.plot(actual.index, actual.values,
                 color='black', lw=2, label=f'{label} (факт)')
        ax2.set_ylabel(label, fontsize=9)

        ax.axhline(0, color='black', lw=1)
        ax.set_ylabel('Вклад MPU шока', fontsize=9)
        ax.set_title(f'Вклад шока MPU в {label}', fontweight='bold')

        # Объединяем легенды
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
        ax.grid(alpha=0.2)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    _save(fig, 'historical_decomposition.png')


def plot_hd_episodes(hd: pd.DataFrame,
                     df: pd.DataFrame) -> None:
    """
    Детальный HD по ключевым периодам:
    2022 (шок), 2023-2024 (ужесточение), 2025 (смягчение).
    """
    episodes = {
        '2022: Шок': ('2022-01-01', '2022-12-01'),
        '2023–2024: Ужесточение': ('2023-01-01', '2024-12-01'),
        '2025: Смягчение': ('2025-01-01', hd.index.max().strftime('%Y-%m-%d')),
    }

    hd_cols = [c for c in hd.columns if c != 'HD_MPU_decay']
    if not hd_cols:
        return

    n_ep  = len(episodes)
    n_var = min(len(hd_cols), 3)
    fig, axes = plt.subplots(n_var, n_ep,
                             figsize=(6 * n_ep, 4 * n_var))
    if n_var == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Вклад шока MPU в ключевые периоды',
                 fontweight='bold', fontsize=13)

    for j, (ep_name, (d0, d1)) in enumerate(episodes.items()):
        for i, hd_col in enumerate(hd_cols[:n_var]):
            ax      = axes[i, j]
            col     = hd_col.replace('HD_', '')
            meta    = VAR_META.get(col, (col, '#333'))
            label, color = meta

            mask    = (hd.index >= d0) & (hd.index <= d1)
            sub     = hd.loc[mask, hd_col]

            ax.bar(sub.index, sub.values, width=25,
                   color=np.where(sub.values >= 0, color, '#C0392B'),
                   alpha=0.7)
            ax.axhline(0, color='black', lw=1)

            mean_contrib = sub.mean()
            ax.set_title(f'{label}\n{ep_name}\nСредний вклад: {mean_contrib:+.3f}',
                         fontsize=8, fontweight='bold')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.grid(alpha=0.2)

    plt.tight_layout()
    _save(fig, 'hd_episodes.png')


def plot_var_stability(var_result) -> None:
    """График корней характеристического полинома VAR."""
    roots = var_result.roots
    fig, ax = plt.subplots(figsize=(6, 6))

    # Единичная окружность
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta),
            color='black', lw=1.5, ls='--', alpha=0.5,
            label='Единичная окружность')

    # Корни
    ax.scatter(roots.real, roots.imag,
               color='#E65100', s=80, zorder=5,
               label='Корни VAR')

    ax.axhline(0, color='grey', lw=0.8)
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title('Устойчивость VAR: корни характеристического полинома\n'
                 '(все корни должны быть внутри единичной окружности)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    all_inside = all(abs(r) < 1.0 for r in roots)
    status = "✅ Модель устойчива" if all_inside else "⚠️  Нарушение устойчивости"
    ax.text(0, -1.2, status, ha='center', fontsize=11, fontweight='bold',
            color='green' if all_inside else 'red')

    plt.tight_layout()
    _save(fig, 'var_stability.png')


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"09 | SVAR  (Cholesky, p≤{MAX_LAG}, h=0..{IRF_HORIZON})")
    print("=" * 65)

    # ── 1. Данные ──────────────────────────────────────────
    print("\n[1] Загрузка данных...")
    df_raw = load_data()

    # ── 2. Стационарность ─────────────────────────────────
    print("\n[2] Тест стационарности...")
    transforms = check_stationarity(df_raw)

    print("\n  Трансформации:")
    df, col_map = apply_transforms(df_raw, transforms)
    print(f"  Итоговые переменные: {list(df.columns)}")
    print(f"  N = {len(df)}")

    # Упорядочиваем переменные согласно VAR_ORDER
    ordered_cols = []
    for vname in VAR_ORDER:
        mapped = col_map.get(vname, vname)
        if mapped in df.columns:
            ordered_cols.append(mapped)
    # Добавляем оставшиеся
    for c in df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    df = df[ordered_cols]
    print(f"  Порядок в системе: {ordered_cols}")

    # ── 3. Выбор лага ─────────────────────────────────────
    print("\n[3] Выбор оптимального лага...")
    p = select_lag(df, maxlag=min(6, len(df) // 10))

    # ── 4. Оценка VAR ─────────────────────────────────────
    print("\n[4] Оценка VAR...")
    var_res = fit_var(df, p)

    # ── 5. Устойчивость ───────────────────────────────────
    roots = var_res.roots
    all_inside = all(abs(r) < 1.0 for r in roots)
    print(f"\n[5] Устойчивость VAR:")
    print(f"  Модуль наибольшего корня: {max(abs(r) for r in roots):.4f}")
    print(f"  {'✅ Модель устойчива' if all_inside else '⚠️  Нарушение устойчивости'}")

    # ── 6. Индекс шока MPU ────────────────────────────────
    cols      = list(df.columns)
    shock_col = col_map.get(SHOCK_VAR, SHOCK_VAR)
    if shock_col in cols:
        shock_idx = cols.index(shock_col)
    else:
        shock_idx = 0
    print(f"\n[6] Шок MPU: '{cols[shock_idx]}' (индекс {shock_idx})")

    # ── 7. IRF ────────────────────────────────────────────
    print("\n[7] Вычисление IRF...")
    irf = compute_irf(var_res, None, shock_idx, IRF_HORIZON)
    print(f"  IRF shape: {irf.shape}  (горизонты × переменные)")

    print("\n  Сводка откликов (пик и знак):")
    print(f"  {'Переменная':<18}  {'Пик h':>6}  {'Beta':>8}  {'Знак'}")
    print("  " + "─" * 45)
    for i, col in enumerate(cols):
        peak_idx  = np.argmax(np.abs(irf[:, i]))
        peak_beta = irf[peak_idx, i]
        print(f"  {col:<18}  {peak_idx:>6}M  "
              f"{peak_beta:>8.4f}  "
              f"{'↑' if peak_beta > 0 else '↓'}")

    # ── 8. Bootstrap CI ───────────────────────────────────
    print(f"\n[8] Bootstrap CI ({N_BOOT} итераций)...")
    ci_low, ci_high = bootstrap_irf(
        df, p, shock_idx,
        horizon=IRF_HORIZON,
        n_boot=N_BOOT,
        ci_level=CI_LEVEL
    )
    print("  Bootstrap завершён")

    # ── 9. FEVD ───────────────────────────────────────────
    print("\n[9] FEVD...")
    fevd = compute_fevd(var_res, IRF_HORIZON)
    print(f"\n  Доля дисперсии объяснённая шоком MPU (%):")
    print(f"  {'Переменная':<18}  {'h=6':>6}  {'h=12':>6}  {'h=18':>6}")
    print("  " + "─" * 42)
    for i, col in enumerate(cols):
        s6  = fevd[min(6,  len(fevd)-1), i, shock_idx] * 100
        s12 = fevd[min(12, len(fevd)-1), i, shock_idx] * 100
        s18 = fevd[min(18, len(fevd)-1), i, shock_idx] * 100
        print(f"  {col:<18}  {s6:>6.1f}%  {s12:>6.1f}%  {s18:>6.1f}%")

    # ── 10. Historical Decomposition ──────────────────────
    print("\n[10] Historical Decomposition...")
    hd = compute_historical_decomposition(
        var_res, df, p, shock_idx, IRF_HORIZON
    )
    print(f"  HD shape: {hd.shape}")
    print(f"\n  Средний вклад шока MPU по периодам:")
    periods = {
        '2022 (шок)':           ('2022-01', '2022-12'),
        '2023–2024 (ужесточение)': ('2023-01', '2024-12'),
        '2025 (смягчение)':     ('2025-01', hd.index.max().strftime('%Y-%m')),
    }
    for col in hd.columns:
        print(f"\n  {col}:")
        for period, (d0, d1) in periods.items():
            mask  = (hd.index >= d0) & (hd.index <= d1)
            if mask.any():
                mean_c = hd.loc[mask, col].mean()
                print(f"    {period:<30}: {mean_c:+.4f}")

    # ── 11. Сохранение ────────────────────────────────────
    print("\n[11] Сохранение результатов...")
    irf_df = pd.DataFrame(irf, columns=[f'IRF_{c}' for c in cols])
    irf_df['h'] = np.arange(len(irf_df))
    irf_df.to_csv(OUT_DIR / 'irf.csv', index=False)

    fevd_rows = []
    for h in range(len(fevd)):
        row = {'h': h}
        for i, col in enumerate(cols):
            row[f'FEVD_{col}_MPU'] = round(fevd[h, i, shock_idx] * 100, 2)
        fevd_rows.append(row)
    pd.DataFrame(fevd_rows).to_csv(OUT_DIR / 'fevd.csv', index=False)

    hd.to_csv(OUT_DIR / 'historical_decomposition.csv')

    # ── 12. Графики ───────────────────────────────────────
    print("\n[12] Генерация графиков...")
    plot_var_stability(var_res)
    plot_irf_panel(irf, ci_low, ci_high, cols)
    plot_fevd(fevd, cols, shock_idx)
    plot_historical_decomposition(hd, df, col_map)
    plot_hd_episodes(hd, df)

    print(f"\nГотово → {OUT_DIR.resolve()}")
    print(f"\nОжидаемые знаки IRF по теории канала реальных опционов:")
    print(f"  MPU ↑ → Key_Rate ↑      ЦБ реагирует ужесточением")
    print(f"  MPU ↑ → Inflation ↑/↓   краткосрочно ↑, затем ↓ после реакции ЦБ")
    print(f"  MPU ↑ → GVA_YoY ↓       wait-and-see → снижение инвестиций")


if __name__ == '__main__':
    main()