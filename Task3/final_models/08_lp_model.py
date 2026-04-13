# 08_lp_model.py
# Local Projections (LP) модель для анализа макро-финансового влияния MPU.
#
# Метод: Jordà (2005) Local Projections.
# Идея: вместо одной VAR-модели строим серию регрессий —
# по одной на каждый горизонт h. Это даёт импульсные отклики (IRF)
# напрямую, без накопления ошибок спецификации как в VAR.
#
# Модель для каждой переменной y_j и горизонта h:
#
#   y_j(t+h) - y_j(t-1) = α_h
#                        + β_h · shock_MPU(t)       ← шок MPU
#                        + Σ_p γ_h,p · y(t-p)       ← лаги всех переменных
#                        + ε_h(t)
#
# β_h — это и есть IRF: отклик y_j через h периодов на шок MPU размером 1 std.
#
# Переменные модели:
#   MPU_decay   — неопределённость ДКП (z-score)
#   Key_Rate    — ключевая ставка ЦБ РФ (%)
#   Inflation   — инфляция г/г (%)
#   GVA_YoY     — темп роста ВДС г/г (%)
#
# Стационарность:
#   MPU_decay:  z-score, стационарен по построению
#   Key_Rate:   возможно I(1) → проверяем ADF, при необходимости берём Δ
#   Inflation:  возможно I(1) → аналогично
#   GVA_YoY:   год-к-году темп, обычно стационарен
#
# Идентификация шока MPU:
#   Рекурсивная (Cholesky): MPU стоит первым в системе →
#   его шок не зависит от одновременных изменений других переменных.
#   Экономическое обоснование: опционный рынок фиксирует неопределённость
#   ДО решения ЦБ на заседании.
#
# Результаты -> Task3/final_models/lp_model/
# Запуск: python 08_lp_model.py

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))
MPU_DIR  = BASE_DIR / 'mpu_models'
OUT_DIR  = BASE_DIR / 'lp_model'
PLOT_DIR = OUT_DIR / 'plots'

from data_loading.case_2 import get_key_rate_dataframe
from data_loading.extra_data import get_gva_monthly_dataframe

# ── Настройки ────────────────────────────────────────────────
MAX_HORIZON  = 18   # максимальный горизонт IRF (месяцев)
N_LAGS       = 2    # число лагов в контрольных переменных LP
N_BOOT       = 500  # итераций bootstrap для доверительных интервалов
CI_LEVEL     = 0.68 # уровень доверительного интервала (68% ≈ 1 std)
SHOCK_SIZE   = 1.0  # размер шока в std MPU
SAMPLE_START = '2019-03-01'


# ════════════════════════════════════════════════════════════
# Загрузка и подготовка данных
# ════════════════════════════════════════════════════════════

def load_mpu() -> pd.Series:
    p  = MPU_DIR / 'MPU_decay.csv'
    df = pd.read_csv(p, parse_dates=['Date'])
    s  = df.set_index('Date')['MPU_decay_norm'].sort_index().dropna()
    s.index = pd.to_datetime(s.index)
    return s


def load_macro() -> pd.DataFrame:
    """
    Загружает и объединяет макропеременные:
      Key_Rate, Inflation, GVA_YoY.

    Для стационарности:
      Key_Rate  → берём уровень; если ADF p>0.10 → добавим Δ
      Inflation → берём уровень; аналогично
      GVA_YoY  → год-к-году, обычно стационарен
    """
    key_rate_df = get_key_rate_dataframe()
    kr = (key_rate_df[['Date', 'Key Rate', 'Inflation']]
          .dropna(subset=['Date'])
          .rename(columns={'Key Rate': 'Key_Rate'})
          .sort_values('Date').set_index('Date'))
    kr.index = pd.to_datetime(kr.index)

    gva_df = get_gva_monthly_dataframe()
    gva_df['Date'] = pd.to_datetime(gva_df['Date'])
    gva_df = gva_df.sort_values('Date').set_index('Date')
    gva_df['GVA_YoY'] = (gva_df['Value'] / gva_df['Value'].shift(12) - 1) * 100

    macro = pd.concat([
        kr[['Key_Rate', 'Inflation']],
        gva_df[['GVA_YoY']],
    ], axis=1)

    return macro


def prepare_dataset() -> pd.DataFrame:
    """
    Объединяет MPU и макропеременные в единый DataFrame.
    Применяет ADF тест и при необходимости берёт первые разности.
    """
    mpu   = load_mpu().rename('MPU_decay')
    macro = load_macro()

    # Объединение по дате (inner join)
    df = pd.merge(
        mpu.reset_index().rename(columns={'index': 'Date',
                                           'Date': 'Date'}),
        macro.reset_index(),
        on='Date', how='inner'
    ).set_index('Date').sort_index()

    df = df[df.index >= SAMPLE_START].dropna()

    print(f"Объединённый датасет:")
    print(f"  Период: {df.index.min().date()} — {df.index.max().date()}")
    print(f"  N = {len(df)} наблюдений")
    print(f"  Переменные: {list(df.columns)}")

    return df


def check_stationarity(df: pd.DataFrame) -> dict:
    """
    ADF тест для каждой переменной.
    Возвращает dict с рекомендацией: 'level' или 'diff'.
    """
    print(f"\nТест ADF на стационарность (H0: единичный корень):")
    print(f"  {'Переменная':<15}  {'ADF stat':>9}  {'p-value':>8}  "
          f"{'Вывод'}")
    print("  " + "─" * 55)

    results = {}
    for col in df.columns:
        s = df[col].dropna()
        adf_stat, pval, _, nobs, crit, _ = adfuller(s, maxlag=3,
                                                      autolag='AIC')
        stationary = pval < 0.10
        verdict = "стационарен" if stationary else "I(1)? → берём Δ"
        transform = 'level' if stationary else 'diff'
        results[col] = transform
        print(f"  {col:<15}  {adf_stat:>9.3f}  {pval:>8.4f}  {verdict}")

    return results


def apply_transforms(df: pd.DataFrame,
                      transforms: dict) -> pd.DataFrame:
    """
    Применяет трансформации по результатам ADF.
    MPU всегда остаётся в уровнях (z-score стационарен).
    """
    result = pd.DataFrame(index=df.index)
    col_names = {}

    for col in df.columns:
        if col == 'MPU_decay' or transforms.get(col) == 'level':
            result[col] = df[col]
            col_names[col] = col
        else:
            new_col = f'D_{col}'
            result[new_col] = df[col].diff()
            col_names[col] = new_col
            print(f"  {col} → взята первая разность: {new_col}")

    return result.dropna()


# ════════════════════════════════════════════════════════════
# Ортогонализация шока MPU (Cholesky)
# ════════════════════════════════════════════════════════════

def orthogonalize_mpu_shock(df: pd.DataFrame,
                              mpu_col: str,
                              n_lags: int) -> pd.Series:
    """
    Получает ортогональный шок MPU: остаток от регрессии
    MPU(t) на его собственные лаги и лаги других переменных.

    Это реализует Cholesky идентификацию: шок MPU —
    это часть движения MPU, которую не объясняют лаги системы.

    Экономически: неожиданное изменение неопределённости,
    не связанное с прошлым состоянием экономики.
    """
    other_cols = [c for c in df.columns if c != mpu_col]

    # Собираем регрессоры: лаги MPU + лаги всех других переменных
    X_parts = []
    for lag in range(1, n_lags + 1):
        for col in [mpu_col] + other_cols:
            lagged = df[col].shift(lag)
            lagged.name = f'{col}_lag{lag}'
            X_parts.append(lagged)

    X = pd.concat(X_parts, axis=1).dropna()
    y = df[mpu_col].loc[X.index]

    X_const = sm.add_constant(X)
    model   = sm.OLS(y, X_const).fit()
    shock   = model.resid

    # Нормируем на std чтобы шок = 1 std
    shock_std = shock / shock.std()

    print(f"\nОртогонализация шока MPU:")
    print(f"  R² регрессии MPU на лаги: {model.rsquared:.3f}")
    print(f"  std остатков (до норм.):  {shock.std():.4f}")
    print(f"  N наблюдений шока:        {len(shock_std)}")

    return shock_std


# ════════════════════════════════════════════════════════════
# Local Projections
# ════════════════════════════════════════════════════════════

def run_lp(df: pd.DataFrame,
            shock: pd.Series,
            target_col: str,
            max_horizon: int = MAX_HORIZON,
            n_lags: int = N_LAGS) -> pd.DataFrame:
    """
    Local Projections для одной целевой переменной.

    Для каждого горизонта h = 0..max_horizon:

        y(t+h) - y(t-1) = α + β_h · shock(t)
                        + Σ_{p=1}^{n_lags} γ_p · y_all(t-p+1)
                        + ε(t)

    β_h — IRF: кумулятивный отклик y через h месяцев на шок MPU.

    Используем HAC (Newey-West) стандартные ошибки для
    устранения автокорреляции перекрывающихся регрессий.

    Параметры:
        df          — DataFrame со всеми переменными
        shock       — ортогональный шок MPU (нормированный)
        target_col  — имя целевой переменной
        max_horizon — максимальный горизонт
        n_lags      — число лагов контрольных переменных

    Возвращает:
        DataFrame: h, beta, se, t_stat, pval, ci_low, ci_high
    """
    records = []

    for h in range(0, max_horizon + 1):
        # Целевая переменная: кумулятивное изменение от t-1 до t+h
        y_fwd = df[target_col].shift(-h) - df[target_col].shift(1)

        # Контрольные переменные: лаги всех переменных
        controls = []
        for lag in range(1, n_lags + 1):
            for col in df.columns:
                c = df[col].shift(lag - 1)
                c.name = f'{col}_L{lag}'
                controls.append(c)

        X_ctrl = pd.concat(controls, axis=1)

        # Объединяем всё
        data = pd.concat([y_fwd.rename('y'), shock.rename('shock'),
                           X_ctrl], axis=1).dropna()

        if len(data) < n_lags + 5:
            continue

        y = data['y'].values
        X = sm.add_constant(data.drop(columns='y').values)

        try:
            model = sm.OLS(y, X).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': max(1, h // 2)}
            )
            beta   = float(model.params[1])   # коэф. при shock
            se     = float(model.bse[1])
            t_stat = float(model.tvalues[1])
            pval   = float(model.pvalues[1])
        except Exception:
            continue

        records.append(dict(
            h      = h,
            beta   = round(beta,   4),
            se     = round(se,     4),
            t_stat = round(t_stat, 4),
            pval   = round(pval,   4),
            ci_low  = round(beta - 1.645 * se, 4),  # 90% CI
            ci_high = round(beta + 1.645 * se, 4),
            ci_low_68  = round(beta - 1.0 * se, 4),  # 68% CI
            ci_high_68 = round(beta + 1.0 * se, 4),
            n_obs  = len(data),
        ))

    return pd.DataFrame(records)


def run_all_lp(df: pd.DataFrame,
                shock: pd.Series,
                max_horizon: int = MAX_HORIZON,
                n_lags: int = N_LAGS) -> dict[str, pd.DataFrame]:
    """
    Запускает LP для всех переменных системы.
    """
    results = {}
    target_cols = [c for c in df.columns if c != 'MPU_decay']
    # Добавляем сам MPU для автоотклика
    all_targets = ['MPU_decay'] + target_cols

    print(f"\nLocal Projections (h=0..{max_horizon}, p={n_lags}):")
    for col in all_targets:
        irf = run_lp(df, shock, col,
                     max_horizon=max_horizon, n_lags=n_lags)
        results[col] = irf
        sig_hs = irf[irf['pval'] < 0.10]['h'].tolist()
        peak_h = int(irf.loc[irf['beta'].abs().idxmax(), 'h']) if not irf.empty else -1
        print(f"  {col:<18}: пик h={peak_h}M  "
              f"значимо (p<0.10) на h={sig_hs}")

    return results


# ════════════════════════════════════════════════════════════
# Bootstrap доверительные интервалы
# ════════════════════════════════════════════════════════════

def bootstrap_lp(df: pd.DataFrame,
                  shock: pd.Series,
                  target_col: str,
                  max_horizon: int = MAX_HORIZON,
                  n_lags: int = N_LAGS,
                  n_boot: int = N_BOOT,
                  ci_level: float = CI_LEVEL) -> pd.DataFrame:
    """
    Bootstrap доверительные интервалы для IRF.

    Метод: residual bootstrap (Efron).
    1. Оцениваем LP на исходных данных → получаем остатки
    2. Случайно пересэмплируем остатки → строим псевдоданные
    3. Повторяем LP на псевдоданных → получаем распределение β_h
    4. Квантили этого распределения → CI

    Параметры:
        ci_level — ширина CI (0.68 ≈ 1 std, 0.90 → ±1.645 std)
    """
    alpha = (1 - ci_level) / 2
    boot_betas = {h: [] for h in range(0, max_horizon + 1)}

    # Исходные остатки
    base_irf = run_lp(df, shock, target_col, max_horizon, n_lags)

    np.random.seed(42)
    for _ in range(n_boot):
        # Пересэмплируем строки датафрейма (блочный bootstrap с блоком 1)
        n = len(df)
        idx_boot = np.random.choice(n, size=n, replace=True)
        idx_boot = np.sort(idx_boot)  # сохраняем временной порядок частично

        df_boot    = df.iloc[idx_boot].reset_index(drop=True)
        shock_boot = shock.iloc[:len(df_boot)].reset_index(drop=True) \
                     if len(shock) >= len(df_boot) else \
                     pd.Series(np.random.choice(shock.values, len(df_boot)))

        boot_irf = run_lp(df_boot, shock_boot, target_col,
                           max_horizon, n_lags)

        for _, row in boot_irf.iterrows():
            h = int(row['h'])
            if h in boot_betas:
                boot_betas[h].append(row['beta'])

    # Формируем CI
    ci_records = []
    for _, row in base_irf.iterrows():
        h   = int(row['h'])
        bbs = boot_betas.get(h, [])
        if len(bbs) < 10:
            ci_records.append(dict(
                h=h, beta=row['beta'],
                ci_low=row['ci_low_68'], ci_high=row['ci_high_68'],
                ci_low_90=row['ci_low'],  ci_high_90=row['ci_high'],
            ))
        else:
            ci_records.append(dict(
                h=h, beta=row['beta'],
                ci_low=np.quantile(bbs, alpha),
                ci_high=np.quantile(bbs, 1 - alpha),
                ci_low_90=np.quantile(bbs, 0.05),
                ci_high_90=np.quantile(bbs, 0.95),
            ))

    return pd.DataFrame(ci_records)


# ════════════════════════════════════════════════════════════
# Графики
# ════════════════════════════════════════════════════════════

def _save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {name}")


# Метки и ожидаемые знаки для каждой переменной
VAR_META = {
    'MPU_decay':  ('MPU неопределённость',  '#E65100', '+'),
    'D_Key_Rate': ('Δ Ключевая ставка',      '#1565C0', '+'),
    'Key_Rate':   ('Ключевая ставка',        '#1565C0', '+'),
    'D_Inflation':('Δ Инфляция',             '#C62828', '+/-'),
    'Inflation':  ('Инфляция',               '#C62828', '+/-'),
    'GVA_YoY':   ('ВДС YoY (%)',            '#27AE60', '-'),
}


def plot_irf_panel(irf_dict: dict,
                    title: str = 'IRF на шок MPU_decay (+1 std)',
                    save_name: str = 'irf_panel.png') -> None:
    """
    Основной график: IRF для всех переменных в одной фигуре.
    4 панели: MPU (автоотклик), Key_Rate, Inflation, GVA_YoY.
    """
    var_order = [k for k in ['MPU_decay',
                              'D_Key_Rate', 'Key_Rate',
                              'D_Inflation', 'Inflation',
                              'GVA_YoY']
                 if k in irf_dict]

    n   = len(var_order)
    fig = plt.figure(figsize=(6 * min(n, 2), 5 * ((n + 1) // 2)))
    gs  = gridspec.GridSpec((n + 1) // 2, min(n, 2),
                             hspace=0.45, wspace=0.35)
    fig.suptitle(title, fontweight='bold', fontsize=13)

    for idx, var in enumerate(var_order):
        ax  = fig.add_subplot(gs[idx // 2, idx % 2])
        irf = irf_dict[var]
        if irf is None or irf.empty:
            ax.set_visible(False)
            continue

        meta  = VAR_META.get(var, (var, '#333', '?'))
        label, color, expected = meta
        hs    = irf['h'].values
        betas = irf['beta'].values

        # Заливка CI
        if 'ci_low' in irf.columns:
            ax.fill_between(hs, irf['ci_low'], irf['ci_high'],
                            alpha=0.20, color=color, label=f'{int(CI_LEVEL*100)}% CI')
        if 'ci_low_90' in irf.columns:
            ax.fill_between(hs, irf['ci_low_90'], irf['ci_high_90'],
                            alpha=0.10, color=color, label='90% CI')

        ax.plot(hs, betas, color=color, lw=2.5, zorder=5,
                label='IRF')
        ax.axhline(0, color='black', lw=1.2, ls='--', alpha=0.6)
        ax.axvline(0, color='grey', lw=0.8, ls=':', alpha=0.5)

        # Отмечаем значимые точки
        if 'pval' in irf.columns:
            sig = irf[irf['pval'] < 0.10]
            ax.scatter(sig['h'], sig['beta'],
                       color=color, s=50, zorder=6,
                       marker='*', label='p<0.10')

        ax.set_title(f'{label}\n(ожидаемый знак: {expected})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Горизонт h (месяцев)')
        ax.set_ylabel('Кум. отклик')
        ax.set_xticks(range(0, max(hs) + 1, 3))
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    _save(fig, save_name)


def plot_irf_single(irf: pd.DataFrame,
                     var_name: str,
                     save_name: str) -> None:
    """Детальный график IRF для одной переменной."""
    if irf is None or irf.empty:
        return

    meta  = VAR_META.get(var_name, (var_name, '#333', '?'))
    label, color, expected = meta
    hs    = irf['h'].values
    betas = irf['beta'].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(hs, irf['ci_low_90'], irf['ci_high_90'],
                    alpha=0.10, color=color, label='90% CI')
    ax.fill_between(hs, irf['ci_low'], irf['ci_high'],
                    alpha=0.25, color=color, label=f'{int(CI_LEVEL*100)}% CI')
    ax.plot(hs, betas, color=color, lw=2.5, label='IRF (точечная оценка)')

    # Значимые горизонты
    if 'pval' in irf.columns:
        sig = irf[irf['pval'] < 0.10]
        ax.scatter(sig['h'], sig['beta'], color=color,
                   s=80, zorder=6, marker='*', label='p<0.10')

    ax.axhline(0, color='black', lw=1.5, ls='--', alpha=0.7)
    ax.set_title(
        f'IRF: шок MPU (+1 std) → {label}\n'
        f'Теоретически ожидаемый знак: {expected}',
        fontweight='bold', fontsize=12
    )
    ax.set_xlabel('Горизонт h (месяцев)')
    ax.set_ylabel('Кумулятивный отклик')
    ax.set_xticks(range(0, max(hs) + 1, 2))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    _save(fig, save_name)


def plot_data_overview(df: pd.DataFrame) -> None:
    """Временные ряды всех переменных модели."""
    n   = len(df.columns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle('Переменные LP модели', fontweight='bold', fontsize=13)

    for ax, col in zip(axes, df.columns):
        meta  = VAR_META.get(col, (col, '#333', ''))
        label, color, _ = meta
        ax.plot(df.index, df[col], color=color, lw=2)
        ax.fill_between(df.index, df[col], alpha=0.08, color=color)
        ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(alpha=0.2)

    import matplotlib.dates as mdates
    axes[-1].set_xlabel('Дата')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    _save(fig, 'data_overview.png')


def plot_shock_series(shock: pd.Series,
                       df: pd.DataFrame) -> None:
    """Временной ряд шока MPU с разметкой ключевых событий."""
    import matplotlib.dates as mdates

    events = {
        '2020-03': 'COVID',
        '2022-02': 'Шок КС +11.5пп',
        '2023-07': 'Цикл ужесточения',
        '2025-06': 'Цикл смягчения',
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(shock.index, shock.values,
           color=np.where(shock.values > 0, '#E65100', '#1565C0'),
           alpha=0.7, width=25, label='Шок MPU')
    ax.axhline(0, color='black', lw=1.2)
    ax.axhline(1, color='grey', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(-1, color='grey', lw=0.8, ls='--', alpha=0.5)

    for date_str, label in events.items():
        try:
            d = pd.Timestamp(date_str)
            ax.axvline(d, color='black', lw=1.2, ls=':', alpha=0.6)
            ax.text(d, ax.get_ylim()[1] * 0.85, label,
                    rotation=90, fontsize=8, ha='right', va='top')
        except Exception:
            pass

    ax.set_title('Ортогональный шок MPU_decay\n'
                 '(нормирован: 1 ед. = 1 std неожиданного роста неопределённости)',
                 fontweight='bold')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Шок (std)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    _save(fig, 'shock_series.png')


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"08 | LOCAL PROJECTIONS  (h=0..{MAX_HORIZON}, p={N_LAGS})")
    print("=" * 65)

    # ── 1. Данные ──────────────────────────────────────────
    print("\n[1] Загрузка и подготовка данных...")
    df_raw = prepare_dataset()

    # ── 2. Стационарность ─────────────────────────────────
    print("\n[2] Тест на стационарность...")
    transforms = check_stationarity(df_raw)

    print("\n  Применяемые трансформации:")
    df = apply_transforms(df_raw, transforms)
    print(f"  Итоговые переменные: {list(df.columns)}")
    print(f"  N после трансформаций: {len(df)}")

    # ── 3. Шок MPU ─────────────────────────────────────────
    print("\n[3] Ортогонализация шока MPU...")
    mpu_col = 'MPU_decay'
    shock   = orthogonalize_mpu_shock(df, mpu_col, N_LAGS)

    # ── 4. Local Projections ───────────────────────────────
    print("\n[4] Local Projections (аналитические SE)...")
    irf_analytic = run_all_lp(df, shock,
                               max_horizon=MAX_HORIZON,
                               n_lags=N_LAGS)

    # ── 5. Bootstrap CI ────────────────────────────────────
    print(f"\n[5] Bootstrap CI ({N_BOOT} итераций)...")
    irf_boot = {}
    target_cols = list(df.columns)
    for col in target_cols:
        print(f"  Bootstrap: {col}...", end=' ')
        irf_b = bootstrap_lp(df, shock, col,
                              max_horizon=MAX_HORIZON,
                              n_lags=N_LAGS,
                              n_boot=N_BOOT,
                              ci_level=CI_LEVEL)
        # Добавляем pval из аналитической оценки
        analytic = irf_analytic.get(col, pd.DataFrame())
        if not analytic.empty and 'pval' in analytic.columns:
            irf_b = irf_b.merge(
                analytic[['h', 'pval', 'se']],
                on='h', how='left'
            )
        irf_boot[col] = irf_b
        print("готово")

    # ── 6. Сохранение ──────────────────────────────────────
    print("\n[6] Сохранение результатов...")
    for col, irf in irf_boot.items():
        fname = f'irf_{col}.csv'
        irf.to_csv(OUT_DIR / fname, index=False)

    # Сводная таблица пиков и значимых горизонтов
    summary = []
    for col, irf in irf_boot.items():
        if irf.empty:
            continue
        peak_idx  = irf['beta'].abs().idxmax()
        peak_h    = int(irf.loc[peak_idx, 'h'])
        peak_beta = float(irf.loc[peak_idx, 'beta'])
        sig_hs    = []
        if 'pval' in irf.columns:
            sig_hs = irf[irf['pval'] < 0.10]['h'].tolist()
        summary.append(dict(
            variable     = col,
            peak_h       = peak_h,
            peak_beta    = round(peak_beta, 4),
            sign         = '+' if peak_beta > 0 else '-',
            sig_horizons = str(sig_hs),
        ))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / 'irf_summary.csv', index=False)

    print("\nСводка IRF (пики откликов):")
    print(f"  {'Переменная':<18}  {'Пик h':>6}  "
          f"{'Beta':>8}  {'Знак':>5}  Значимые горизонты")
    print("  " + "─" * 65)
    for _, row in summary_df.iterrows():
        print(f"  {row['variable']:<18}  {row['peak_h']:>6}M  "
              f"  {row['peak_beta']:>8.4f}  {row['sign']:>5}  "
              f"  {row['sig_horizons']}")

    # ── 7. Графики ─────────────────────────────────────────
    print("\n[7] Генерация графиков...")
    plot_data_overview(df)
    plot_shock_series(shock, df)
    plot_irf_panel(irf_boot,
                   title=f'IRF: шок MPU (+1 std) → макропеременные\n'
                         f'LP(p={N_LAGS}), {int(CI_LEVEL*100)}%/90% CI (bootstrap)',
                   save_name='irf_panel.png')

    for col in target_cols:
        if col in irf_boot:
            plot_irf_single(irf_boot[col], col,
                            f'irf_{col}_detail.png')

    print(f"\nГотово → {OUT_DIR.resolve()}")
    print(f"\nИнтерпретация (ожидаемые знаки по теории):")
    print(f"  MPU ↑ → Key_Rate ↑     ЦБ реагирует ужесточением")
    print(f"  MPU ↑ → Inflation ↑/↓  кратко ↑ (ожидания), затем ↓ (реакция ЦБ)")
    print(f"  MPU ↑ → GVA_YoY ↓      wait-and-see → меньше инвестиций")


if __name__ == '__main__':
    main()
