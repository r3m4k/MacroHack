# rnd_pipeline.py
# Пайплайн извлечения RND и расчёта статистик для индекса MPU
# Кейс 2: Индекс неопределённости ДКП на основе поверхности вменённой волатильности
#
# Принцип построения MPU:
#   - Форвард F извлекается ТОЛЬКО из поверхности IV (ATM-point улыбки)
#   - Ключевая ставка ЦБ РФ — только справочные линии на графиках
#   - Агрегация по срокам: два метода — Decay weights и PCA

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Совместимость np.trapz -> np.trapezoid (NumPy >= 2.0)
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from data_loading.case_2 import get_case_2_IV, get_key_rate_dataframe


# ============================================================
# НАСТРОЙКИ
# ============================================================

PLOT_DIR = Path('results') / 'plots'

COLORS = {
    '1M':        '#2196F3',
    '3M':        '#4CAF50',
    '6M':        '#FF9800',
    '1Y':        '#E91E63',
    'decay':     '#E65100',
    'pca':       '#6A1B9A',
    'key_rate':  '#212121',
    'inflation': '#F44336',
    'iqr':       '#1565C0',
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
    Загружает поверхность IV и данные по ключевой ставке.
    key_rate_df используется ТОЛЬКО для справочных линий на графиках.
    """
    iv_df       = get_case_2_IV()
    key_rate_df = get_key_rate_dataframe()

    print("=" * 60)
    print("IV данные:")
    print(iv_df.head())
    print(f"Период: {iv_df['Date'].min().date()} -> {iv_df['Date'].max().date()}")
    print(f"Дат: {iv_df['Date'].nunique()} | "
          f"Сроков: {iv_df['Maturity'].nunique()} | "
          f"Страйков: {iv_df['Strike'].nunique()}")
    print("\nКлючевая ставка (справочно):")
    print(key_rate_df.head(5))
    print("=" * 60)

    return iv_df, key_rate_df


# ============================================================
# ШАГ 1: ATM-форвард из улыбки IV
# ============================================================

def get_atm_forward(strikes: np.ndarray, ivols: np.ndarray) -> float:
    """
    Извлекает форвард F как страйк с минимальной вменённой волатильностью.

    Обоснование: по теории безарбитражного ценообразования форвард
    соответствует минимуму улыбки IV — при этом страйке колл и пут
    равны по стоимости (put-call parity). Внешние данные не нужны.
    """
    return float(strikes[np.argmin(ivols)])


# ============================================================
# ШАГ 2: Модель Башелье
# ============================================================

def bachelier_call(F: float, K: float, T: float, sigma_n: float) -> float:
    """
    Цена колл-опциона по нормальной модели Башелье.
    Волатильность задана в абсолютных единицах (нормальная IV).
    """
    if T <= 1e-10:
        return max(F - K, 0.0)
    denom = sigma_n * np.sqrt(T)
    if denom < 1e-10:
        return max(F - K, 0.0)
    d = (F - K) / denom
    return denom * (d * norm.cdf(d) + norm.pdf(d))


# ============================================================
# ШАГ 3: Извлечение RND для одной даты и срока
# ============================================================

def extract_rnd(date: pd.Timestamp,
                iv_df: pd.DataFrame,
                maturity: str = '3M',
                n_points: int = 1000,
                verbose: bool = False) -> dict | None:
    """
    Извлекает RND методом Breeden-Litzenberger.
    Форвард = ATM-point улыбки IV. Внешние данные не используются.

    Алгоритм:
        1. Срез IV(K) для даты и срока T
        2. F = страйк с min(IV)
        3. Кубический сплайн по улыбке
        4. Цены коллов по Башелье на плотной сетке
        5. d2C/dK2 -> RND (Breeden-Litzenberger)
        6. Моменты и квантили
    """
    mask     = (iv_df['Date'] == date) & (iv_df['Maturity'] == maturity)
    slice_df = iv_df[mask].sort_values('Strike').copy()

    if len(slice_df) < 5:
        if verbose:
            print(f"  [SKIP] {date.date()} {maturity}: мало точек ({len(slice_df)})")
        return None

    strikes = slice_df['Strike'].values
    ivols   = slice_df['Volatility'].values
    T_val   = slice_df['Maturity (year fraction)'].iloc[0]

    F = get_atm_forward(strikes, ivols)

    cs      = CubicSpline(strikes, ivols, extrapolate=False)
    K_grid  = np.linspace(strikes.min(), strikes.max(), n_points)
    dK      = K_grid[1] - K_grid[0]
    iv_grid = cs(K_grid)
    iv_grid = np.where(np.isnan(iv_grid),
                       np.where(K_grid < strikes[0], ivols[0], ivols[-1]),
                       iv_grid)
    iv_grid = np.maximum(iv_grid, 0.01)

    C_grid = np.array([bachelier_call(F, K, T_val, iv)
                       for K, iv in zip(K_grid, iv_grid)])

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
    skew     = np.trapz((K_grid - mean_rnd) ** 3 * RND, K_grid) / std_rnd ** 3
    kurt     = np.trapz((K_grid - mean_rnd) ** 4 * RND, K_grid) / std_rnd ** 4

    # Квантили
    cdf = np.clip(np.cumsum(RND) * dK, 0.0, 1.0)
    cdf = cdf / cdf[-1]

    def q(p):
        return float(K_grid[np.clip(np.searchsorted(cdf, p), 0, len(K_grid) - 1)])

    Q01, Q10, Q25 = q(0.01), q(0.10), q(0.25)
    Q50, Q75, Q90 = q(0.50), q(0.75), q(0.90)
    Q99           = q(0.99)

    return {
        'Date': date,        'Maturity':   maturity,
        'F':    F,           'Mean':       mean_rnd,
        'Std':  std_rnd,     'Skew':       skew,
        'Kurt': kurt,
        'Q01':  Q01,         'Q10':        Q10,
        'Q25':  Q25,         'Q50':        Q50,
        'Q75':  Q75,         'Q90':        Q90,
        'Q99':  Q99,
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
    """Прогоняет extract_rnd по всем датам и заданным срокам."""
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']

    dates       = sorted(iv_df['Date'].unique())
    all_results = []

    for maturity in maturities:
        print(f"Срок {maturity} ({len(dates)} дат)...", end=' ')
        ok = skip = 0
        for date in dates:
            r = extract_rnd(date, iv_df, maturity=maturity, verbose=verbose)
            if r is not None:
                all_results.append(r)
                ok += 1
            else:
                skip += 1
        print(f"OK={ok}  skip={skip}")

    df = pd.DataFrame(all_results)
    print(f"Итого: {len(df)} наблюдений")
    return df


# ============================================================
# ШАГ 5: Проверка качества
# ============================================================

def check_rnd_quality(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Флаги качества:
        1. |Mean - F| < 2%   — среднее близко к ATM-форварду
        2. Std in [0.1, 15]% — разумный разброс
        3. IQR_9010 > 0
        4. Квантили монотонны
    """
    df = results_df.copy()
    df['flag_mean_ok'] = (df['Mean'] - df['F']).abs() < 2.0
    df['flag_std_ok']  = df['Std'].between(0.1, 15.0)
    df['flag_iqr_ok']  = df['IQR_9010'] > 0
    df['flag_mono_ok'] = (
            (df['Q10'] < df['Q25']) & (df['Q25'] < df['Q50']) &
            (df['Q50'] < df['Q75']) & (df['Q75'] < df['Q90'])
    )
    df['quality_ok'] = (df['flag_mean_ok'] & df['flag_std_ok'] &
                        df['flag_iqr_ok']  & df['flag_mono_ok'])

    n_bad = (~df['quality_ok']).sum()
    if n_bad > 0:
        print(f"Проблемных наблюдений: {n_bad}")
        print(df[~df['quality_ok']][
                  ['Date', 'Maturity', 'F', 'Mean', 'Std', 'IQR_9010',
                   'flag_mean_ok', 'flag_std_ok', 'flag_iqr_ok', 'flag_mono_ok']
              ].to_string())
    else:
        print("Все наблюдения прошли контроль качества")
    return df


# ============================================================
# ШАГ 6: Описательная статистика
# ============================================================

def print_summary(results_df: pd.DataFrame) -> None:
    cols = ['IQR_9010', 'IQR_7525', 'Std', 'Skew', 'Kurt',
            'Tail_right', 'Tail_left', 'Tail_total']
    print("\n" + "=" * 60)
    for mat, grp in results_df.groupby('Maturity'):
        print(f"\n--- {mat} (n={len(grp)}) ---")
        print(grp[cols].describe().round(4).to_string())


# ============================================================
# ШАГ 7: Агрегация MPU по срокам экспирации
# ============================================================

def aggregate_mpu_across_maturities(
        clean_df: pd.DataFrame,
        maturities: list[str] | None = None,
        metric: str = 'IQR_9010',
) -> tuple[pd.DataFrame, dict]:
    """
    Агрегирует MPU по нескольким срокам экспирации в единый индекс.

    Реализует две спецификации и сравнивает их.

    ── Спецификация 1: MPU_decay (взвешенное среднее) ──────────
    Вес срока i: w_i = exp(-lambda*i) / sum_j exp(-lambda*j), lambda = ln(2).

    Обоснование: краткосрочные сроки (1M, 3M) улавливают сигнал
    о ближайших решениях ЦБ. Долгосрочные (1Y+) содержат структурные
    премии за риск, не связанные напрямую с неопределённостью ДКП.

    ── Спецификация 2: MPU_pca (первая главная компонента) ─────
    PCA извлекает общий латентный фактор, объясняющий максимум
    совместной дисперсии всех сроков. Веса определяются данными —
    это делает MPU_pca устойчивым к произвольному выбору lambda.

    ── Сравнение ────────────────────────────────────────────────
    corr > 0.9: результат устойчив к методу — оба индекса валидны.
    corr < 0.7: методы улавливают разные аспекты неопределённости;
                MPU_pca предпочтителен как data-driven.

    Возвращает:
        mpu_df — DataFrame: Date, MPU_decay, MPU_pca,
                            MPU_decay_norm, MPU_pca_norm
        meta   — словарь с весами, нагрузками PCA, дисперсией
    """
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']

    pivot = (
        clean_df[clean_df['Maturity'].isin(maturities)]
        .pivot_table(values=metric, index='Date', columns='Maturity')
        .reindex(columns=maturities)
        .dropna()
    )

    if pivot.empty:
        raise ValueError(
            f"Нет полных данных для агрегации по срокам {maturities}."
        )

    n_mat = pivot.shape[1]
    print(f"\n{'='*60}")
    print(f"АГРЕГАЦИЯ MPU | метрика: {metric} | сроки: {maturities}")
    print(f"Дат для агрегации: {len(pivot)}")
    print("=" * 60)

    # ── 1. Decay-веса ─────────────────────────────────────────
    lam           = np.log(2)
    raw_w         = np.array([np.exp(-lam * i) for i in range(n_mat)])
    decay_weights = raw_w / raw_w.sum()
    mpu_decay     = pd.Series(pivot.values @ decay_weights, index=pivot.index)

    print(f"\nDecay-веса (lambda = ln2 ~ {lam:.3f}):")
    for mat, w in zip(maturities, decay_weights):
        bar = '#' * max(1, int(w * 40))
        print(f"  {mat:>3}  {w:.4f}  {bar}")

    # ── 2. PCA ────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(pivot.values)
    pca_fit  = PCA(n_components=n_mat)
    scores   = pca_fit.fit_transform(X_scaled)

    explained = pca_fit.explained_variance_ratio_
    loadings  = pca_fit.components_[0].copy()

    print(f"\nPCA — объяснённая дисперсия:")
    cum = 0.0
    for i, ev in enumerate(explained, 1):
        cum += ev
        flag = " <- MPU_pca" if i == 1 else ""
        print(f"  PC{i}: {ev:.1%}  (cum. {cum:.1%}){flag}")

    print(f"\nНагрузки PC1 по срокам:")
    for mat, load in zip(maturities, loadings):
        bar  = '#' * max(1, int(abs(load) * 20))
        sign = '+' if load >= 0 else '-'
        print(f"  {mat:>3}  {load:+.4f}  {sign}{bar}")

    # Ориентация PC1: должна расти вместе с MPU_decay
    pc1 = pd.Series(scores[:, 0], index=pivot.index)
    if pc1.corr(mpu_decay) < 0:
        pc1      = -pc1
        loadings = -loadings
        print("\n  [INFO] PC1 инвертирована для согласованности с MPU_decay")
    mpu_pca = pc1

    # ── Нормировка (z-score) ──────────────────────────────────
    def zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std()

    mpu_df = pd.DataFrame({
        'Date':           pivot.index,
        'MPU_decay':      mpu_decay.values,
        'MPU_pca':        mpu_pca.values,
        'MPU_decay_norm': zscore(mpu_decay).values,
        'MPU_pca_norm':   zscore(mpu_pca).values,
    }).reset_index(drop=True)

    # ── Сравнение спецификаций ────────────────────────────────
    rho = float(mpu_df['MPU_decay'].corr(mpu_df['MPU_pca']))
    print(f"\nКорреляция MPU_decay <-> MPU_pca: {rho:.4f}", end="  ")
    if rho > 0.90:
        print("-> высокая, результат устойчив к методу")
    elif rho > 0.70:
        print("-> умеренная, методы частично расходятся")
    else:
        print("-> низкая, методы улавливают разные аспекты")

    print(f"\nОписательная статистика:")
    print(mpu_df[['MPU_decay', 'MPU_pca']].describe().round(4).to_string())

    meta = {
        'pca':           pca_fit,
        'scaler':        scaler,
        'loadings':      dict(zip(maturities, loadings)),
        'explained':     {f'PC{i+1}': v for i, v in enumerate(explained)},
        'decay_weights': dict(zip(maturities, decay_weights)),
        'maturities':    maturities,
        'metric':        metric,
        'corr':          rho,
    }
    return mpu_df, meta


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ГРАФИКОВ
# ============================================================

def _save_or_show(fig: plt.Figure, save: bool, filename: str) -> None:
    if save:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        path = PLOT_DIR / filename
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"  saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def _compute_rnd_arrays(date, iv_df, maturity, n_points=1000):
    """Возвращает (K_grid, RND, F, result) для одной даты/срока."""
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
    F       = result['F']

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
# ВИЗУАЛИЗАЦИИ 1–8: RND и компоненты MPU
# ============================================================

def plot_iv_smile(date, iv_df, key_rate_df=None,
                  maturities=None, save=True):
    """Улыбка IV(K): рыночные точки + сплайн + ATM-форвард."""
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']
    n = len(maturities)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    kr_str = ""
    if key_rate_df is not None:
        past = key_rate_df[key_rate_df['Date'] <= date]
        if not past.empty:
            kr_str = f"  |  КС (справ.): {past.iloc[-1]['Key Rate']:.2f}%"

    fig.suptitle(f'Улыбка IV  |  {date.date()}{kr_str}',
                 fontsize=13, fontweight='bold')

    for ax, mat in zip(axes, maturities):
        mask = (iv_df['Date'] == date) & (iv_df['Maturity'] == mat)
        sl   = iv_df[mask].sort_values('Strike')
        if sl.empty:
            ax.set_title(f'{mat} — нет данных')
            continue

        strikes, ivols = sl['Strike'].values, sl['Volatility'].values
        F_atm = get_atm_forward(strikes, ivols)
        cs    = CubicSpline(strikes, ivols, extrapolate=False)
        Kf    = np.linspace(strikes.min(), strikes.max(), 500)
        ivf   = np.where(np.isnan(cs(Kf)),
                         np.interp(Kf, strikes, ivols), cs(Kf))
        color = COLORS.get(mat, '#333')
        ax.plot(Kf, ivf, color=color, lw=2, label='Сплайн')
        ax.scatter(strikes, ivols, color=color, s=40, zorder=5,
                   label='Рынок', edgecolors='white', lw=0.5)
        ax.axvline(F_atm, color='black', lw=1.2, ls='--',
                   label=f'ATM={F_atm:.1f}%')
        ax.set_title(mat)
        ax.set_xlabel('Страйк (%)')
        ax.set_ylabel('IV')
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save, f'iv_smile_{date.date()}.png')


def plot_rnd(date, iv_df, key_rate_df=None,
             maturities=None, save=True):
    """RND с квантилями для одной даты."""
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']
    n = len(maturities)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    kr_str = ""
    if key_rate_df is not None:
        past = key_rate_df[key_rate_df['Date'] <= date]
        if not past.empty:
            kr_str = f"  |  КС (справ.): {past.iloc[-1]['Key Rate']:.2f}%"

    fig.suptitle(f'RND  |  {date.date()}{kr_str}',
                 fontsize=13, fontweight='bold')

    for ax, mat in zip(axes, maturities):
        out = _compute_rnd_arrays(date, iv_df, mat)
        if out is None:
            ax.set_title(f'{mat} — нет данных')
            continue
        K_grid, RND, F, res = out
        color         = COLORS.get(mat, '#1565C0')
        Q10, Q50, Q90 = res['Q10'], res['Q50'], res['Q90']
        ax.fill_between(K_grid, RND,
                        where=(K_grid >= Q10) & (K_grid <= Q90),
                        alpha=0.25, color=color, label='Q10-Q90')
        ax.plot(K_grid, RND, color=color, lw=2)
        y_top = RND.max()
        for qv, ql, ls in [(Q10, 'Q10', ':'), (Q50, 'Q50', '--'),
                           (Q90, 'Q90', ':')]:
            ax.axvline(qv, color=color, lw=1.2, ls=ls)
            ax.text(qv, y_top * 0.05, f'{ql}\n{qv:.1f}%',
                    ha='center', fontsize=7, color=color)
        ax.axvline(F, color='black', lw=1.5, ls='-', alpha=0.6,
                   label=f'ATM={F:.1f}%')
        ax.set_title(
            f'{mat}  IQR={res["IQR_9010"]:.2f}%  Skew={res["Skew"]:.2f}')
        ax.set_xlabel('Ставка (%)')
        ax.set_ylabel('Плотность')
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save, f'rnd_{date.date()}.png')


def plot_rnd_fan(clean_df, maturity='3M', key_rate_df=None, save=True):
    """Fan chart: квантильные полосы RND во времени."""
    df = clean_df[clean_df['Maturity'] == maturity].sort_values('Date')
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(df['Date'], df['Q10'], df['Q90'],
                    alpha=0.20, color=COLORS['iqr'], label='Q10-Q90 (80%)')
    ax.fill_between(df['Date'], df['Q25'], df['Q75'],
                    alpha=0.35, color=COLORS['iqr'], label='Q25-Q75 (50%)')
    ax.plot(df['Date'], df['Q50'], color=COLORS['iqr'], lw=2, label='Q50')
    ax.plot(df['Date'], df['F'], color='navy', lw=1.5, ls=':',
            label='ATM-форвард из IV')
    if key_rate_df is not None:
        kr = key_rate_df[
            (key_rate_df['Date'] >= df['Date'].min()) &
            (key_rate_df['Date'] <= df['Date'].max())]
        ax.plot(kr['Date'], kr['Key Rate'],
                color=COLORS['key_rate'], lw=2.5, ls='--',
                label='КС (справочно)')
    ax.set_title(f'Fan Chart RND  |  {maturity}', fontweight='bold')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Ставка (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.legend(loc='upper left')
    plt.tight_layout()
    _save_or_show(fig, save, f'rnd_fan_{maturity}.png')


def plot_mpu_timeseries(clean_df, key_rate_df=None,
                        maturities=None, save=True):
    """MPU (IQR Q90-Q10) по срокам + КС справочно."""
    if maturities is None:
        maturities = ['1M', '3M', '6M', '1Y']
    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 1, figure=fig, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for mat in maturities:
        sub = clean_df[clean_df['Maturity'] == mat].sort_values('Date')
        if sub.empty:
            continue
        ax1.plot(sub['Date'], sub['IQR_9010'],
                 color=COLORS.get(mat, '#333'), lw=2, label=f'MPU {mat}')
    ax1.set_title('MPU = Q90 - Q10 по срокам (из поверхности IV)',
                  fontweight='bold')
    ax1.set_ylabel('IQR, п.п.')
    ax1.legend(ncol=len(maturities))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    if key_rate_df is not None:
        ref = clean_df[clean_df['Maturity'] == maturities[0]].sort_values('Date')
        kr  = key_rate_df[
            (key_rate_df['Date'] >= ref['Date'].min()) &
            (key_rate_df['Date'] <= ref['Date'].max())]
        ax2.plot(kr['Date'], kr['Key Rate'],
                 color=COLORS['key_rate'], lw=2, label='КС (справочно)')
        if 'Inflation' in kr.columns:
            ax2.plot(kr['Date'], kr['Inflation'],
                     color=COLORS['inflation'], lw=1.5, ls='--',
                     label='Инфляция г/г (справочно)')
        ax2.legend()
    ax2.set_title('КС и инфляция (справочно)')
    ax2.set_ylabel('%')
    ax2.set_xlabel('Дата')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    fig.suptitle('Индекс неопределённости ДКП | Обзор',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save, 'mpu_timeseries.png')


def plot_moments_timeseries(clean_df, maturity='3M', save=True):
    """Динамика моментов RND: Std, Skew, Kurt, Tail."""
    df = clean_df[clean_df['Maturity'] == maturity].sort_values('Date')
    if df.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f'Моменты RND  |  {maturity}',
                 fontsize=14, fontweight='bold')
    panels = [
        ('Std',        'Стандартное отклонение',              '#1565C0', None),
        ('Skew',       'Асимметрия  [>0 = риск роста ставки]', '#6A1B9A', 0.0),
        ('Kurt',       'Эксцесс  [>3 = толстые хвосты]',      '#E65100', 3.0),
        ('Tail_total', 'Хвостовой риск (Q99-Q90)+(Q10-Q01)',   '#B71C1C', None),
    ]
    for ax, (col, title, color, hline) in zip(axes, panels):
        ax.plot(df['Date'], df[col], color=color, lw=2)
        ax.fill_between(df['Date'], df[col], alpha=0.15, color=color)
        if hline is not None:
            ax.axhline(hline, color='grey', lw=1, ls='--',
                       label=f'= {hline}')
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(col)
    axes[-1].set_xlabel('Дата')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    _save_or_show(fig, save, f'moments_{maturity}.png')


def plot_term_structure(clean_df, dates_highlight=None, save=True):
    """Срочная структура MPU: средняя + отдельные даты."""
    order     = ['1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y']
    available = [m for m in order if m in clean_df['Maturity'].unique()]
    avg_iqr   = clean_df.groupby('Maturity')['IQR_9010'].mean().reindex(available)

    if dates_highlight is None:
        all_d = sorted(clean_df['Date'].unique())
        n     = len(all_d)
        dates_highlight = [all_d[0], all_d[n // 2], all_d[-1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Срочная структура MPU (из поверхности IV)',
                 fontsize=13, fontweight='bold')

    ax1.bar(available, avg_iqr.values,
            color=COLORS['iqr'], alpha=0.8, edgecolor='white', width=0.6)
    ax1.set_title('Средний MPU по срокам')
    ax1.set_xlabel('Срок')
    ax1.set_ylabel('IQR, п.п.')
    for i, (m, v) in enumerate(zip(available, avg_iqr.values)):
        if not np.isnan(v):
            ax1.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=8)

    cmap   = plt.cm.plasma
    colors = [cmap(i / max(len(dates_highlight) - 1, 1))
              for i in range(len(dates_highlight))]
    for date, color in zip(dates_highlight, colors):
        sub = (clean_df[clean_df['Date'] == date]
               .set_index('Maturity').reindex(available))
        ax2.plot(available, sub['IQR_9010'].values,
                 marker='o', color=color, lw=2,
                 label=pd.Timestamp(date).strftime('%Y-%m'))
    ax2.set_title('Срочная структура в отдельные даты')
    ax2.set_xlabel('Срок')
    ax2.set_ylabel('IQR, п.п.')
    ax2.legend(title='Дата', fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save, 'term_structure.png')


def plot_correlation_matrix(clean_df, maturity='3M', save=True):
    """Корреляционная матрица компонент MPU."""
    df   = clean_df[clean_df['Maturity'] == maturity].copy()
    cols = [c for c in ['IQR_9010', 'IQR_7525', 'Std', 'Skew',
                        'Kurt', 'Tail_right', 'Tail_left', 'Tail_total']
            if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Корреляция Пирсона')
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8,
                    color='white' if abs(v) > 0.7 else 'black')
    ax.set_title(f'Корреляция компонент MPU  |  {maturity}',
                 fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save, f'correlation_{maturity}.png')


def plot_rnd_comparison(dates_compare, iv_df, maturity='3M', save=True):
    """Наложение RND нескольких дат на один график."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap   = plt.cm.viridis
    colors = [cmap(i / max(len(dates_compare) - 1, 1))
              for i in range(len(dates_compare))]
    for date, color in zip(dates_compare, colors):
        out = _compute_rnd_arrays(date, iv_df, maturity)
        if out is None:
            continue
        K_grid, RND, F, res = out
        ax.plot(K_grid, RND, color=color, lw=2,
                label=(f"{pd.Timestamp(date).strftime('%Y-%m')}  "
                       f"IQR={res['IQR_9010']:.1f}%  ATM={F:.1f}%"))
        ax.axvline(F, color=color, lw=0.8, ls=':', alpha=0.5)
    ax.set_title(f'Сравнение RND  |  {maturity}', fontweight='bold')
    ax.set_xlabel('Ставка (%)')
    ax.set_ylabel('Плотность')
    ax.legend(fontsize=9, title='Дата | IQR | ATM')
    plt.tight_layout()
    _save_or_show(fig, save, f'rnd_comparison_{maturity}.png')


# ============================================================
# ВИЗУАЛИЗАЦИЯ 9: Агрегированный MPU — Decay vs PCA
# ============================================================

def plot_mpu_aggregated(mpu_df: pd.DataFrame,
                        meta: dict,
                        key_rate_df: pd.DataFrame | None = None,
                        save: bool = True) -> None:
    """
    Главный график агрегации: MPU_decay vs MPU_pca.

    Панель 1 (верх):  нормированные индексы (z-score) — сравнение формы.
    Панель 2 (лево):  сырые значения.
    Панель 3 (право): нагрузки PC1 и decay-веса по срокам.
    Панель 4 (низ):   ключевая ставка справочно.
    """
    fig = plt.figure(figsize=(16, 14))
    gs  = GridSpec(3, 2, figure=fig,
                   height_ratios=[2.5, 2.5, 1.5],
                   hspace=0.50, wspace=0.35)

    ax_norm = fig.add_subplot(gs[0, :])
    ax_raw  = fig.add_subplot(gs[1, 0])
    ax_load = fig.add_subplot(gs[1, 1])
    ax_kr   = fig.add_subplot(gs[2, :])

    rho = meta['corr']

    # -- Панель 1: нормированные MPU --------------------------
    ax_norm.plot(mpu_df['Date'], mpu_df['MPU_decay_norm'],
                 color=COLORS['decay'], lw=2.5, ls='-',
                 label='MPU_decay (взвешенное среднее)')
    ax_norm.plot(mpu_df['Date'], mpu_df['MPU_pca_norm'],
                 color=COLORS['pca'], lw=2.5, ls='--',
                 label='MPU_pca (первая главная компонента)')
    ax_norm.axhline(0, color='grey', lw=0.8, ls=':')
    ax_norm.fill_between(mpu_df['Date'],
                         mpu_df['MPU_decay_norm'],
                         mpu_df['MPU_pca_norm'],
                         alpha=0.08, color='grey',
                         label='Расхождение методов')
    ax_norm.set_ylabel('Z-score')
    ax_norm.set_title(
        f'Агрегированный MPU: Decay vs PCA  '
        f'(корреляция = {rho:.3f})\n'
        f'Сроки: {", ".join(meta["maturities"])}  |  '
        f'Метрика: {meta["metric"]}',
        fontsize=12, fontweight='bold'
    )
    ax_norm.legend(loc='upper left')
    ax_norm.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_norm.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_norm.xaxis.get_majorticklabels(), rotation=45)

    # -- Панель 2: сырые значения -----------------------------
    ax_raw.plot(mpu_df['Date'], mpu_df['MPU_decay'],
                color=COLORS['decay'], lw=2, label='MPU_decay (п.п.)')
    ax_raw.plot(mpu_df['Date'], mpu_df['MPU_pca'],
                color=COLORS['pca'], lw=2, ls='--',
                label='MPU_pca (у.е.)')
    ax_raw.set_title(f'Сырые значения MPU\nметрика: {meta["metric"]}')
    ax_raw.set_ylabel('Значение')
    ax_raw.legend(fontsize=9)
    ax_raw.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_raw.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax_raw.xaxis.get_majorticklabels(), rotation=45)

    # -- Панель 3: нагрузки PC1 vs decay-веса ----------------
    maturities    = meta['maturities']
    loadings_vals = [meta['loadings'][m]      for m in maturities]
    decay_vals    = [meta['decay_weights'][m] for m in maturities]
    explained     = list(meta['explained'].values())

    x     = np.arange(len(maturities))
    width = 0.35

    bars1 = ax_load.bar(x - width / 2, loadings_vals, width,
                        color=COLORS['pca'], alpha=0.8,
                        edgecolor='white', label='Нагрузка PC1')
    bars2 = ax_load.bar(x + width / 2, decay_vals, width,
                        color=COLORS['decay'], alpha=0.8,
                        edgecolor='white', label='Decay-вес')

    for bar, val in zip(list(bars1) + list(bars2),
                        loadings_vals + decay_vals):
        ax_load.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax_load.axhline(0, color='grey', lw=0.8)
    ax_load.set_xticks(x)
    ax_load.set_xticklabels(maturities)
    ax_load.set_xlabel('Срок экспирации')
    ax_load.set_ylabel('Вес / нагрузка')
    ax_load.legend(fontsize=9)

    ax_load2 = ax_load.twinx()
    ax_load2.plot(maturities, explained,
                  color='#c0392b', marker='D', lw=1.5,
                  ls=':', markersize=7, label='Объясн. дисперсия')
    ax_load2.set_ylabel('Доля дисперсии PCA', color='#c0392b')
    ax_load2.tick_params(axis='y', labelcolor='#c0392b')
    ax_load2.set_ylim(0, 1)
    ax_load.set_title(
        f'Веса по срокам: Decay vs PCA\n'
        f'PC1 объясняет {explained[0]:.1%} общей дисперсии'
    )

    # -- Панель 4: КС справочно --------------------------------
    if key_rate_df is not None:
        kr = key_rate_df[
            (key_rate_df['Date'] >= mpu_df['Date'].min()) &
            (key_rate_df['Date'] <= mpu_df['Date'].max())
            ].sort_values('Date')
        ax_kr.step(kr['Date'], kr['Key Rate'], where='post',
                   color=COLORS['key_rate'], lw=2,
                   label='Ключевая ставка (справочно)')
        if 'Inflation' in kr.columns:
            ax_kr.plot(kr['Date'], kr['Inflation'],
                       color=COLORS['inflation'], lw=1.5, ls='--',
                       label='Инфляция г/г (справочно)')
        ax_kr.legend(fontsize=9)
    ax_kr.set_title('Ключевая ставка (справочно, не участвует в расчёте)',
                    fontsize=10)
    ax_kr.set_ylabel('%')
    ax_kr.set_xlabel('Дата')
    ax_kr.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_kr.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_kr.xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle(
        'Агрегированный MPU: Decay weights vs PCA\n'
        'Оба метода агрегируют IQR по срокам в единый индекс',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    _save_or_show(fig, save, 'mpu_aggregated.png')


# ============================================================
# ВИЗУАЛИЗАЦИЯ 10: Scatter MPU_decay vs MPU_pca
# ============================================================

def plot_mpu_scatter(mpu_df: pd.DataFrame,
                     meta: dict,
                     save: bool = True) -> None:
    """
    Scatter plot MPU_decay vs MPU_pca (нормированные).

    Чем ближе точки к диагонали y=x — тем выше согласованность методов.
    Цвет = хронология (светлый = ранние даты, тёмный = поздние).
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    x      = mpu_df['MPU_decay_norm'].values
    y      = mpu_df['MPU_pca_norm'].values
    t_vals = np.linspace(0, 1, len(mpu_df))
    colors = plt.cm.viridis(t_vals)

    ax.scatter(x, y, c=colors, s=60, zorder=5, alpha=0.85)

    lim = max(abs(x).max(), abs(y).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color='grey',
            lw=1.5, ls='--', label='y = x (идеальное совпадение)')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Аннотируем выбросы
    for xi, yi, d in zip(x, y, mpu_df['Date']):
        if abs(xi) > 1.5 or abs(yi) > 1.5:
            ax.annotate(pd.Timestamp(d).strftime('%Y-%m'),
                        (xi, yi), fontsize=7,
                        xytext=(5, 5), textcoords='offset points')

    sm = plt.cm.ScalarMappable(
        cmap='viridis',
        norm=plt.Normalize(
            vmin=mpu_df['Date'].min().year,
            vmax=mpu_df['Date'].max().year
        )
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Год')

    rho = meta['corr']
    ax.set_xlabel('MPU_decay (z-score)')
    ax.set_ylabel('MPU_pca (z-score)')
    ax.set_title(
        f'Scatter: MPU_decay vs MPU_pca\n'
        f'Корреляция = {rho:.3f}  |  '
        f'Сроки: {", ".join(meta["maturities"])}',
        fontweight='bold'
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_or_show(fig, save, 'mpu_scatter.png')


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

if __name__ == '__main__':

    # 1. Загрузка
    iv_df, key_rate_df = load_data()

    # 2. RND по всем датам и срокам
    results_df = run_pipeline(
        iv_df,
        maturities=['1M', '3M', '6M', '1Y'],
        verbose=True
    )

    # 3. Контроль качества
    results_df = check_rnd_quality(results_df)
    clean_df   = results_df[results_df['quality_ok']].copy()
    print(f"\nПосле фильтрации: {len(clean_df)} наблюдений")

    # 4. Описательная статистика
    print_summary(clean_df)

    # 5. Агрегация MPU по срокам: Decay vs PCA
    mpu_df, meta = aggregate_mpu_across_maturities(
        clean_df,
        maturities=['1M', '3M', '6M', '1Y'],
        metric='IQR_9010'
    )

    # 6. Сохранение результатов
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    clean_df.to_csv(out_dir / 'rnd_statistics.csv', index=False)
    mpu_df.to_csv(out_dir / 'mpu_aggregated.csv', index=False)
    print(f"\nСохранено:")
    print(f"  {out_dir / 'rnd_statistics.csv'}")
    print(f"  {out_dir / 'mpu_aggregated.csv'}")

    # 7. Визуализация
    print("\nГенерация графиков...")
    all_dates  = sorted(iv_df['Date'].unique())
    date_first = all_dates[0]
    date_mid   = all_dates[len(all_dates) // 2]
    date_last  = all_dates[-1]

    print("\n[1/10] Улыбка IV...")
    plot_iv_smile(date_last, iv_df, key_rate_df=key_rate_df,
                  maturities=['1M', '3M', '6M', '1Y'])

    print("[2/10] RND на последнюю дату...")
    plot_rnd(date_last, iv_df, key_rate_df=key_rate_df,
             maturities=['1M', '3M', '6M', '1Y'])

    print("[3/10] Fan chart (3M)...")
    plot_rnd_fan(clean_df, maturity='3M', key_rate_df=key_rate_df)

    print("[4/10] MPU по срокам + КС...")
    plot_mpu_timeseries(clean_df, key_rate_df=key_rate_df,
                        maturities=['1M', '3M', '6M', '1Y'])

    print("[5/10] Моменты RND (3M)...")
    plot_moments_timeseries(clean_df, maturity='3M')

    print("[6/10] Срочная структура...")
    plot_term_structure(clean_df,
                        dates_highlight=[date_first, date_mid, date_last])

    print("[7/10] Корреляция компонент (3M)...")
    plot_correlation_matrix(clean_df, maturity='3M')

    print("[8/10] Сравнение RND (3 даты)...")
    plot_rnd_comparison([date_first, date_mid, date_last],
                        iv_df, maturity='3M')

    print("[9/10] Агрегированный MPU: Decay vs PCA...")
    plot_mpu_aggregated(mpu_df, meta, key_rate_df=key_rate_df)

    print("[10/10] Scatter MPU_decay vs MPU_pca...")
    plot_mpu_scatter(mpu_df, meta)

    print(f"\nГотово. Графики: {PLOT_DIR.resolve()}")