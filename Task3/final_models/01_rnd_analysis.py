# 01_rnd_analysis.py
# Анализ RND: извлечение характеристик risk-neutral distribution
# из поверхности вменённой волатильности (IV).
#
# Алгоритм:
#   1. ATM-форвард = страйк с min(IV) по улыбке  [без внешних данных]
#   2. Кубический сплайн по улыбке IV
#   3. Цены коллов по модели Башелье
#   4. d²C/dK² → RND  (Breeden-Litzenberger 1978)
#   5. Моменты: Mean, Std, Skew, Kurt
#   6. Квантили: Q01, Q10, Q25, Q50, Q75, Q90, Q99
#   7. Меры неопределённости: IQR_9010, IQR_7525, Tail_total
#
# Результаты → final_models/rnd_analysis/
# Запуск:  python 01_rnd_analysis.py

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

BASE_DIR  = Path(__file__).parent
OUT_DIR   = BASE_DIR / 'rnd_analysis'
PLOT_DIR  = OUT_DIR  / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR.parent))

MATURITIES = ['1M', '3M', '6M', '1Y']

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10,
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ------------------------------------------------------------------
# Базовые функции
# ------------------------------------------------------------------

def atm_forward(strikes, ivols):
    return float(strikes[np.argmin(ivols)])


def bachelier_call(F, K, T, sigma):
    if T <= 1e-10:
        return max(F - K, 0.0)
    d = sigma * np.sqrt(T)
    if d < 1e-10:
        return max(F - K, 0.0)
    x = (F - K) / d
    return d * (x * norm.cdf(x) + norm.pdf(x))


def extract_rnd(date, iv_df, maturity, n=1000):
    sl = iv_df[(iv_df['Date'] == date) & (iv_df['Maturity'] == maturity)
               ].sort_values('Strike')
    if len(sl) < 5:
        return None

    K  = sl['Strike'].values
    iv = sl['Volatility'].values
    T  = sl['Maturity (year fraction)'].iloc[0]
    F  = atm_forward(K, iv)

    cs  = CubicSpline(K, iv, extrapolate=False)
    Kg  = np.linspace(K.min(), K.max(), n)
    dK  = Kg[1] - Kg[0]
    ivg = cs(Kg)
    ivg = np.where(np.isnan(ivg),
                   np.where(Kg < K[0], iv[0], iv[-1]), ivg)
    ivg = np.maximum(ivg, 0.01)

    C   = np.array([bachelier_call(F, k, T, s) for k, s in zip(Kg, ivg)])
    rnd = np.gradient(np.gradient(C, dK), dK)
    rnd = np.maximum(rnd, 0.0)
    Z   = np.trapz(rnd, Kg)
    if Z < 1e-10:
        return None
    rnd /= Z

    mu   = np.trapz(Kg * rnd, Kg)
    var  = np.trapz((Kg - mu) ** 2 * rnd, Kg)
    sig  = np.sqrt(max(var, 1e-10))
    skew = np.trapz((Kg - mu) ** 3 * rnd, Kg) / sig ** 3
    kurt = np.trapz((Kg - mu) ** 4 * rnd, Kg) / sig ** 4

    cdf  = np.clip(np.cumsum(rnd) * dK, 0, 1)
    cdf /= cdf[-1]

    def q(p):
        return float(Kg[np.clip(np.searchsorted(cdf, p), 0, n - 1)])

    q01, q10, q25 = q(.01), q(.10), q(.25)
    q50, q75, q90 = q(.50), q(.75), q(.90)
    q99           = q(.99)

    return dict(
        Date=date, Maturity=maturity, F=F,
        Mean=mu, Std=sig, Skew=skew, Kurt=kurt,
        Q01=q01, Q10=q10, Q25=q25, Q50=q50,
        Q75=q75, Q90=q90, Q99=q99,
        IQR_9010=q90 - q10,
        IQR_7525=q75 - q25,
        Tail_right=q99 - q90,
        Tail_left=q10 - q01,
        Tail_total=(q99 - q90) + (q10 - q01),
    )


def run_pipeline(iv_df, maturities=None):
    if maturities is None:
        maturities = MATURITIES
    dates, rows = sorted(iv_df['Date'].unique()), []
    for mat in maturities:
        print(f'  {mat} ({len(dates)} дат)...', end=' ')
        ok = skip = 0
        for d in dates:
            r = extract_rnd(d, iv_df, mat)
            if r:
                rows.append(r); ok += 1
            else:
                skip += 1
        print(f'OK={ok} skip={skip}')
    return pd.DataFrame(rows)


def quality_filter(df):
    ok = (
            ((df['Mean'] - df['F']).abs() < 2.0)
            & (df['Std'].between(0.1, 15.0))
            & (df['IQR_9010'] > 0)
            & (df['Q10'] < df['Q25'])
            & (df['Q25'] < df['Q50'])
            & (df['Q50'] < df['Q75'])
            & (df['Q75'] < df['Q90'])
    )
    n_bad = (~ok).sum()
    print(f'  Качество: {ok.sum()}/{len(df)} OK'
          + (f'  ({n_bad} проблемных)' if n_bad else ''))
    return df[ok].copy()


# ------------------------------------------------------------------
# Графики
# ------------------------------------------------------------------

def _save(name):
    plt.savefig(PLOT_DIR / name, bbox_inches='tight')
    plt.close()
    print(f'  saved: {name}')


def plot_fan(df, key_rate_df, mat='3M'):
    sub = df[df['Maturity'] == mat].sort_values('Date')
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(sub['Date'], sub['Q10'], sub['Q90'],
                    alpha=0.20, color='#1565C0', label='Q10-Q90 (80%)')
    ax.fill_between(sub['Date'], sub['Q25'], sub['Q75'],
                    alpha=0.35, color='#1565C0', label='Q25-Q75 (50%)')
    ax.plot(sub['Date'], sub['Q50'], color='#1565C0', lw=2, label='Q50')
    ax.plot(sub['Date'], sub['F'], color='navy', lw=1.5, ls=':', label='ATM F')
    kr = key_rate_df.dropna(subset=['Date']).sort_values('Date')
    ax.step(kr['Date'], kr['Key Rate'], where='post',
            color='black', lw=2.5, label='Ключевая ставка (факт)')
    ax.set_title(f'Fan Chart RND | {mat}', fontweight='bold')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Ставка (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(f'fan_{mat}.png')


def plot_moments(df, mat='3M'):
    sub = df[df['Maturity'] == mat].sort_values('Date')
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle(f'Моменты RND | {mat}', fontweight='bold')
    panels = [
        ('IQR_9010', 'IQR Q90-Q10  (уровень неопределённости)', '#E65100', None),
        ('Std',      'Стандартное отклонение',                   '#1565C0', None),
        ('Skew',     'Асимметрия  [>0 = риск роста ставки]',     '#6A1B9A', 0.0),
        ('Kurt',     'Эксцесс  [>3 = толстые хвосты]',           '#27AE60', 3.0),
    ]
    for ax, (col, title, color, hline) in zip(axes, panels):
        ax.plot(sub['Date'], sub[col], color=color, lw=2)
        ax.fill_between(sub['Date'], sub[col], alpha=0.12, color=color)
        if hline is not None:
            ax.axhline(hline, color='grey', lw=1, ls='--')
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(col)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    _save(f'moments_{mat}.png')


def plot_term_structure(df):
    avail = [m for m in MATURITIES if m in df['Maturity'].unique()]
    avg   = df.groupby('Maturity')['IQR_9010'].mean().reindex(avail)
    dates = sorted(df['Date'].unique())
    n     = len(dates)
    kd    = [dates[0], dates[n // 3], dates[2 * n // 3], dates[-1]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Срочная структура IQR Q90-Q10', fontweight='bold')

    a1.bar(avail, avg.values, color='#1565C0', alpha=0.8, width=0.5)
    a1.set_title('Средний IQR по срокам')
    a1.set_ylabel('IQR (п.п.)')
    for i, v in enumerate(avg.values):
        a1.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)

    cmap = plt.cm.plasma
    for i, d in enumerate(kd):
        row = (df[df['Date'] == d].set_index('Maturity').reindex(avail))
        a2.plot(avail, row['IQR_9010'].values, marker='o',
                color=cmap(i / max(len(kd) - 1, 1)), lw=2,
                label=pd.Timestamp(d).strftime('%Y-%m'))
    a2.set_title('Срочная структура в отдельные даты')
    a2.set_ylabel('IQR (п.п.)')
    a2.legend(fontsize=8)
    plt.tight_layout()
    _save('term_structure.png')


# ------------------------------------------------------------------
# Точка входа
# ------------------------------------------------------------------

if __name__ == '__main__':
    from data_loading.case_2 import get_case_2_IV, get_key_rate_dataframe

    print('=' * 50)
    print('01  АНАЛИЗ RND')
    print('=' * 50)

    iv_df       = get_case_2_IV()
    key_rate_df = get_key_rate_dataframe()

    print('\nИзвлечение RND:')
    raw_df   = run_pipeline(iv_df, MATURITIES)
    clean_df = quality_filter(raw_df)
    print(f'  Итого: {len(clean_df)} наблюдений')

    clean_df.to_csv(OUT_DIR / 'rnd_stats.csv', index=False)
    print('  Сохранено: rnd_stats.csv')

    stat_cols = ['IQR_9010', 'IQR_7525', 'Std', 'Skew', 'Kurt',
                 'Tail_right', 'Tail_left', 'Tail_total']
    desc = clean_df.groupby('Maturity')[stat_cols].describe().round(4)
    desc.to_csv(OUT_DIR / 'rnd_descriptive.csv')
    print('  Сохранено: rnd_descriptive.csv')

    print('\nВизуализация:')
    for mat in ['3M', '6M']:
        plot_fan(clean_df, key_rate_df, mat=mat)
        plot_moments(clean_df, mat=mat)
    plot_term_structure(clean_df)

    print(f'\nГотово -> {OUT_DIR.resolve()}')