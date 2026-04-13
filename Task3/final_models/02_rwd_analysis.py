# 02_rwd_analysis.py
# Перевод RND в Real World Distribution (RWD) через Esscher transform.
#
# q_rw(x) = q_rn(x) * exp(lambda*x) / Z,   lambda = -0.1
# lambda < 0: RWD сдвинута влево (реальное среднее < риск-нейтрального).
#
# Результаты -> final_models/rwd_analysis/
# Запуск: python 02_rwd_analysis.py

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

BASE_DIR = Path(__file__).parent
OUT_DIR  = BASE_DIR / 'rwd_analysis'
PLOT_DIR = OUT_DIR / 'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR.parent))

MATURITIES = ['1M', '3M', '6M', '1Y']
LAMBDA     = -0.1

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10,
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def _rnd(date, iv_df, maturity, n=1000):
    sl = iv_df[(iv_df['Date'] == date) & (iv_df['Maturity'] == maturity)
               ].sort_values('Strike')
    if len(sl) < 5:
        return None, None, None
    K, iv = sl['Strike'].values, sl['Volatility'].values
    T     = sl['Maturity (year fraction)'].iloc[0]
    F     = float(K[np.argmin(iv)])
    cs    = CubicSpline(K, iv, extrapolate=False)
    Kg    = np.linspace(K.min(), K.max(), n)
    dK    = Kg[1] - Kg[0]
    ivg   = cs(Kg)
    ivg   = np.where(np.isnan(ivg), np.where(Kg < K[0], iv[0], iv[-1]), ivg)
    ivg   = np.maximum(ivg, 0.01)

    def bc(F, k, T, s):
        if T <= 1e-10: return max(F - k, 0.0)
        d = s * np.sqrt(T)
        if d < 1e-10: return max(F - k, 0.0)
        x = (F - k) / d
        return d * (x * norm.cdf(x) + norm.pdf(x))

    C   = np.array([bc(F, k, T, s) for k, s in zip(Kg, ivg)])
    rnd = np.gradient(np.gradient(C, dK), dK)
    rnd = np.maximum(rnd, 0.0)
    Z   = np.trapz(rnd, Kg)
    if Z < 1e-10:
        return None, None, None
    return Kg, rnd / Z, F


def _stats(Kg, p):
    dK  = Kg[1] - Kg[0]
    mu  = np.trapz(Kg * p, Kg)
    var = np.trapz((Kg - mu) ** 2 * p, Kg)
    sig = np.sqrt(max(var, 1e-10))
    cdf = np.clip(np.cumsum(p) * dK, 0, 1); cdf /= cdf[-1]
    def q(prob):
        return float(Kg[np.clip(np.searchsorted(cdf, prob), 0, len(Kg)-1)])
    return dict(Mean=mu, Std=sig, Var=float(var),
                Q10=q(.10), Q90=q(.90), IQR=q(.90)-q(.10))


def run_pipeline(iv_df, maturities=MATURITIES, lam=LAMBDA):
    dates, rows = sorted(iv_df['Date'].unique()), []
    for mat in maturities:
        print(f'  {mat}...', end=' ')
        ok = skip = 0
        for d in dates:
            Kg, rnd, F = _rnd(d, iv_df, mat)
            if Kg is None:
                skip += 1; continue
            w   = np.exp(lam * Kg)
            rwd = rnd * w
            Z   = np.trapz(rwd, Kg)
            rwd = rwd / Z if Z > 1e-10 else rnd.copy()
            sr, sw = _stats(Kg, rnd), _stats(Kg, rwd)
            rows.append(dict(
                Date=d, Maturity=mat, F=F, lambda_=lam,
                Mean_rnd=sr['Mean'], Std_rnd=sr['Std'],
                IQR_rnd=sr['IQR'], Q10_rnd=sr['Q10'], Q90_rnd=sr['Q90'],
                Mean_rwd=sw['Mean'], Std_rwd=sw['Std'],
                IQR_rwd=sw['IQR'], Q10_rwd=sw['Q10'], Q90_rwd=sw['Q90'],
                VRP=sr['Var']-sw['Var'],
                Mean_shift=sr['Mean']-sw['Mean'],
                IQR_diff=sr['IQR']-sw['IQR'],
            ))
            ok += 1
        print(f'OK={ok} skip={skip}')
    return pd.DataFrame(rows)


def _save(name):
    plt.savefig(PLOT_DIR / name, bbox_inches='tight')
    plt.close()
    print(f'  saved: {name}')


def plot_comparison(df, mat='3M'):
    sub = df[df['Maturity'] == mat].sort_values('Date')
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f'RND vs RWD | {mat}  (lambda={LAMBDA})', fontweight='bold')

    ax = axes[0]
    ax.plot(sub['Date'], sub['Mean_rnd'], color='#1565C0', lw=2, label='Mean RND')
    ax.plot(sub['Date'], sub['Mean_rwd'], color='#E65100', lw=2, ls='--', label='Mean RWD')
    ax.fill_between(sub['Date'], sub['Mean_rnd'], sub['Mean_rwd'],
                    alpha=0.15, color='grey', label='Премия за риск')
    ax.set_ylabel('Среднее (%)'); ax.legend(fontsize=9)
    ax.set_title('Среднее: RND vs RWD')

    ax = axes[1]
    ax.plot(sub['Date'], sub['IQR_rnd'], color='#1565C0', lw=2, label='IQR RND')
    ax.plot(sub['Date'], sub['IQR_rwd'], color='#E65100', lw=2, ls='--', label='IQR RWD')
    ax.plot(sub['Date'], sub['VRP'],     color='#27AE60', lw=1.5, ls=':', label='VRP')
    ax.set_ylabel('п.п.'); ax.legend(fontsize=9)
    ax.set_title('IQR и VRP = Var(RND) - Var(RWD)')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout(); _save(f'rnd_vs_rwd_{mat}.png')


def plot_premium_structure(df):
    avail = [m for m in MATURITIES if m in df['Maturity'].unique()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Срочная структура премий за риск', fontweight='bold')
    for ax, col, title, color in [
        (a1, 'Mean_shift', 'Сдвиг Mean_rnd - Mean_rwd (п.п.)', '#E65100'),
        (a2, 'VRP',        'VRP = Var(RND) - Var(RWD)',         '#27AE60'),
    ]:
        vals = df.groupby('Maturity')[col].mean().reindex(avail).values
        ax.bar(avail, vals, color=color, alpha=0.8, width=0.5)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.001 * np.sign(v),
                    f'{v:.4f}', ha='center', fontsize=9)
    plt.tight_layout(); _save('premium_term_structure.png')


if __name__ == '__main__':
    from data_loading.case_2 import get_case_2_IV

    print('=' * 50)
    print(f'02  АНАЛИЗ RWD  (lambda={LAMBDA})')
    print('=' * 50)

    df = run_pipeline(get_case_2_IV(), MATURITIES, lam=LAMBDA)
    df.to_csv(OUT_DIR / 'rwd_stats.csv', index=False)
    print(f'  Сохранено: rwd_stats.csv  ({len(df)} строк)')

    compare_cols = ['IQR_rnd', 'IQR_rwd', 'VRP', 'Mean_shift', 'IQR_diff']
    df.groupby('Maturity')[compare_cols].describe().round(4).to_csv(
        OUT_DIR / 'rwd_comparison.csv')
    print('  Сохранено: rwd_comparison.csv')

    print('\nВизуализация:')
    for mat in ['3M', '6M']:
        plot_comparison(df, mat=mat)
    plot_premium_structure(df)

    print(f'\nГотово -> {OUT_DIR.resolve()}')
