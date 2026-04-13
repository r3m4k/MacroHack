# 03_mpu_models.py
# Построение трёх финальных MPU-индексов:
#   MPU_rv_std — реализованная волатильность ключевой ставки
#   MPU_decay  — взвешенное среднее IQR по срокам (decay-веса)
#   MPU_atm    — ATM вменённая волатильность (PCA по срокам)
#
# Читает rnd_stats.csv из rnd_analysis/.
# Результаты -> Task3/final_models/mpu_models/
#
# Запуск: python 03_mpu_models.py

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))
RND_DIR  = BASE_DIR / 'rnd_analysis'
OUT_DIR  = BASE_DIR / 'mpu_models'
PLOT_DIR = OUT_DIR / 'plots'

from data_loading.case_2 import get_case_2_IV, get_key_rate_dataframe

MATURITIES = ['1M', '3M', '6M', '1Y']


# ════════════════════════════════════════════════════════════
# MPU_rv_std
# ════════════════════════════════════════════════════════════

def build_mpu_rv_std(key_rate_df, window=3):
    """
    RV_t = std(Δr_{t-w+1}, ..., Δr_t)
    Нормировка: z-score.
    """
    df = (key_rate_df[['Date','Key Rate']]
          .dropna(subset=['Date'])
          .rename(columns={'Key Rate':'Key_Rate'})
          .sort_values('Date').copy())
    df['Delta']      = df['Key_Rate'].diff()
    df['RV_std']     = df['Delta'].rolling(window, min_periods=window).std()
    df['MPU_rv_std'] = (df['RV_std'] - df['RV_std'].mean()) / df['RV_std'].std()
    return df[['Date','Key_Rate','Delta','RV_std','MPU_rv_std']].dropna()


# ════════════════════════════════════════════════════════════
# MPU_decay
# ════════════════════════════════════════════════════════════

def build_mpu_decay(rnd_clean):
    """
    Decay-взвешенное среднее IQR_9010 по срокам.
    w_i = exp(-ln2 * i) / sum,  i = 0..3 для ['1M','3M','6M','1Y']
    Нормировка: z-score.
    """
    pivot = (rnd_clean[rnd_clean['Maturity'].isin(MATURITIES)]
             .pivot_table('IQR_9010', 'Date', 'Maturity')
             .reindex(columns=MATURITIES)
             .dropna())

    lam = np.log(2)
    w   = np.exp(-lam * np.arange(len(MATURITIES)))
    w  /= w.sum()

    mpu = pd.Series(pivot.values @ w, index=pivot.index, name='MPU_decay')
    mpu_norm = (mpu - mpu.mean()) / mpu.std()
    mpu_norm.name = 'MPU_decay_norm'

    result = pd.DataFrame({
        'Date':          pivot.index,
        'MPU_decay':     mpu.values,
        'MPU_decay_norm': mpu_norm.values,
    }).reset_index(drop=True)

    print(f"  MPU_decay: {len(result)} дат  |  "
          f"веса: " + "  ".join(f"{m}={w_:.3f}"
                                for m, w_ in zip(MATURITIES, w)))
    return result


# ════════════════════════════════════════════════════════════
# MPU_atm
# ════════════════════════════════════════════════════════════

def get_atm_iv(date, iv_df, maturity):
    """ATM IV = IV при страйке с min волатильностью."""
    mask = (iv_df['Date']==date) & (iv_df['Maturity']==maturity)
    sl   = iv_df[mask].sort_values('Strike')
    if sl.empty: return np.nan
    K, iv = sl['Strike'].values, sl['Volatility'].values
    F     = float(K[np.argmin(iv)])
    cs    = CubicSpline(K, iv, extrapolate=False)
    val   = float(cs(F))
    return float(np.interp(F, K, iv)) if np.isnan(val) else val


def build_mpu_atm(iv_df):
    """
    ATM IV для каждой даты и срока → PCA по срокам → MPU_atm.
    Нормировка: z-score.
    """
    dates = sorted(iv_df['Date'].unique())
    rows  = []
    for d in dates:
        row = {'Date': d}
        for mat in MATURITIES:
            row[mat] = get_atm_iv(d, iv_df, mat)
        rows.append(row)

    atm_df = pd.DataFrame(rows).dropna()

    X    = atm_df[MATURITIES].values
    sc   = StandardScaler()
    Xs   = sc.fit_transform(X)
    pca  = PCA(n_components=1)
    pc1  = pca.fit_transform(Xs).ravel()
    ev   = pca.explained_variance_ratio_[0]

    # Ориентация: коррелируем с простым средним ATM
    mean_atm = X.mean(axis=1)
    if np.corrcoef(pc1, mean_atm)[0,1] < 0:
        pc1 = -pc1

    def zscore(a): return (a - a.mean()) / a.std()

    result = pd.DataFrame({
        'Date':         atm_df['Date'].values,
        'MPU_atm':      pc1,
        'MPU_atm_norm': zscore(pc1),
        **{f'ATM_{m}': atm_df[m].values for m in MATURITIES},
    }).reset_index(drop=True)

    print(f"  MPU_atm: {len(result)} дат  |  "
          f"PC1 объясняет {ev:.1%}")
    return result


# ════════════════════════════════════════════════════════════
# Графики
# ════════════════════════════════════════════════════════════

def _save(fig, name):
    fig.savefig(PLOT_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mpu_overview(rv_df, decay_df, atm_df, key_rate_df):
    """Три MPU-индекса + ключевая ставка на одном графике."""
    kr = key_rate_df.dropna(subset=['Date']).sort_values('Date')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1.5]})
    fig.suptitle('Три MPU-индекса', fontweight='bold', fontsize=13)

    rv  = rv_df.set_index('Date')['MPU_rv_std']
    dec = decay_df.set_index('Date')['MPU_decay_norm']
    atm = atm_df.set_index('Date')['MPU_atm_norm']

    for s, name, color, ls in [
        (rv,  'MPU_rv_std', '#8E44AD', '-'),
        (dec, 'MPU_decay',  '#E65100', '--'),
        (atm, 'MPU_atm',    '#27AE60', '-.'),
    ]:
        ax1.plot(s.index, s.values, color=color, lw=2, ls=ls, label=name)

    ax1.axhline(0, color='grey', lw=0.8, ls=':')
    ax1.set_ylabel('Z-score'); ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.step(kr['Date'], kr['Key Rate'], where='post',
             color='#C62828', lw=2, label='Ключевая ставка')
    ax2.set_ylabel('Ставка (%)'); ax2.set_xlabel('Дата')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout(); _save(fig, 'mpu_overview.png')


def plot_mpu_correlation(rv_df, decay_df, atm_df):
    """Scatter-матрица трёх нормированных MPU."""
    rv  = rv_df.set_index('Date')['MPU_rv_std']
    dec = decay_df.set_index('Date')['MPU_decay_norm']
    atm = atm_df.set_index('Date')['MPU_atm_norm']
    merged = pd.concat([rv, dec, atm], axis=1).dropna()
    merged.columns = ['MPU_rv_std', 'MPU_decay', 'MPU_atm']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Попарные корреляции MPU-индексов', fontweight='bold')
    pairs = [('MPU_rv_std','MPU_decay','#E65100'),
             ('MPU_rv_std','MPU_atm',  '#27AE60'),
             ('MPU_decay', 'MPU_atm',  '#1565C0')]
    for ax, (x, y, c) in zip(axes, pairs):
        ax.scatter(merged[x], merged[y], alpha=0.6, color=c, s=35)
        z  = np.polyfit(merged[x], merged[y], 1)
        xf = np.linspace(merged[x].min(), merged[x].max(), 100)
        ax.plot(xf, np.polyval(z, xf), color='black', lw=2)
        rho = merged[[x,y]].corr().iloc[0,1]
        ax.set_title(f'{x} vs {y}\nr={rho:.3f}', fontweight='bold')
        ax.set_xlabel(x); ax.set_ylabel(y)
    plt.tight_layout(); _save(fig, 'mpu_correlation.png')


def plot_mpu_recent(rv_df, decay_df, atm_df, key_rate_df, n_months=24):
    """Последние n_months — детальный вид."""
    kr  = key_rate_df.dropna(subset=['Date']).sort_values('Date')
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=n_months)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1.5]})
    fig.suptitle(f'MPU-индексы: последние {n_months} месяцев',
                 fontweight='bold', fontsize=13)

    for df_s, col, name, color, ls in [
        (rv_df,    'MPU_rv_std',   'MPU_rv_std', '#8E44AD', '-'),
        (decay_df, 'MPU_decay_norm','MPU_decay', '#E65100', '--'),
        (atm_df,   'MPU_atm_norm', 'MPU_atm',   '#27AE60', '-.'),
    ]:
        s = df_s.set_index('Date')[col]
        s = s[s.index >= cutoff]
        ax1.plot(s.index, s.values, color=color, lw=2.5, ls=ls, label=name)

    ax1.axhline(0, color='grey', lw=0.8, ls=':')
    ax1.set_ylabel('Z-score'); ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

    kr_r = kr[kr['Date'] >= cutoff]
    ax2.step(kr_r['Date'], kr_r['Key Rate'], where='post',
             color='#C62828', lw=2.5)
    ax2.set_ylabel('КС (%)'); ax2.set_xlabel('Дата'); ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout(); _save(fig, 'mpu_recent.png')


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 55)
    print("03 | ПОСТРОЕНИЕ MPU-ИНДЕКСОВ")
    print("=" * 55)

    rnd_path = RND_DIR / 'rnd_stats.csv'
    if not rnd_path.exists():
        raise FileNotFoundError(
            f"{rnd_path} не найден. Сначала запустите 01_rnd_analysis.py"
        )

    iv_df       = get_case_2_IV()
    key_rate_df = get_key_rate_dataframe()
    rnd_clean   = pd.read_csv(rnd_path, parse_dates=['Date'])

    print("\nMPU_rv_std:")
    rv_df = build_mpu_rv_std(key_rate_df)
    print(f"  N={len(rv_df)}  "
          f"mean_RV={rv_df['RV_std'].mean():.3f}  "
          f"max_RV={rv_df['RV_std'].max():.3f}")

    print("\nMPU_decay:")
    decay_df = build_mpu_decay(rnd_clean)

    print("\nMPU_atm:")
    atm_df = build_mpu_atm(iv_df)

    # Сохранение
    rv_df.to_csv(OUT_DIR    / 'MPU_rv_std.csv', index=False)
    decay_df.to_csv(OUT_DIR / 'MPU_decay.csv',  index=False)
    atm_df.to_csv(OUT_DIR   / 'MPU_atm.csv',    index=False)
    print(f"\nСохранено в {OUT_DIR}:")
    print(f"  MPU_rv_std.csv  ({len(rv_df)} строк)")
    print(f"  MPU_decay.csv   ({len(decay_df)} строк)")
    print(f"  MPU_atm.csv     ({len(atm_df)} строк)")

    # Последние значения
    print("\nПоследние значения индексов:")
    for df_s, col in [(rv_df,'MPU_rv_std'),
                      (decay_df,'MPU_decay_norm'),
                      (atm_df,'MPU_atm_norm')]:
        last = df_s.sort_values('Date').iloc[-1]
        print(f"  {col:<18}  {last['Date'].date() if hasattr(last['Date'],'date') else last['Date']}  "
              f"z={last[col]:+.3f}")

    print("\nГенерация графиков...")
    plot_mpu_overview(rv_df, decay_df, atm_df, key_rate_df)
    plot_mpu_correlation(rv_df, decay_df, atm_df)
    plot_mpu_recent(rv_df, decay_df, atm_df, key_rate_df)

    print(f"\nГотово → {OUT_DIR}")


if __name__ == '__main__':
    main()