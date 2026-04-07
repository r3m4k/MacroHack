import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from data_loading.problem_1 import get_curve_train_dataframe
from data_loading.problem_1 import get_IV_train_dataframe
from data_loading.problem_1 import get_curve_predict_dataframe


def visualize_iv_surface(df, plot_type='time_series', save=True, **kwargs):
    """
    Визуализация поверхности вмененной волатильности.

    Параметры:
    df : pd.DataFrame
        DataFrame с колонками: Date, Maturity, Maturity (year fraction), Strike, Volatility.
        Date должен быть в формате datetime.
    plot_type : str
        Тип графика: 'time_series', 'smile', 'heatmap', 'surface'.
    save : bool
        Сохранять ли график в папку ./results.
    **kwargs : дополнительные параметры для конкретного типа графика.

    Возвращает:
    fig, ax : объекты matplotlib Figure и Axes.
    """
    # Создаём папку для результатов, если нужно сохранить
    if save:
        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)

    # Проверка наличия необходимых колонок
    required_cols = ['Date', 'Maturity', 'Maturity (year fraction)', 'Strike', 'Volatility']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"В DataFrame отсутствует колонка {col}")

    # Убедимся, что Date в datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    if plot_type == 'time_series':
        fig, ax, filename = _plot_time_series(df, **kwargs)
    elif plot_type == 'smile':
        fig, ax, filename = _plot_smile(df, **kwargs)
    elif plot_type == 'heatmap':
        fig, ax, filename = _plot_heatmap(df, **kwargs)
    elif plot_type == 'surface':
        fig, ax, filename = _plot_3d_surface(df, **kwargs)
    else:
        raise ValueError(f"Неизвестный тип графика: {plot_type}. Доступны: 'time_series', 'smile', 'heatmap', 'surface'.")

    if save:
        filepath = results_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {filepath}")
    plt.show()
    return fig, ax


def _plot_time_series(df, maturities=None, strikes=None, agg='mean', figsize=(12,6)):
    """Временные ряды волатильности."""
    fig, ax = plt.subplots(figsize=figsize)

    if strikes is None:
        grouped = df.groupby(['Date', 'Maturity'])['Volatility'].agg(agg).reset_index()
        maturities_to_plot = maturities if maturities else grouped['Maturity'].unique()
        for mat in maturities_to_plot:
            subset = grouped[grouped['Maturity'] == mat]
            ax.plot(subset['Date'], subset['Volatility'], label=mat)
    else:
        filtered = df[df['Strike'].isin(strikes)]
        if maturities:
            filtered = filtered[filtered['Maturity'].isin(maturities)]
        pivot = filtered.pivot_table(index='Date', columns=['Maturity', 'Strike'], values='Volatility')
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], label=f"{col[0]} strike={col[1]}")
            if len(pivot.columns) > 10:
                break

    ax.set_xlabel('Date')
    ax.set_ylabel('Implied Volatility')
    ax.set_title('Временная динамика вмененной волатильности')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig, ax, "iv_time_series.png"


def _plot_smile(df, target_date, maturity, figsize=(8,6)):
    """Улыбка волатильности."""
    fig, ax = plt.subplots(figsize=figsize)
    target_date = pd.to_datetime(target_date)
    subset = df[(df['Date'] == target_date) & (df['Maturity'] == maturity)]
    if subset.empty:
        raise ValueError(f"Нет данных для даты {target_date} и срока {maturity}")
    subset = subset.sort_values('Strike')
    ax.plot(subset['Strike'], subset['Volatility'], marker='o', linestyle='-', color='b')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied Volatility')
    ax.set_title(f'Улыбка волатильности: {maturity}, {target_date.date()}')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = f"iv_smile_{target_date.strftime('%Y%m')}_{maturity}.png"
    return fig, ax, filename


def _plot_heatmap(df, target_date, figsize=(10,8)):
    """Тепловая карта: Maturity vs Strike."""
    target_date = pd.to_datetime(target_date)
    subset = df[df['Date'] == target_date]
    if subset.empty:
        raise ValueError(f"Нет данных для даты {target_date}")
    pivot = subset.pivot_table(index='Maturity', columns='Strike', values='Volatility')
    order = subset.groupby('Maturity')['Maturity (year fraction)'].first().sort_values().index
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=False, cmap='viridis', ax=ax, cbar_kws={'label': 'Volatility'})
    ax.set_title(f'Поверхность волатильности (дата: {target_date.date()})')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    plt.tight_layout()
    filename = f"iv_heatmap_{target_date.strftime('%Y%m')}.png"
    return fig, ax, filename


def _plot_3d_surface(df, target_date, figsize=(12,8)):
    """3D-поверхность."""
    target_date = pd.to_datetime(target_date)
    subset = df[df['Date'] == target_date]
    if subset.empty:
        raise ValueError(f"Нет данных для даты {target_date}")

    strikes = sorted(subset['Strike'].unique())
    maturities = sorted(subset['Maturity (year fraction)'].unique())
    X, Y = np.meshgrid(strikes, maturities)
    Z = np.full_like(X, np.nan, dtype=float)

    for _, row in subset.iterrows():
        i = maturities.index(row['Maturity (year fraction)'])
        j = strikes.index(row['Strike'])
        Z[i, j] = row['Volatility']

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity (year fraction)')
    ax.set_zlabel('Volatility')
    ax.set_title(f'3D поверхность вмененной волатильности ({target_date.date()})')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Volatility')
    plt.tight_layout()
    filename = f"iv_3d_surface_{target_date.strftime('%Y%m')}.png"
    return fig, ax, filename


if __name__ == '__main__':
    # Загрузка данных
    iv_df = get_IV_train_dataframe()
    iv_df['Date'] = pd.to_datetime(iv_df['Date'])

    # 1. Временной ряд: средняя волатильность по страйкам для нескольких сроков
    visualize_iv_surface(iv_df, plot_type='time_series',
                         maturities=['1M', '3M', '1Y', '5Y'], agg='mean', save=True)

    # 2. Улыбка для самой последней даты и срока 1Y
    latest_date = iv_df['Date'].max()
    visualize_iv_surface(iv_df, plot_type='smile',
                         target_date=latest_date, maturity='1Y', save=True)

    # 3. Тепловая карта для той же даты
    visualize_iv_surface(iv_df, plot_type='heatmap', target_date=latest_date, save=True)

    # 4. 3D поверхность
    visualize_iv_surface(iv_df, plot_type='surface', target_date=latest_date, save=True)