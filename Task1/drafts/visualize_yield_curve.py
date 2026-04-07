import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from data_loading.problem_1 import get_curve_train_dataframe


def visualize_yield_curve(df, plot_type='time_series', save=True, **kwargs):
    """
    Визуализация кривой доходности из Problem_1_yield_curve_train.xlsx.

    Параметры:
    df : pd.DataFrame
        DataFrame с индексом Month (datetime) и колонками теноров:
        ['O/N', '1W', '2W', '1M', '2M', '3M', '6M', '1Y', '2Y'].
    plot_type : str
        Тип графика: 'time_series', 'curve', 'heatmap', 'surface', 'difference'.
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

    # Проверим, что индекс datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    if plot_type == 'time_series':
        fig, ax, filename = _plot_yield_time_series(df, **kwargs)
    elif plot_type == 'curve':
        fig, ax, filename = _plot_yield_curve_at_date(df, **kwargs)
    elif plot_type == 'heatmap':
        fig, ax, filename = _plot_yield_heatmap(df, **kwargs)
    elif plot_type == 'surface':
        fig, ax, filename = _plot_yield_3d_surface(df, **kwargs)
    elif plot_type == 'difference':
        fig, ax, filename = _plot_yield_difference(df, **kwargs)
    else:
        raise ValueError(f"Неизвестный тип графика: {plot_type}. Доступны: 'time_series', 'curve', 'heatmap', 'surface', 'difference'.")

    if save:
        filepath = results_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {filepath}")
    plt.show()
    return fig, ax


def _plot_yield_time_series(df, tenors=None, figsize=(12,6)):
    """
    Временные ряды доходности для выбранных теноров.
    tenors: список названий теноров, например ['O/N', '1Y', '2Y'].
    """
    if tenors is None:
        tenors = df.columns.tolist()
    else:
        tenors = [t for t in tenors if t in df.columns]
    fig, ax = plt.subplots(figsize=figsize)
    for tenor in tenors:
        ax.plot(df.index, df[tenor], label=tenor)
    ax.set_xlabel('Date')
    ax.set_ylabel('Yield (transformed)')
    ax.set_title('Динамика доходности по тенорам')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = "yield_time_series.png"
    return fig, ax, filename


def _plot_yield_curve_at_date(df, target_date, figsize=(8,6)):
    """
    Кривая доходности на фиксированную дату.
    target_date: дата (строка или datetime).
    """
    if target_date not in df.index:
        target_date = pd.to_datetime(target_date)
        if target_date not in df.index:
            raise ValueError(f"Дата {target_date} отсутствует в индексе. Доступны: {df.index[0]} ... {df.index[-1]}")
    yields = df.loc[target_date]
    tenors = df.columns.tolist()
    tenor_to_years = {
        'O/N': 1/365,
        '1W': 7/365,
        '2W': 14/365,
        '1M': 30/365,
        '2M': 60/365,
        '3M': 90/365,
        '6M': 180/365,
        '1Y': 1.0,
        '2Y': 2.0
    }
    x = [tenor_to_years[t] for t in tenors]
    y = yields.values
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, marker='o', linestyle='-', linewidth=2)
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Yield (transformed)')
    date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
    ax.set_title(f'Кривая доходности на {date_str}')
    ax.grid(True, linestyle='--', alpha=0.6)
    for i, txt in enumerate(tenors):
        ax.annotate(txt, (x[i], y[i]), xytext=(5,5), textcoords='offset points', fontsize=8)
    plt.tight_layout()
    filename = f"yield_curve_{target_date.strftime('%Y%m') if hasattr(target_date, 'strftime') else target_date}.png"
    return fig, ax, filename


def _plot_yield_heatmap(df, figsize=(12,8), cmap='viridis'):
    """
    Тепловая карта: даты по оси Y, теныры по оси X, цвет - доходность.
    """
    data = df.copy()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, annot=False, cmap=cmap, ax=ax, cbar_kws={'label': 'Yield'})
    ax.set_title('Тепловая карта доходности (даты x теныры)')
    ax.set_xlabel('Tenor')
    ax.set_ylabel('Date')
    plt.tight_layout()
    return fig, ax, "yield_heatmap.png"


def _plot_yield_3d_surface(df, figsize=(12,8)):
    """
    3D поверхность: оси X - теныры (в годах), Y - даты, Z - доходность.
    """
    tenors = df.columns.tolist()
    tenor_years = {
        'O/N': 1/365,
        '1W': 7/365,
        '2W': 14/365,
        '1M': 30/365,
        '2M': 60/365,
        '3M': 90/365,
        '6M': 180/365,
        '1Y': 1.0,
        '2Y': 2.0
    }
    x = np.array([tenor_years[t] for t in tenors])
    y = np.array(df.index.to_julian_date())
    X, Y = np.meshgrid(x, y)
    Z = df.values
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Date (julian)')
    ax.set_zlabel('Yield')
    ax.set_title('Эволюция кривой доходности (3D)')
    y_ticks = np.linspace(y.min(), y.max(), 5)
    y_tick_labels = [pd.Timestamp.fromordinal(int(yt - 1721424.5)) for yt in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([d.strftime('%Y-%m') for d in y_tick_labels], rotation=45)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Yield')
    plt.tight_layout()
    return fig, ax, "yield_3d_surface.png"


def _plot_yield_difference(df, base_tenor='O/N', figsize=(12,6)):
    """
    График спредов: разница между другими тенорами и базовым тенором.
    """
    if base_tenor not in df.columns:
        raise ValueError(f"Базовый тенор {base_tenor} не найден.")
    fig, ax = plt.subplots(figsize=figsize)
    for tenor in df.columns:
        if tenor == base_tenor:
            continue
        spread = df[tenor] - df[base_tenor]
        ax.plot(df.index, spread, label=f'{tenor} - {base_tenor}')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Yield spread (relative to {base_tenor})')
    ax.set_title(f'Спреды доходности относительно {base_tenor}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = f"yield_spreads_vs_ON.png"
    return fig, ax, filename


if __name__ == "__main__":
    curve_df = get_curve_train_dataframe()

    # 1. Временные ряды для всех теноров
    visualize_yield_curve(curve_df, plot_type='time_series', save=True)

    # 2. Кривая на последнюю дату
    latest_date = curve_df.index[-1]
    visualize_yield_curve(curve_df, plot_type='curve', target_date=latest_date, save=True)

    # 3. Тепловая карта
    # visualize_yield_curve(curve_df, plot_type='heatmap', save=True)

    # 4. 3D поверхность
    visualize_yield_curve(curve_df, plot_type='surface', save=True)

    # 5. Спреды относительно O/N
    visualize_yield_curve(curve_df, plot_type='difference', base_tenor='O/N', save=True)