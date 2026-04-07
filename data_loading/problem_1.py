from pathlib import Path
import pandas as pd
import numpy as np


_problem_1_IV_train_file = Path(__file__).parent.parent / 'MacroHack_data/Problem 1' / 'Problem_1_IV_train.xlsx'
_problem_1_yield_curve_predict_file = Path(__file__).parent.parent / 'MacroHack_data/Problem 1' / 'Problem_1_yield_curve_predict.xlsx'
_problem_1_yield_curve_train_file = Path(__file__).parent.parent / 'MacroHack_data/Problem 1' / 'Problem_1_yield_curve_train.xlsx'


def _load_curve_train_file() -> pd.DataFrame:
    """ Загрузка DataFrame из Problem_1_IV_train_file """
    return pd.read_excel(_problem_1_yield_curve_train_file,
                         parse_dates=["Month"])


def get_curve_train_dataframe() -> pd.DataFrame:
    """ Получение данных из Problem_1_yield_curve_train_file с заполнением пропусков """
    tenors = {
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

    def fill_long_rates(row):
        if not pd.isna(row["1Y"]) and not pd.isna(row["2Y"]):
            return row
        x, y = [], []
        for col, t in tenors.items():
            v = row[col]
            if not pd.isna(v):
                x.append(t)
                y.append(v)

        slope, intercept = np.polyfit(x, y, 1)

        if pd.isna(row["1Y"]):
            row["1Y"] = slope * 1.0 + intercept
        if pd.isna(row["2Y"]):
            row["2Y"] = slope * 2.0 + intercept
        return row

    df = _load_curve_train_file()
    df.set_index("Month", inplace=True)
    return  df.apply(fill_long_rates, axis=1)



def get_IV_train_dataframe() -> pd.DataFrame:
    """ Загрузка данных волатильности процентных ставок из файла Problem_1_IV_train.xlsx """
    return pd.read_excel(_problem_1_IV_train_file, parse_dates=['Date'])


def get_curve_predict_dataframe() -> pd.DataFrame:
    """ Получение таблицы с прогнозируемыми данным """
    return pd.read_excel(_problem_1_yield_curve_predict_file, parse_dates=['Date'])


if __name__ == "__main__":
    print(get_IV_train_dataframe())
    print(get_curve_train_dataframe())
    print(get_curve_predict_dataframe())
