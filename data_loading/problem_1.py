from pathlib import Path
import pandas as pd


_problem_1_IV_train_file = Path(__file__).parent / 'MacroHack_data/Problem 1' / 'Problem_1_IV_train.xlsx'
_problem_1_yield_curve_predict_file = Path(__file__).parent / 'MacroHack_data/Problem 1' / 'Problem_1_yield_curve_predict.xlsx'
_problem_1_yield_curve_train_file = Path(__file__).parent / 'MacroHack_data/Problem 1' / 'Problem_1_yield_curve_train.xlsx'


def _load_curve_train_file() -> pd.DataFrame:
    """ Загрузка DataFrame из Problem_1_IV_train_file """
    ...


def get_curve_train_dataframe() -> pd.DataFrame:
    """ Получение данных из Problem_1_IV_train_file с заполнением пропусков """
    ...


def get_IV_train_dataframe() -> pd.DataFrame:
    """ Загрузка данных волатильности процентных ставок из файла Problem_1_IV_train.xlsx """
    return pd.read_excel(_problem_1_IV_train_file)


def get_curve_train_file() -> pd.DataFrame:
    """ Получение таблицы с прогнозируемыми данным """
    return pd.read_excel(_problem_1_yield_curve_train_file)