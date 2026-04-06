from pathlib import Path
import pandas as pd


_case_1_curve = Path(__file__).parent  / 'MacroHack_data/Case 1' / 'Case_1_yield_curve.xlsx'


def get_case_1_curve() -> pd.DataFrame:
    """ Загрузка таблицы Case_2_IV.xlsx """
    return pd.read_excel(_case_1_curve)