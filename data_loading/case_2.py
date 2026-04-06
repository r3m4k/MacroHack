from pathlib import Path
import pandas as pd


_case_2_IV_file = Path(__file__).parent  / 'MacroHack_data/Case 1' / 'Case_2_IV.xlsx'


def get_case_2_IV() -> pd.DataFrame:
    """ Загрузка таблицы Case_2_IV.xlsx """
    return pd.read_excel(_case_2_IV_file)