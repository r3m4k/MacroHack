from pathlib import Path
import pandas as pd


_case_2_IV_file = Path(__file__).parent.parent  / 'MacroHack_data/Case 2' / 'Case_2_IV.xlsx'
_key_rate_file = Path(__file__).parent.parent / 'MacroHack_data' / 'Case 2' / 'Инфляция и ключевая ставка Банка России_F01_02_2019_T10_04_2026.xlsx'

def get_case_2_IV() -> pd.DataFrame:
    """ Загрузка таблицы Case_2_IV.xlsx """
    return pd.read_excel(_case_2_IV_file, parse_dates=['Date'])


def get_key_rate_dataframe() -> pd.DataFrame:
    """ Загрузка данных о ключевой ставке и инфляции. """
    df = pd.read_excel(_key_rate_file, sheet_name=0)

    # Названия колонок
    column_mapping = {
        'Дата': 'Date',
        'Ключевая ставка, % годовых': 'Key Rate',
        'Инфляция, % г/г': 'Inflation',
        'Цель по инфляции': 'Inflation Target'
    }
    df = df.rename(columns=column_mapping)

    df['Date'] = df['Date'].astype(str).str.strip()

    # Преобразование дат
    df['Date'] = pd.to_datetime(df['Date'], format='%m.%Y', errors='coerce')

    # Преобразование числовых колонок (замена запятых на точки, если нужно)
    for col in ['Key Rate', 'Inflation']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[['Date', 'Key Rate', 'Inflation']].iloc[::-1].reset_index(drop=True)
    return df


if __name__ == "__main__":
    # print(f'Характеристики таблицы из файла {_case_2_IV_file.name}\n')
    # df = get_case_2_IV()
    #
    # # 0. Временной охват
    # print('\n--------------------------------\n'
    #       'Структура таблицы на 2025-09-01:\n')
    # print(df[df['Date'] == '2025-09-01'])
    #
    # # 1. Временной охват
    # print('\n--------------------------------\n'
    #       'Временной охват:\n')
    # print(df['Date'].min(), df['Date'].max())
    # # Ожидаем: 2019-03-01 → 2026-03-01
    #
    # # 2. Уникальные сроки экспирации
    # print('\n--------------------------------\n'
    #       'Уникальные сроки экспирации:\n')
    # print(df['Maturity'].unique())
    #
    # # 3. Уникальные страйки
    # print('\n--------------------------------\n'
    #       'Уникальные страйки:\n')
    # print(sorted(df['Strike'].unique()))
    #
    # # 4. Пропуски
    # print('\n--------------------------------\n'
    #       'Пропуски:\n')
    # print(df.isnull().sum())
    #
    # # 5. Сколько дат наблюдений
    # print('\n--------------------------------\n'
    #       'Сколько дат наблюдений:\n')
    # print(df['Date'].nunique())
    #
    # # 6. Полнота сетки на каждую дату
    # print('\n--------------------------------\n'
    #       'Полнота сетки на каждую дату:\n')
    # check = df.groupby('Date')[['Maturity','Strike']].nunique()
    # print(check.describe())

    print(get_key_rate_dataframe())