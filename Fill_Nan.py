import pandas as pd

file_path = "Problem_1_yield_curve_train.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1", parse_dates=["Month"])
df.set_index("Month", inplace=True)

tenors = {
    "O/N": 0,
    "1W": 1/52,
    "2W": 2/52,
    "1M": 1/12,
    "2M": 2/12,
    "3M": 3/12,
    "6M": 0.5
}
short_cols = list(tenors.keys())

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

df_filled = df.apply(fill_long_rates, axis=1)

output_path = "yield_curve_filled.xlsx"
df_filled.to_excel(output_path, sheet_name="Sheet1")
