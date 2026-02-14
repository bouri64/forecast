import pandas as pd
import numpy as np
import operator
from datetime import datetime

# Delay
def delay(df, delay_years, metric):
    new_metric = f"{metric}_{delay_years}y-ago"
    if new_metric in df.columns:
        print("Already exists")
        return df
    df_delayed = df[["Symbol", "Year", metric]].copy()
    df_delayed = df_delayed.drop_duplicates(subset=['Symbol', 'Year'])

    df_delayed["Year"] += delay_years
    df = df.merge(
        df_delayed,
        on=["Symbol", "Year"],
        how="left",
        suffixes=("", f"_{delay_years}y-ago")
    )

    return df

# Aggregation (start by sum)
mode_dict = {"sum": np.sum, "max": np.max, "min": np.min, }
def aggregate(df, mode, window, metric):
    new_metric = f"{metric}_{window}"
    if new_metric in df.columns:
        print("Already exists")
        return df
    df[new_metric] = (
        df
        .groupby("Symbol")[metric]
        .rolling(window=window, min_periods=window)
        .apply(mode_dict[mode])
        .reset_index(level=0, drop=True))
    shifted_year = df.groupby("Symbol")["Year"].shift(window - 1)
    valid = shifted_year == (df["Year"] - (window - 1))
    df.loc[~valid, new_metric] = None  # or np.nan
    return df

# Interaction
ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv
}
def interaction(df, operator, metric_1, metric_2):
    if new_metric in df.columns:
        print("Already exists")
        return df
    new_metric = metric_1 + operator + metric_2
    values = ops[operator](df[metric_1], df[metric_2])
    df[new_metric] = values
    return df

# SaveDf
def save(df, name = "", path="./data/"):
    if name == "":
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    df.to_csv(path + name, index=None)

# df = pd.read_csv("../sec_parser/data/sp500_enriched_2.csv", parse_dates=["Date"], index_col=None)

# metric = "EpsNormalized_3y"
# delay_years = 5
# df = delay(df,delay_years, metric)

# window = 2
# metric = "EpsNormalized"
# mode = "sum"
# df = aggregate(df, mode, window, metric)

# operator = "-"
# metric_1 = "EpsNormalized"
# metric_2 = "EpsRecalulcated"
# df = interaction(df, operator, metric_1, metric_2)
# save(df)