import operator
from datetime import datetime
import streamlit as st

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
mode_dict = {"sum": sum, "max": max, "min": min, }
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
    df.loc[~valid, new_metric] = None  # or nan
    return df

# Interaction
ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv
}
def interaction(df, operator, metric_1, metric_2):
    new_metric = metric_1 + operator + metric_2
    if new_metric in df.columns:
        print("Already exists")
        return df
    values = ops[operator](df[metric_1], df[metric_2])
    df[new_metric] = values
    return df

# SaveDf
def save(df, name = "", path="./data/"):
    if name == "":
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    df.to_csv(path + name, index=None)

def op_config(operation_type, i, metric_cols):
    if operation_type == "Delay":
        metric = st.selectbox(
            "Metric",
            metric_cols,
            key=f"delay_metric_{i}")

        delay_years = st.number_input(
            "Delay (years)",
            min_value=1,
            step=1,
            key=f"delay_years_{i}")
        return metric, delay_years
    elif operation_type == "Aggregate":
        metric = st.selectbox(
            "Metric",
            metric_cols,
            key=f"delay_metric_{i}")

        mode = st.selectbox(
            "Mode",
            list(mode_dict.keys()),
            key=f"agg_mode_{i}")

        window = st.number_input(
            "Window",
            min_value=1,
            step=1,
            key=f"agg_window_{i}")
        return metric, mode, window
    elif operation_type == "Interaction":
        col1, col2, col3 = st.columns(3)

        with col1:
            metric1 = st.selectbox(
                "Metric 1",
                metric_cols,
                key=f"int_metric1_{i}"
            )

        with col2:
            operator = st.selectbox(
                "Operator",
                list(ops.keys()),
                key=f"int_operator_{i}"
            )

        with col3:
            metric2 = st.selectbox(
                "Metric 2",
                metric_cols,
                key=f"int_metric2_{i}"
            )
        return metric1, operator, metric2

def apply_op(df, session, op_type, i):
    # --------- Delay ---------
    if op_type == "Delay":
        metric = session.get(f"delay_metric_{i}")
        delay_years = session.get(f"delay_years_{i}")

        # Simple example: shift by delay_years within each Symbol
        df = delay(df, delay_years, metric)

    # --------- Aggregate ---------
    elif op_type == "Aggregate":
        metric = session.get(f"agg_metric_{i}")
        mode = session.get(f"agg_mode_{i}")
        window = session.get(f"agg_window_{i}")

        df = aggregate(df, mode, window, metric)
    # --------- Interaction ---------
    elif op_type == "Interaction":
        m1 = session.get(f"int_metric1_{i}")
        m2 = session.get(f"int_metric2_{i}")
        operator_ = session.get(f"int_operator_{i}")

        df = interaction(df, operator_, m1, m2)
    return df

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