import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Local Metrics Dashboard")

# ---- Load data ----
@st.cache_data
def load_data():
    # CSV example
    # df = pd.read_csv("sp500.csv", parse_dates=["Date"], index_col=None)
    df = pd.read_csv("../sec_parser/data/sp500_enriched_2.csv", parse_dates=["Date"], index_col=None)

    # # SQL example
    # engine = create_engine("sqlite:///data.db")
    # df = pd.read_sql("metrics", engine, parse_dates=["Date"])
    return df

df = load_data()

# ---- Column roles ----
time_col = "Date"
hue_col = "Symbol"  # optional
ignore_cols = ['Unnamed: 0', 'Year']

metric_cols = [c for c in df.columns if c not in [time_col, hue_col] + ignore_cols]


# -----------------------
# SIDEBAR CONTROLS
# -----------------------
with st.sidebar:
    st.header("⚙️ Controls")

    metrics = st.multiselect(
        "Metrics",
        metric_cols,
        default=metric_cols[0]
    )

    combine_metrics = st.checkbox(
        "Combine metrics in one graph",
        value=True
    )

    scale_metrics = st.checkbox(
        "Scale metrics: only works with 2 metrics + falsifies numbers",
        value=False
    )

    hue_values = sorted(df[hue_col].dropna().unique())
    options = ["All"] + hue_values
    selected_hues = st.multiselect(
        f"Filter {hue_col}",
        options,
        default=hue_values[0]
    )
    if "All" in selected_hues:
        selected_hues = hue_values

    time_range = st.slider(
        "Time range",
        min_value=df[time_col].min().to_pydatetime(),
        max_value=df[time_col].max().to_pydatetime(),
        value=(
            df[time_col].min().to_pydatetime(),
            df[time_col].max().to_pydatetime(),
        )
    )

    # # metric

    # metric_col = st.selectbox(
    #     "Metric to filter on",
    #     options=metric_cols
    # )

    # metric_min, metric_max = float(df[metric_col].min()), float(df[metric_col].max())
    # col1, col2 = st.columns(2)

    # with col1:
    #     min_val = st.number_input(
    #         f"{metric_col} min",
    #         min_value=metric_min,
    #         max_value=metric_max,
    #         value=metric_min
    #     )

    # with col2:
    #     max_val = st.number_input(
    #         f"{metric_col} max",
    #         min_value=metric_min,
    #         max_value=metric_max,
    #         value=metric_max
    #     )

    # value_range = (min_val, max_val)

    # # metric
    selected_metrics = st.multiselect(
    "Metrics to filter on",
    options=metric_cols
    )

    metric_filters = {}

    for metric_col in selected_metrics:
        metric_min = float(df[metric_col].min())
        metric_max = float(df[metric_col].max())

        with st.expander(f"{metric_col} filter", expanded=True):
            # operator = st.selectbox(
            # "Operator",
            # options=[">=", "<=", "between"],
            # index=2,
            # key=f"{metric_col}_op"
            #     )

            col1, col2 = st.columns(2)

            with col1:
                min_val = st.number_input(
                    f"{metric_col} min",
                    min_value=metric_min,
                    max_value=metric_max,
                    value=metric_min,
                    key=f"{metric_col}_min"
                )

            with col2:
                max_val = st.number_input(
                    f"{metric_col} max",
                    min_value=metric_min,
                    max_value=metric_max,
                    value=metric_max,
                    key=f"{metric_col}_max"
                )
            metric_filters[metric_col] = (min_val, max_val)

        # metric_filters[metric_col] = {
        #     "op": operator,
        #     "min": min_val,
        #     "max": max_val
        # }

# -----------------------
# Validate selection
# -----------------------
if not metrics:
    st.info("👈 Select at least one metric in the sidebar.")
    st.stop()

# -----------------------
# Filter data
# -----------------------
# df_filtered = df[
#     (df[time_col] >= time_range[0]) &
#     (df[time_col] <= time_range[1]) &
#     (df[metric_col].between(value_range[0], value_range[1])) &
#     (df[hue_col].isin(selected_hues))
# ]

mask = (
    (df[time_col] >= time_range[0]) &
    (df[time_col] <= time_range[1]) &
    (df[hue_col].isin(selected_hues))
)

for metric_col, (min_val, max_val) in metric_filters.items():
    mask &= df[metric_col].between(min_val, max_val)

df_filtered = df[mask]


# mask = (
#     (df[time_col] >= time_range[0]) &
#     (df[time_col] <= time_range[1]) &
#     (df[hue_col].isin(selected_hues))
# )

# for metric_col, cfg in metric_filters.items():
#     if cfg["op"] == "between":
#         mask &= df[metric_col].between(cfg["min"], cfg["max"])
#     elif cfg["op"] == ">=":
#         mask &= df[metric_col] >= cfg["min"]
#     elif cfg["op"] == "<=":
#         mask &= df[metric_col] <= cfg["min"]

# df_filtered = df[mask]

if df_filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

# -----------------------
# Plotting
# -----------------------
if combine_metrics:
    if scale_metrics and len(metrics) == 2:
        scale_factor = 100  # <-- choose your linear multiplier
        metric_to_scale = metrics[0] # the one with the smaller/larger range

        df_long = df_filtered.melt(
            id_vars=[time_col, hue_col],
            value_vars=metrics,
            var_name="metric",
            value_name="value"
        )
        scale_factor = df_long.loc[
            df_long["metric"] == metrics[1], "value"
        ].max() / df_long.loc[
            df_long["metric"] == metric_to_scale, "value"
        ].max()
        # Apply linear scaling to ONE metric
        df_long.loc[
            df_long["metric"] == metric_to_scale, "value"
        ] *= scale_factor

        fig = px.line(
            df_long,
            x=time_col,
            y="value",
            color="metric",
            line_dash=hue_col,
            title="Metrics over time (linearly rescaled)"
        )

        fig.update_layout(
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        # ---- ONE GRAPH, MULTIPLE METRICS ----
        df_long = df_filtered.melt(
            id_vars=[time_col, hue_col],
            value_vars=metrics,
            var_name="metric",
            value_name="value"
        )

        fig = px.line(
            df_long,
            x=time_col,
            y="value",
            color="metric",
            line_dash=hue_col,
            title="Metrics over time"
        )

        fig.update_layout(
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    # ---- ONE GRAPH PER METRIC ----
    for metric in metrics:
        fig = px.line(
            df_filtered,
            x=time_col,
            y=metric,
            color=hue_col,
            title=metric
        )

        fig.update_layout(
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Footer
# -----------------------
st.caption("Streamlit • Plotly • Localhost • Free")