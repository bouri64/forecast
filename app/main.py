# Add etiquette or select from line + better colors
# Save and load filters
# Make refacto
from display_utils import *
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np



st.set_page_config(layout="wide")
st.title("Local Metrics Dashboard")

# ---- Load data ----
@st.cache_data
def load_data():
    if "current_df" not in st.session_state:
        st.session_state.current_df = pd.read_csv("./app/sp500.csv", parse_dates=["Date"], index_col=None)

    return st.session_state.current_df

df = load_data()
# ---- Column roles ----
time_col = "Date"
hue_col = "Symbol"  # optional
ignore_cols = ['Unnamed: 0', 'Year']

metric_cols = [c for c in st.session_state.current_df.columns if c not in [time_col, hue_col] + ignore_cols]

# ---- Initialize session state ----
if "operation_count" not in st.session_state:
    st.session_state.operation_count = 0

# ---- Sidebar button to add operation ----
if st.button("➕ Add Operation"):
    st.session_state.operation_count += 1

st.markdown(f"Total operations: {st.session_state.operation_count}")
st.checkbox("Display resulting DataFrame", value=False, key="show_df")

# Feature enginering
# ---- Render all operations ----
for i in range(st.session_state.operation_count):

    with st.expander(f"Operation {i+1}", expanded=True):

        operation_type = st.selectbox(
            "Select operation",
            ["Delay", "Aggregate", "Interaction"],
            key=f"op_type_{i}"
        )

        config = op_config(operation_type, i, metric_cols)

st.divider()

# ---- Apply operations button ----
if st.button("Apply Operations"):

    for i in range(st.session_state.operation_count):
        op_type = st.session_state.get(f"op_type_{i}")
        df = apply_op(df, st.session_state, op_type, i)

    st.success("Operations applied!")
    st.session_state.current_df = df
# ---- Display resulting DataFrame ----
if st.session_state.show_df:
    st.dataframe(st.session_state.current_df)

# ---- Save button ----
if st.button("Save DF"):
    save(st.session_state.current_df)
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

    hue_values = sorted(st.session_state.current_df[hue_col].dropna().unique())
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
        min_value=st.session_state.current_df[time_col].min().to_pydatetime(),
        max_value=st.session_state.current_df[time_col].max().to_pydatetime(),
        value=(
            st.session_state.current_df[time_col].min().to_pydatetime(),
            st.session_state.current_df[time_col].max().to_pydatetime(),
        )
    )

    # metric
    time_range_metric = st.slider(
        "Time range applied on metrics",
        min_value=st.session_state.current_df[time_col].min().to_pydatetime(),
        max_value=st.session_state.current_df[time_col].max().to_pydatetime(),
        value=(
            st.session_state.current_df[time_col].min().to_pydatetime(),
            st.session_state.current_df[time_col].max().to_pydatetime(),
        )
    )

    selected_metrics = st.multiselect(
    "Metrics to filter on",
    options=metric_cols
    )

    metric_filters = {}

    for metric_col in selected_metrics:
        metric_min = float(st.session_state.current_df[metric_col].min())
        metric_max = float(st.session_state.current_df[metric_col].max())
        # Make safe for Streamlit
        if not np.isfinite(metric_min):
            metric_min = -1.797e+308
        if not np.isfinite(metric_max):
            metric_max = 1.797e+308

        with st.expander(f"{metric_col} filter", expanded=True):
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

# -----------------------
# Validate selection
# -----------------------
if not metrics:
    st.info("👈 Select at least one metric in the sidebar.")
    st.stop()

mask = (
    (st.session_state.current_df[time_col] >= time_range[0]) &
    (st.session_state.current_df[time_col] <= time_range[1]) &
    (st.session_state.current_df[hue_col].isin(selected_hues))
)
df_filtered = st.session_state.current_df[mask]

metric_mask = (
    (st.session_state.current_df[time_col] >= time_range_metric[0]) &
    (st.session_state.current_df[time_col] <= time_range_metric[1])
)
df_filtered_metrics = st.session_state.current_df[metric_mask]
for metric_col, (min_val, max_val) in metric_filters.items():
    cur_mask = df_filtered_metrics[metric_col].between(min_val, max_val)
    valid_symbols = df_filtered_metrics.loc[cur_mask, "Symbol"].unique()
    df_filtered_metrics = df_filtered_metrics[df_filtered_metrics["Symbol"].isin(valid_symbols)] # Not only year with cur_mask is active
    df_filtered = df_filtered[df_filtered["Symbol"].isin(valid_symbols)]


if df_filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

valid = len(df_filtered["Symbol"].unique())

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
            markers=True,
            custom_data=["Symbol"],
            title=f"Metrics over time (linearly rescaled) | Number of results: {valid}"
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
            markers=True,
            custom_data=["Symbol"],
            title=f"Metrics over time | Number of results: {valid}"
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
            markers=True,
            custom_data=["Symbol"],
            title=f"{metric} | Number of results: {valid}"
        )

        fig.update_layout(
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Footer
# -----------------------
st.caption("Streamlit • Plotly • Localhost • Free")
