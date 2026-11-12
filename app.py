import sys
import os
from pathlib import Path

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

#script_path = Path(r"C:\Users\MSUSERSL123\Downloads\MARSH")
#sys.path.append(str(script_path))

# Create database in the same folder
#db_path = script_path / "project_db.db"

script_path = Path(__file__).resolve().parent
sys.path.append(str(script_path))
db_path = script_path / "project_db.db"

from return_modeling_engine import calculate_portfolio_factor_attribution
from return_modeling_engine import calculate_factor_correlations


script_path = Path(__file__).resolve().parent

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Portfolio Risk & Return Attribution",
    page_icon="📊",
    layout="wide"
)

script_path = Path(__file__).resolve().parent
db_path = script_path / "project_db.db"

ETF_TICKERS = [
    "ACWI", "TLT", "LQD", "HYG", "EMB", "USRT", "COMT"
]

LOOKBACK_OPTIONS = [12, 18, 24, 30, 36]
MODEL_OPTIONS = ["OLS", "Ridge"]

MIN_START_MONTH = pd.Period("2019-11", freq="M")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_available_months(db_path):

    with sqlite3.connect(db_path) as conn:
        dates = pd.read_sql_query(
            "SELECT DISTINCT Date FROM etf_prices",
            conn
        )

    dates["Date"] = pd.to_datetime(dates["Date"])

    months = (
        dates["Date"]
        .dt.to_period("M")
        .drop_duplicates()
        .sort_values()
    )

    return [
        month for month in months
        if month >= MIN_START_MONTH
    ]


def format_month(period):
    return period.strftime("%m-%Y")


def parse_month(month_string):
    return pd.Period(month_string, freq="M")


def display_percentage(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("Portfolio Risk & Return Attribution")

st.caption(
    "Portfolio performance, macroeconomic factor attribution "
    "and rolling factor correlations."
)

available_months = get_available_months(db_path)

if not available_months:
    st.error("No ETF price data available in the database.")
    st.stop()

month_options = [
    format_month(month)
    for month in available_months
]


# --------------------------------------------------
# TABS
# --------------------------------------------------

portfolio_tab, correlation_tab = st.tabs([
    "Portfolio Attribution",
    "Factor Correlations"
])


# ==================================================
# TAB 1: PORTFOLIO ATTRIBUTION
# ==================================================

with portfolio_tab:

    st.header("Portfolio Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        start_month = st.selectbox(
            "Start Month",
            month_options,
            index=month_options.index("01-2020")
            if "01-2020" in month_options else 0
        )

    start_period = parse_month(start_month)

    valid_end_months = [
        month for month in available_months
        if month >= start_period
    ]

    valid_end_options = [
        format_month(month)
        for month in valid_end_months
    ]

    with col2:
        end_month = st.selectbox(
            "End Month",
            valid_end_options,
            index=valid_end_options.index("12-2020")
            if "12-2020" in valid_end_options else (len(valid_end_options) - 1)
        )

    with col3:
        rolling_window = st.selectbox(
            "Rolling Lookback (Months)",
            LOOKBACK_OPTIONS,
            index=3
        )

    with col4:
        selected_model = st.selectbox(
            "Attribution Model",
            MODEL_OPTIONS,
            index=0
        )


    st.divider()

    st.subheader("Asset Allocation")

    st.caption(
        "Enter portfolio weights. Total allocation must equal 100%."
    )

    default_allocations = {
        "ACWI": 30.0,
        "TLT": 10.0,
        "LQD": 25.0,
        "HYG": 15.0,
        "EMB": 10.0,
        "USRT": 5.0,
        "COMT": 5.0
    }

    allocation_columns = st.columns(4)

    allocations = {}

    for i, ticker in enumerate(ETF_TICKERS):

        with allocation_columns[i % 4]:

            allocations[ticker] = st.number_input(
                f"{ticker} (%)",
                min_value=0.0,
                max_value=100.0,
                value=default_allocations[ticker],
                step=1.0,
                key=f"allocation_{ticker}"
            )

    total_allocation = sum(allocations.values())

    allocation_col1, allocation_col2 = st.columns(2)

    with allocation_col1:
        st.metric(
            "Total Allocation",
            f"{total_allocation:.2f}%"
        )

    with allocation_col2:
        if np.isclose(total_allocation, 100.0):
            st.success("Allocation is valid")
        else:
            st.warning("Allocations must add up to 100%")

    run_portfolio = st.button(
        "Run Portfolio Attribution",
        type="primary",
        use_container_width=True
    )

    if run_portfolio:

        if not np.isclose(total_allocation, 100.0):
            st.error("Please ensure allocations add up to 100%.")
            st.stop()

        if parse_month(end_month) < parse_month(start_month):
            st.error("End month must be after start month.")
            st.stop()

        with st.spinner("Running portfolio attribution model..."):

            try:

                # Convert allocations into decimal weights
                weights = {
                    ticker: value / 100
                    for ticker, value in allocations.items()
                }

                results = calculate_portfolio_factor_attribution(
                    start_date=start_month,
                    end_date=end_month,
                    allocations=weights,
                    db_path=db_path,
                    rolling_window=rolling_window,
                    model_type = selected_model
                )

                (
                    portfolio_returns,
                    attribution,
                    rolling_betas,
                    cumulative_attribution,
                    selected_factors,
                    model_statistics
                ) = results

                # Save results in session state
                st.session_state["portfolio_results"] = results
                st.session_state["portfolio_model_used"] = selected_model

                st.success("Portfolio attribution completed.")

            except Exception as e:
                st.error(f"Model execution failed: {e}")
                st.stop()

    # Display results from most recent run
    if "portfolio_results" in st.session_state:

        (
            portfolio_returns,
            attribution,
            rolling_betas,
            cumulative_attribution,
            selected_factors,
            model_statistics
        ) = st.session_state["portfolio_results"]

        st.divider()
        st.header("Portfolio Results")

        # ------------------------------------------
        # PERFORMANCE SUMMARY
        # ------------------------------------------

        actual_returns = attribution["Actual_Return"].dropna()
        model_returns = attribution["Model_Return"].dropna()

        if not actual_returns.empty:

            cumulative_actual = (1 + actual_returns).prod() - 1

            cumulative_model = (1 + model_returns).prod() - 1

            unexplained = attribution[
                "Unexplained_Return"
            ].dropna()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Cumulative Portfolio Return",
                display_percentage(cumulative_actual)
            )

            col2.metric(
                "Cumulative Model Return",
                display_percentage(cumulative_model)
            )

            col3.metric(
                "Average Monthly Return",
                display_percentage(actual_returns.mean())
            )

            col4.metric(
                "Average Unexplained Return",
                display_percentage(unexplained.mean())
            )

        # ------------------------------------------
        # ACTUAL VS MODEL RETURN
        # ------------------------------------------

        st.subheader("Actual vs Model Return")

        performance_data = attribution[
            ["Actual_Return", "Model_Return"]
        ].dropna()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=performance_data.index,
            y=performance_data["Actual_Return"],
            name="Actual Return",
            mode="lines"
        ))

        fig.add_trace(go.Scatter(
            x=performance_data.index,
            y=performance_data["Model_Return"],
            name="Model Return",
            mode="lines"
        ))

        fig.update_layout(
            height=400,
            yaxis_tickformat=".1%",
            xaxis_title="Date",
            yaxis_title="Monthly Return",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------
        # FACTOR ATTRIBUTION
        # ------------------------------------------

        st.subheader("Monthly Factor Attribution")

        factor_columns = [
            col for col in attribution.columns
            if col not in [
                "Actual_Return",
                "Model_Return",
                "Regression_Intercept",
                "Unexplained_Return"
            ]
        ]

        fig = px.bar(
            attribution.reset_index(),
            x=attribution.index,
            y=factor_columns,
            title="Monthly Factor Contributions"
        )

        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Return Contribution",
            barmode="relative",
            yaxis_tickformat=".1%"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------
        # CUMULATIVE FACTOR ATTRIBUTION
        # ------------------------------------------

        st.subheader("Cumulative Factor Attribution")

        if isinstance(cumulative_attribution, pd.DataFrame):

            fig = px.line(
                cumulative_attribution,
                title="Cumulative Factor Contributions"
            )

            fig.update_layout(
                height=450,
                yaxis_title="Cumulative Contribution",
                yaxis_tickformat=".1%"
            )

            st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------
        # ROLLING FACTOR BETAS
        # ------------------------------------------

        st.subheader("Rolling Factor Exposures")

        fig = px.line(
            rolling_betas,
            title="Rolling Factor Betas"
        )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------
        # MODEL STATISTICS
        # ------------------------------------------

        st.subheader("Model Statistics")
        st.caption(
            f"Selected model: {st.session_state.get('portfolio_model_used', selected_model)}. Training metrics are calculated "
            "on the preceding rolling lookback window; they are not the current "
            "month's out-of-sample prediction accuracy. Model-specific diagnostics "
            "are shown only where applicable."
        )

        stats_to_display = model_statistics.dropna(axis=1, how="all").dropna(axis=0, how="all")
        st.dataframe(
            stats_to_display,
            use_container_width=True
        )

        # ------------------------------------------
        # ATTRIBUTION TABLE
        # ------------------------------------------

        st.subheader("Attribution Data")

        st.dataframe(
            attribution.style.format("{:.2%}"),
            use_container_width=True
        )

        # Download attribution
        csv = attribution.to_csv().encode("utf-8")

        st.download_button(
            "Download Attribution CSV",
            data=csv,
            file_name="portfolio_attribution.csv",
            mime="text/csv"
        )


# ==================================================
# TAB 2: FACTOR CORRELATIONS
# ==================================================

with correlation_tab:

    st.header("Factor Correlation Analysis")

    st.caption(
        "Calculate correlations between the macroeconomic factors "
        "over a selected historical period."
    )

    corr_col1, corr_col2 = st.columns(2)

    with corr_col1:

        correlation_date = st.selectbox(
            "As-of Month",
            month_options,
            index=len(month_options) - 1,
            key="correlation_date"
        )

    with corr_col2:

        correlation_window = st.selectbox(
            "Correlation Lookback (Months)",
            LOOKBACK_OPTIONS,
            index=2,
            key="correlation_window"
        )

    run_correlation = st.button(
        "Run Factor Correlations",
        type="primary",
        use_container_width=True
    )

    if run_correlation:

        with st.spinner("Calculating factor correlations..."):

            try:

                # Request factor history up to selected date.
                monthly_factors, _ = calculate_factor_correlations(
                    start_date="11-2019",
                    end_date=correlation_date,
                    db_path=db_path,
                    rolling_window=correlation_window
                )

                monthly_factors.index = pd.to_datetime(
                    monthly_factors.index
                )

                selected_date = pd.Period(
                    correlation_date,
                    freq="M"
                ).end_time

                # Use observations through the selected month
                available_factors = monthly_factors.loc[
                    monthly_factors.index <= selected_date
                ]

                # Take the selected lookback period
                correlation_sample = (
                    available_factors
                    .tail(correlation_window)
                    .dropna()
                )

                if len(correlation_sample) < correlation_window:
                    st.warning(
                        f"Only {len(correlation_sample)} complete "
                        "monthly observations are available."
                    )

                if len(correlation_sample) < 2:
                    st.error(
                        "Insufficient observations to calculate correlations."
                    )
                    st.stop()

                correlation_matrix = correlation_sample.corr()

                st.session_state["correlation_matrix"] = (
                    correlation_matrix
                )

                st.session_state["correlation_sample"] = (
                    correlation_sample
                )

                st.success("Factor correlations calculated.")

            except Exception as e:
                st.error(f"Correlation calculation failed: {e}")
                st.stop()

    # Display correlation results
    if "correlation_matrix" in st.session_state:

        correlation_matrix = st.session_state[
            "correlation_matrix"
        ]

        correlation_sample = st.session_state[
            "correlation_sample"
        ]

        st.divider()

        st.subheader("Factor Correlation Heatmap")

        fig = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )

        fig.update_layout(
            height=700,
            coloraxis_colorbar_title="Correlation"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Matrix")

        st.dataframe(
            correlation_matrix.style.format("{:.2f}"),
            use_container_width=True
        )

        st.caption(
            f"Based on {len(correlation_sample)} monthly observations."
        )

        csv = correlation_matrix.to_csv().encode("utf-8")

        st.download_button(
            "Download Correlation Matrix",
            data=csv,
            file_name="factor_correlation_matrix.csv",
            mime="text/csv"
        )
