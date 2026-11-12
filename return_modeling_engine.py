
from pathlib import Path
import os

import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.decomposition import PCA


def calculate_factor_correlations(
    start_date,
    end_date,
    db_path,
    rolling_window=12
):
    """
    Calculate monthly factor movements and rolling correlations
    between the 12 macroeconomic and market risk factors.
    """

    # --------------------------------------------------
    # 1. Convert dates
    # --------------------------------------------------

    start = pd.Timestamp(
        pd.to_datetime(start_date, format="%m-%Y")
    )

    end = (
        pd.Timestamp(pd.to_datetime(end_date, format="%m-%Y"))
        + pd.offsets.MonthEnd(0)
    )

    if start > end:
        raise ValueError("start_date must be before end_date")

    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")

    start_month_begin = start.to_period("M").start_time

    # Retrieve sufficient historical data for rolling correlations
    history_start = (
        start_month_begin - pd.DateOffset(months=rolling_window)
    )

    # --------------------------------------------------
    # 2. Read data from SQLite
    # --------------------------------------------------

    conn = sqlite3.connect(db_path)

    try:

        macro = pd.read_sql_query(
            """
            SELECT *
            FROM macro_data
            WHERE Date >= (
                SELECT MAX(Date)
                FROM macro_data
                WHERE Date < ?
            )
            AND Date <= ?
            ORDER BY Date
            """,
            conn,
            params=[
                history_start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d")
            ]
        )

        prices = pd.read_sql_query(
            """
            SELECT *
            FROM etf_prices
            WHERE Date >= (
                SELECT MAX(Date)
                FROM etf_prices
                WHERE Date < ?
            )
            AND Date <= ?
            ORDER BY Date
            """,
            conn,
            params=[
                history_start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d")
            ]
        )

    finally:
        conn.close()

    # --------------------------------------------------
    # 3. Prepare daily data
    # --------------------------------------------------

    macro["Date"] = pd.to_datetime(macro["Date"])
    prices["Date"] = pd.to_datetime(prices["Date"])

    # Convert long format (Date, Ticker, Price) to wide format
    prices["Date"] = pd.to_datetime(prices["Date"])

    prices = prices.pivot(
        index="Date",
        columns="Ticker",
        values="Price"
    ).reset_index()

    # Remove column index name
    prices.columns.name = None

    # Sort by date
    prices = prices.sort_values("Date")



    macro = macro.set_index("Date").sort_index()
    prices = prices.set_index("Date").sort_index()



    # Combine daily closing levels
    daily_levels = macro.join(prices, how="outer").sort_index()

    # Forward fill missing observations
    daily_levels = daily_levels.ffill()

    # --------------------------------------------------
    # 4. Convert daily closing levels to month-end levels
    # --------------------------------------------------

    monthly_levels = daily_levels.resample("ME").last()

    # --------------------------------------------------
    # 5. Calculate monthly factor movements
    # --------------------------------------------------

    monthly_factors = pd.DataFrame(
        index=monthly_levels.index
    )

    # US Equity Risk
    spy_return = monthly_levels["SPY"].pct_change(fill_method=None)

    monthly_factors["US_Equity_Risk"] = spy_return

    # US Growth Risk
    qqq_return = monthly_levels["QQQ"].pct_change(fill_method=None)

    monthly_factors["US_Growth_Risk"] = (
        qqq_return - spy_return
    )

    # International Equity Risk
    efa_return = monthly_levels["EFA"].pct_change(fill_method=None)

    monthly_factors["International_Equity_Risk"] = (
        efa_return - spy_return
    )

    # Emerging Market Risk
    eem_return = monthly_levels["EEM"].pct_change(fill_method=None)

    monthly_factors["Emerging_Market_Risk"] = (
        eem_return - efa_return
    )

    # Interest Rate Risk
    monthly_factors["Interest_Rate_Risk"] = (
        monthly_levels["US_10yr"].diff()
    )

    # Yield Curve Risk
    monthly_factors["Yield_Curve_Risk"] = (
        monthly_levels["US_10yr"] -
        monthly_levels["US_2yr"]
    ).diff()

    # US IG Credit Risk
    monthly_factors["US_IG_Credit_Risk"] = (
        monthly_levels["ig_oas"].diff()
    )

    # US HY Credit Risk
    monthly_factors["US_HY_Credit_Risk"] = (
        monthly_levels["hy_oas"].diff()
    )

    # Emerging Market Credit Risk
    monthly_factors["EM_Credit_Risk"] = (
        monthly_levels["SA_10yr"] -
        monthly_levels["US_10yr"]
    ).diff()

    # Commodity Risk
    monthly_factors["Commodity_Risk"] = (
        monthly_levels["PDBC"].pct_change(fill_method=None)
    )

    # Inflation Risk
    monthly_factors["Inflation_Risk"] = (
        monthly_levels["T10YIE"].diff()
    )

    # US Dollar Risk
#    monthly_factors["US_Dollar_Risk"] = (
#        monthly_levels["DXY"].pct_change(fill_method=None)
#    )

    # --------------------------------------------------
    # 6. Calculate rolling correlations
    # --------------------------------------------------

    rolling_corr = {}

    for i in range(rolling_window - 1, len(monthly_factors)):

        current_date = monthly_factors.index[i]

        # Keep only requested output period
        if current_date < start or current_date > end:
            continue

        window_data = monthly_factors.iloc[
            i - rolling_window + 1:i + 1
        ]

        corr_matrix = window_data.corr(
            method="pearson",
            min_periods=rolling_window
        )

        rolling_corr[current_date] = corr_matrix

    # --------------------------------------------------
    # 7. Restrict monthly factor output to requested period
    # --------------------------------------------------

#    monthly_factors = monthly_factors.loc[
#        (monthly_factors.index >= start) &
#        (monthly_factors.index <= end)
#    ]

    return monthly_factors, rolling_corr


def _training_fit_statistics(y_true, y_pred, n_parameters):
    """Common in-sample fit metrics for the rolling training window."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    n = len(y_true)
    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r_squared = np.nan if sst <= 0 else 1.0 - sse / sst
    # Adjusted R-squared is only meaningful for unregularised OLS.
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    return n, r_squared, rmse, mae


def _empty_model_statistics(index):
    # The first four metrics are defined consistently for every model.
    # Model-specific diagnostics are populated only where meaningful.
    return pd.DataFrame(index=index, columns=[
        "Observations", "Training_R2", "Training_RMSE", "Training_MAE",
        "Adjusted_R2", "Number_of_Factors", "Model_P_Value",
        "Best_Alpha", "CV_RMSE", "Num_Components",
        "Explained_Factor_Variance"
    ], dtype=float)


def stepwise_ols(combined, factor_columns,
                 rolling_window, st_date, ed_date, min_improvement=0.003,
                 significance_level=0.05):
    # --------------------------------------------------
    # 8. Initialize output DataFrames
    # --------------------------------------------------

    attribution = pd.DataFrame(
        np.nan,
        index=combined.index,
        columns=[
            "Actual_Return",
            "Model_Return",
            *factor_columns,
            "Regression_Intercept",
            "Unexplained_Return"
        ]
    )

    rolling_betas = pd.DataFrame(
        np.nan,
        index=combined.index,
        columns=[
            "Intercept",
            *factor_columns
        ]
    )

    selected_factors = {}

    model_statistics = _empty_model_statistics(combined.index)

    # --------------------------------------------------
    # 9. Forward stepwise rolling regression
    # --------------------------------------------------

    for i in range(rolling_window, len(combined)):

        current_date = combined.index[i]

        if current_date < st_date or current_date > ed_date:
            continue

        # Previous rolling_window observations
        historical_data = combined.iloc[
            i - rolling_window:i
        ]

        y = historical_data["Portfolio_Return"]

        X = historical_data[factor_columns]

        selected = []
        remaining = factor_columns.copy()

        # Initial intercept-only model
        current_model = sm.OLS(
            y,
            np.ones((len(y), 1))
        ).fit()

        current_adj_r2 = current_model.rsquared_adj

        # ----------------------------------------------
        # Forward selection
        # ----------------------------------------------

        while remaining:

            best_factor = None
            best_model = None
            best_improvement = min_improvement

            for factor in remaining:

                candidate_factors = selected + [factor]

                X_candidate = sm.add_constant(
                    X[candidate_factors],
                    has_constant="add"
                )

                # Avoid singular or redundant regressions
                matrix_rank = np.linalg.matrix_rank(
                    X_candidate.to_numpy()
                )

                if matrix_rank < X_candidate.shape[1]:
                    continue

                candidate_model = sm.OLS(
                    y,
                    X_candidate
                ).fit()

                factor_pvalue = candidate_model.pvalues[factor]

                improvement = (
                        candidate_model.rsquared_adj -
                        current_adj_r2
                )

                if (
                        factor_pvalue < significance_level
                        and improvement > best_improvement
                ):
                    best_improvement = improvement
                    best_factor = factor
                    best_model = candidate_model

            if best_factor is None:
                break

            selected.append(best_factor)

            remaining.remove(best_factor)

            current_model = best_model

            current_adj_r2 = current_model.rsquared_adj

        # ----------------------------------------------
        # Store factor selection
        # ----------------------------------------------

        selected_factors[current_date] = selected.copy()

        # ----------------------------------------------
        # Store regression coefficients
        # ----------------------------------------------

        intercept = current_model.params.iloc[0]

        rolling_betas.loc[
            current_date, "Intercept"
        ] = intercept

        if selected:
            betas = current_model.params[selected]

            rolling_betas.loc[
                current_date, selected
            ] = betas.values

        # Unselected factors have zero contribution
        rolling_betas.loc[
            current_date, factor_columns
        ] = rolling_betas.loc[
            current_date, factor_columns
        ].fillna(0.0)

        # ----------------------------------------------
        # Calculate current month factor contributions
        # ----------------------------------------------

        current_factor_values = combined.loc[
            current_date, factor_columns
        ]

        contributions = (
                rolling_betas.loc[
                    current_date, factor_columns
                ] * current_factor_values
        )

        attribution.loc[
            current_date, factor_columns
        ] = contributions.values

        # ----------------------------------------------
        # Intercept and unexplained return
        # ----------------------------------------------

        attribution.loc[
            current_date, "Regression_Intercept"
        ] = intercept

        actual_return = combined.loc[
            current_date, "Portfolio_Return"
        ]

        predicted_return = (
                intercept + contributions.sum()
        )

        attribution.loc[current_date, "Actual_Return"] = actual_return

        # Model return: intercept + selected factor contributions
        attribution.loc[current_date, "Model_Return"] = predicted_return

        attribution.loc[
            current_date, "Unexplained_Return"
        ] = actual_return - predicted_return

        # ----------------------------------------------
        # Model statistics
        # ----------------------------------------------

        fitted_train = current_model.fittedvalues
        n_obs, train_r2, train_rmse, train_mae = _training_fit_statistics(
            y, fitted_train, len(selected) + 1
        )
        model_statistics.loc[current_date, [
            "Observations", "Training_R2", "Training_RMSE", "Training_MAE",
            "Adjusted_R2", "Number_of_Factors", "Model_P_Value"
        ]] = [
            n_obs, train_r2, train_rmse, train_mae,
            current_model.rsquared_adj, len(selected), current_model.f_pvalue
        ]

    return (attribution, model_statistics, rolling_betas,selected_factors)


def ridge(combined, factor_columns,
          rolling_window):
    # --------------------------------------------------
    # 8. Initialize output DataFrames
    # --------------------------------------------------

    attribution = pd.DataFrame(
        np.nan,
        index=combined.index,
        columns=[
            "Actual_Return",
            "Model_Return",
            *factor_columns,
            "Regression_Intercept",
            "Unexplained_Return"
        ]
    )

    rolling_betas = pd.DataFrame(
        np.nan,
        index=combined.index,
        columns=[
            "Intercept",
            *factor_columns
        ]
    )

    selected_factors = {}

    model_statistics = _empty_model_statistics(combined.index)

    # --------------------------------------------------
    # 9. Ridge regression
    # --------------------------------------------------

    # Ridge regularisation parameters to test
    alpha_grid = {
        "ridge__alpha": [0.01, 0.1, 1, 10, 100]
    }

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    for i in range(rolling_window, len(combined)):
        # Historical observations used for model estimation
        train = combined.iloc[i - rolling_window:i]

        # Current month to be attributed
        current_date = combined.index[i]

        X_train = train[factor_columns]
        y_train = train["Portfolio_Return"]

        X_current = combined.loc[[current_date], factor_columns]

        # Standardise factors and estimate Ridge regression
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(fit_intercept=True))
        ])

        # Select alpha using time-series cross-validation
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=alpha_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Extract fitted scaler and Ridge regression
        scaler = best_model.named_steps["scaler"]
        ridge = best_model.named_steps["ridge"]

        # Convert standardised coefficients back to original factor units
        coefficients = ridge.coef_ / scaler.scale_

        # Convert intercept back to original factor units
        intercept = (
                ridge.intercept_
                - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_)
        )

        # Calculate individual factor contributions
        current_factors = X_current.iloc[0]

        factor_contributions = current_factors * coefficients

        # Regression-explained return
        model_return = intercept + factor_contributions.sum()

        # Actual portfolio return
        actual_return = combined.loc[current_date, "Portfolio_Return"]

        # Populate attribution DataFrame
        attribution.loc[current_date, "Actual_Return"] = actual_return
        attribution.loc[current_date, "Model_Return"] = model_return

        attribution.loc[current_date, factor_columns] = (
            factor_contributions.values
        )

        attribution.loc[current_date, "Regression_Intercept"] = intercept

        attribution.loc[current_date, "Unexplained_Return"] = (
                actual_return - model_return
        )

        # Store rolling Ridge coefficients
        rolling_betas.loc[current_date, factor_columns] = coefficients

        # Common training-fit statistics plus Ridge-specific diagnostics.
        y_pred = best_model.predict(X_train)
        n_obs, train_r2, train_rmse, train_mae = _training_fit_statistics(
            y_train, y_pred, len(factor_columns)
        )
        model_statistics.loc[current_date, [
            "Observations", "Training_R2", "Training_RMSE", "Training_MAE",
            "Best_Alpha", "CV_RMSE"
        ]] = [
            n_obs, train_r2, train_rmse, train_mae,
            grid_search.best_params_["ridge__alpha"],
            np.sqrt(-grid_search.best_score_)
        ]

    return (attribution, model_statistics, rolling_betas,selected_factors)



def get_PCA(combined, factor_columns,
          rolling_window, explained_variance_threshold = 0.90,
            max_components = 7):
    # --------------------------------------------------
    # 8. Initialize output DataFrames
    # --------------------------------------------------

    attribution = pd.DataFrame(
        np.nan,
        index=combined.index,
        columns=[
            "Actual_Return",
            "Model_Return",
            *factor_columns,
            "Regression_Intercept",
            "Unexplained_Return"
        ]
    )

    rolling_betas = pd.DataFrame(
        np.nan,
        index=combined.index,
        columns=[
            "Intercept",
            *factor_columns
        ]
    )

    selected_factors = {}

    model_statistics = _empty_model_statistics(combined.index)


    for i in range(rolling_window, len(combined)):
        # Historical observations for model estimation
        train = combined.iloc[i - rolling_window:i]

        # Current month to be attributed
        current_date = combined.index[i]

        X_train = train[factor_columns]
        y_train = train["Portfolio_Return"]

        X_current = combined.loc[[current_date], factor_columns]

        # 1. Standardise the factors using training data only
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_current_scaled = scaler.transform(X_current)

        # 2. Determine number of principal components
        pca_full = PCA()
        pca_full.fit(X_train_scaled)

        cumulative_variance = np.cumsum(
            pca_full.explained_variance_ratio_
        )

        n_components = np.searchsorted(
            cumulative_variance,
            explained_variance_threshold
        ) + 1

        n_components = min(
            n_components,
            max_components,
            X_train.shape[1]
        )

        # 3. Fit PCA
        pca = PCA(n_components=n_components)

        X_train_pca = pca.fit_transform(X_train_scaled)
        X_current_pca = pca.transform(X_current_scaled)

        # 4. Regress portfolio returns against principal components
        model = LinearRegression(fit_intercept=True)

        model.fit(X_train_pca, y_train)

        # 5. Convert PCA regression coefficients
        # back into original factor coefficients

        beta_standardised = (
                pca.components_.T @ model.coef_
        )

        factor_betas = beta_standardised / scaler.scale_

        # Adjust intercept for PCA and standardisation
        intercept = (
                model.intercept_
                - np.dot(beta_standardised, pca.mean_)
                - np.dot(factor_betas, scaler.mean_)
        )

        # 6. Calculate individual original-factor contributions
        current_factors = X_current.iloc[0]

        factor_contributions = current_factors * factor_betas

        # 7. Calculate model-explained return
        model_return = intercept + factor_contributions.sum()

        # Actual portfolio return
        actual_return = combined.loc[
            current_date, "Portfolio_Return"
        ]

        # 8. Populate attribution DataFrame
        attribution.loc[current_date, "Actual_Return"] = actual_return

        attribution.loc[current_date, "Model_Return"] = model_return

        attribution.loc[current_date, factor_columns] = (
            factor_contributions.values
        )

        attribution.loc[
            current_date, "Regression_Intercept"
        ] = intercept

        attribution.loc[
            current_date, "Unexplained_Return"
        ] = actual_return - model_return

        # 9. Store rolling factor betas
        rolling_betas.loc[current_date, factor_columns] = (
            factor_betas
        )

        # Common training-fit statistics plus PCA-specific diagnostics.
        y_pred = model.predict(X_train_pca)
        n_obs, train_r2, train_rmse, train_mae = _training_fit_statistics(
            y_train, y_pred, n_components
        )
        model_statistics.loc[current_date, [
            "Observations", "Training_R2", "Training_RMSE", "Training_MAE",
            "Num_Components", "Explained_Factor_Variance"
        ]] = [
            n_obs, train_r2, train_rmse, train_mae,
            n_components, pca.explained_variance_ratio_.sum()
        ]


    return (attribution, model_statistics, rolling_betas,selected_factors)




def calculate_geometric_attribution(attribution):
    """
    Convert monthly arithmetic attribution into geometrically
    linked cumulative attribution using the Carino method.

    Factor contributions, regression intercept and unexplained
    return will reconcile to the compounded actual portfolio return.
    """

    result = pd.DataFrame(index=attribution.index)

    component_columns = [
        col for col in attribution.columns
        if col not in ["Actual_Return", "Model_Return"]
    ]

    valid = attribution["Actual_Return"].notna()

    data = attribution.loc[valid].copy()

    if data.empty:
        return result

    monthly_returns = data["Actual_Return"].astype(float)

    if (monthly_returns <= -1).any():
        raise ValueError("Monthly returns cannot be -100% or lower.")

    # Monthly Carino linking coefficient
    def carino_k(return_value):
        if abs(return_value) < 1e-12:
            return 1.0
        return np.log1p(return_value) / return_value

    monthly_k = monthly_returns.apply(carino_k)

    # Cumulative compounded portfolio return
    cumulative_returns = (
        (1 + monthly_returns).cumprod() - 1
    )

    linked = pd.DataFrame(
        0.0,
        index=data.index,
        columns=component_columns
    )

    for i, current_date in enumerate(data.index):

        cumulative_return = cumulative_returns.iloc[i]

        cumulative_k = carino_k(cumulative_return)

        weights = monthly_k.iloc[:i + 1] / cumulative_k

        linked.loc[current_date] = (
            data[component_columns]
            .iloc[:i + 1]
            .mul(weights, axis=0)
            .sum(axis=0)
        )

    # Store geometrically linked factor contributions
    result = linked

    # Store actual and model returns as compounded return series
    result["Actual_Return"] = cumulative_returns

    result["Model_Return"] = (
        (1 + data["Model_Return"]).cumprod() - 1
    )

    return result


def calculate_portfolio_factor_attribution(
    start_date,
    end_date,
    allocations,
    db_path,
    model_type="OLS",
    rolling_window=36,
    significance_level=0.05,
    min_improvement=0.001,
    pca_variance_threshold=0.90,
    pca_components = 5
):
    """
    Calculate monthly portfolio returns and rolling factor attribution
    using the selected rolling regression model (OLS, Ridge or PCA + OLS).

    Portfolio ETFs:
        ACWI, TLT, LQD, HYG, EMB, USRT, COMT

    Factor selection:
        1. Start with an intercept-only regression.
        2. Test each available factor.
        3. Select the factor with the highest improvement
           in adjusted R-squared.
        4. Require statistical significance.
        5. Repeat until no factor qualifies.

    Factor contribution:
        Previous rolling-window beta * Current month factor movement.

    Parameters
    ----------
    start_date : str
        Start month in mm-yyyy format.

    end_date : str
        End month in mm-yyyy format.

    allocations : dict
        ETF weights summing to 1.0.

    db_path : str or Path
        SQLite database location.

    rolling_window : int
        Historical months used for regression. Default 36.

    significance_level : float
        Maximum p-value for factor inclusion. Default 0.05.

    min_improvement : float
        Minimum incremental adjusted R-squared improvement.
        Default 0.001 (0.1 percentage points).

    Returns
    -------
    portfolio_returns : pd.Series

    attribution : pd.DataFrame

    rolling_betas : pd.DataFrame

    cumulative_attribution : pd.DataFrame

    selected_factors : dict

    model_statistics : pd.DataFrame
    """

    # --------------------------------------------------
    # 1. Validate dates
    # --------------------------------------------------

    start = pd.to_datetime(
        start_date,
        format="%m-%Y"
    )

    end = (
        pd.to_datetime(end_date, format="%m-%Y")
        + pd.offsets.MonthEnd(0)
    )

    if start > end:
        raise ValueError("Start date must be before end date")

    if rolling_window < 14:
        raise ValueError(
            "Rolling window must be at least 14 months "
            "for a 12-factor regression."
        )

    if not 0 < significance_level < 1:
        raise ValueError("Invalid significance level")

    # --------------------------------------------------
    # 2. Validate ETF allocations
    # --------------------------------------------------

    required_etfs = [
        "ACWI", "TLT", "LQD", "HYG",
        "EMB", "USRT", "COMT"
    ]

    if not isinstance(allocations, dict):
        raise TypeError("Allocations must be a dictionary")

    if set(allocations) != set(required_etfs):
        raise ValueError(
            f"Allocations must contain exactly: {required_etfs}"
        )

    weights = pd.Series(allocations, dtype=float)

    if not np.isfinite(weights).all():
        raise ValueError("Allocations must be finite")

    if (weights < 0).any():
        raise ValueError("Allocations cannot be negative")

    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(
            f"Allocations must sum to 100%. "
            f"Current total: {weights.sum():.2%}"
        )

    held_etfs = weights[weights > 0].index.tolist()

    # --------------------------------------------------
    # 3. Determine historical data requirement
    # --------------------------------------------------
    start_month_begin = start.to_period("M").start_time

    history_start = (
        start_month_begin - pd.DateOffset(months=rolling_window)
    )


    # --------------------------------------------------
    # 4. Read ETF prices from SQLite
    # --------------------------------------------------

    conn = sqlite3.connect(db_path)

    try:

        prices = pd.read_sql_query(
            """
            SELECT Date, Ticker, Price
            FROM etf_prices
            WHERE Date >= (
                SELECT MAX(Date)
                FROM etf_prices
                WHERE Date < ?
            )
            AND Date <= ?
            AND Ticker IN (
                'ACWI', 'TLT', 'LQD', 'HYG',
                'EMB', 'USRT', 'COMT'
            )
            ORDER BY Date
            """,
            conn,
            params=[
                history_start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d")
            ]
        )

    finally:
        conn.close()

    if prices.empty:
        raise ValueError("No ETF price data found")

    prices["Date"] = pd.to_datetime(prices["Date"])

    # Convert long format into wide format
    prices = prices.pivot(
        index="Date",
        columns="Ticker",
        values="Price"
    ).sort_index()

    prices.columns.name = None

    missing_etfs = set(required_etfs) - set(prices.columns)

    if missing_etfs:
        raise ValueError(
            f"Missing ETF data: {sorted(missing_etfs)}"
        )

    prices = prices[held_etfs]

    # --------------------------------------------------
    # 5. Calculate monthly ETF returns
    # --------------------------------------------------

    monthly_prices = prices.resample("ME").last()

    monthly_etf_returns = monthly_prices.pct_change(
        fill_method=None
    )

    # Fixed-weight monthly rebalanced portfolio
    portfolio_returns_all = (
        monthly_etf_returns.mul(weights[held_etfs], axis=1)
        .sum(axis=1, min_count=len(held_etfs))
    )

    portfolio_returns_all.name = "Portfolio_Return"

    # --------------------------------------------------
    # 6. Get monthly factors from previous function
    # --------------------------------------------------

    (monthly_factors, rolling_corr) = calculate_factor_correlations(
        start_date=start_date,
        end_date=end_date,
        db_path=db_path,
        rolling_window=rolling_window
    )

    # --------------------------------------------------
    # 7. Combine portfolio returns and factors
    # --------------------------------------------------

    combined = pd.concat(
        [
            portfolio_returns_all,
            monthly_factors
        ],
        axis=1,
        join="inner"
    ).sort_index()

    combined = combined.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    factor_columns = monthly_factors.columns.tolist()

    if len(combined) <= rolling_window:
        raise ValueError(
            "Insufficient overlapping historical observations "
            "for rolling regression."
        )


    if model_type== "OLS":
        (attribution, model_statistics,rolling_betas,selected_factors) =  stepwise_ols(combined= combined, factor_columns=factor_columns,
                     rolling_window=rolling_window, st_date=start, ed_date=end,
                    min_improvement=min_improvement, significance_level=significance_level)

    elif model_type== "Ridge":
        (attribution, model_statistics,rolling_betas,selected_factors) =  ridge(combined= combined, factor_columns=factor_columns,
                     rolling_window=rolling_window)

    elif model_type== "PCA":
        (attribution, model_statistics,rolling_betas,selected_factors) =  get_PCA(combined= combined, factor_columns=factor_columns,
                     rolling_window=rolling_window, explained_variance_threshold = pca_variance_threshold,
                        max_components = pca_components)


    # --------------------------------------------------
    # 10. Restrict outputs to requested period
    # --------------------------------------------------

    output_index = combined.index[
        (combined.index >= start) &
        (combined.index <= end)
    ]

    portfolio_returns = portfolio_returns_all.reindex(
        output_index
    )

    attribution = attribution.reindex(output_index)

    rolling_betas = rolling_betas.reindex(output_index)

    model_statistics = model_statistics.reindex(output_index)

    # --------------------------------------------------
    # 11. Cumulative additive factor attribution
    # --------------------------------------------------

    cumulative_attribution = calculate_geometric_attribution(attribution)

    # --------------------------------------------------
    # 12. Return results
    # --------------------------------------------------

    return (
        portfolio_returns,
        attribution,
        rolling_betas,
        cumulative_attribution,
        selected_factors,
        model_statistics
    )



