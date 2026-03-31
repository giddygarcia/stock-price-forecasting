import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import scipy.stats as stats
from sklearn.metrics import ndcg_score


def sharpe_rank(df, window=252, date_range=252, confidence=0.95, time="latest"):
    """
    Compute rolling Sharpe ratio for multiple tickers and rank them.

    Args:
        df: Df containing "Return" column for each ticker
        window: Rolling window size in trading days. (default: 252 / 1 trading year)
        date_range: Annualization factor
        confidence: Confidence interval to compute.
        time: "latest" to use only most recent Sharpe, "mean" / else to use average over all time.

    Returns:
        stat_results: df results of Sharpe ratio stat testing.
    """
    returns = df["Return"]
    # benchmark risk-free rate = 10-year US government bond yield
    treasury = yf.download("^TNX", start=df.index[0], end=df.index[-1], progress=False)
    daily_treasury = (treasury["Close"] / 100) / 252
    # ensure that TNX fills all data if it starts later = ffill & bfill
    risk_free_rate = daily_treasury.reindex(returns.index).ffill().bfill()

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std().replace(0, np.nan)
    rolling_sharpe = ((rolling_mean - risk_free_rate.values) / rolling_std) * np.sqrt(
        date_range
    )

    results = []
    for ticker in returns.columns:
        s = rolling_sharpe[ticker].dropna()
        n = len(s)

        if n == 0:
            continue

        # Report most recent rolling Sharpe ratios or overall according to time param
        sharpe = s.iloc[-1] if time == "latest" else s.mean()

        # Sharpe CI via standard error: se = sqrt((1 + 0.5*sharpe^2) / n or window)
        if time == "latest":
            se = np.sqrt((1 + 0.5 * sharpe**2) / window)
        else:
            se = np.sqrt((1 + 0.5 * sharpe**2) / n)

        n_used = window - 1 if time == "latest" else n - 1

        ci_low, ci_high = stats.t.interval(
            confidence=confidence, df=n_used, loc=sharpe, scale=se
        )

        results.append(
            {
                "Ticker": ticker,
                "Sharpe": round(sharpe, 4),
                "CI_Low": round(ci_low, 4),
                "CI_High": round(ci_high, 4),
            }
        )

    stat_results = pd.DataFrame(results)
    stat_results["Rank"] = (
        stat_results["Sharpe"].rank(ascending=False, method="dense").astype(int)
    )
    return stat_results.sort_values("Rank").reset_index(drop=True)


def walk_forward_validation(
    series,
    model_fn,
    ticker: str,
    model_name: str,
    training_window: int = 252,
    forecast_horizon: int = 21,
    results: pd.DataFrame = None,
    series_level=None,
    invert_diff: bool = False,
) -> tuple[pd.DataFrame, list, list, list, object]:
    """
    Perform walk-forward validation training over a trading year to predict the next month.

    Args:
        series: Input time series (differenced values).
        model_fn: Specify model to train and return (model, predictions).
        ticker: Asset identifier.
        model_name: Name of the model.
        training_window: Number of observations per training fold.
        forecast_horizon: Steps ahead to forecast per fold.
        results: DataFrame results to append to.
        series_level: Original series to use for inversion.
        invert_diff: Whether to invert differenced predictions back to price scale.

    Returns:
        results: Updated results df.
        test_actuals: list of true values across all folds.
        test_predictions: list of predictions across all folds.
        fold_starts: list of fold starting indices.
        final_model: Trained model.
    """
    if series_level is not None:
        series_level = np.array(series_level)
    test_rmse_scores = []
    test_predictions, test_actuals = [], []
    fold_starts = []

    for start in range(
        0, len(series) - training_window - forecast_horizon + 1, forecast_horizon
    ):
        train_data = series[start : start + training_window]
        testing_data = series[
            start + training_window : start + training_window + forecast_horizon
        ]

        try:
            _, test_preds = model_fn(train_data, forecast_horizon)
        except Exception as e:
            print(f"[{model_name}] fold {start} failed: {e}")
            continue

        if invert_diff:
            # store last level value for diff inversion
            last_train_value = series_level[start + training_window - 1]
            test_preds = last_train_value + np.cumsum(test_preds)

            actual_level = series_level[
                start + training_window : start + training_window + forecast_horizon
            ]
            test_rmse_scores.append(
                np.sqrt(mean_squared_error(actual_level, test_preds))
            )
            test_actuals.extend(actual_level)
        else:
            test_rmse_scores.append(
                np.sqrt(mean_squared_error(testing_data, test_preds))
            )
            test_actuals.extend(testing_data)

        test_predictions.extend(test_preds)
        fold_starts.append(start + training_window)

    final_model, _ = model_fn(series[:-forecast_horizon], forecast_horizon)

    results = score_predictions(
        actuals=test_actuals,
        preds=test_predictions,
        rmse_scores=test_rmse_scores,
        model_name=model_name,
        ticker=ticker,
        results=results,
    )

    return results, test_actuals, test_predictions, fold_starts, final_model


def ma_baseline_model(train_data, forecast_horizon, window=21):
    """
    Train a baseline Moving Average modell that predicts the mean of the last 'window' / month's values.

    Args:
        train_data: Historical training data.
        forecast_horizon: Number of steps to forecast
        window: Number of recent observations to average (default: 21 for month)

    Returns:
        None: No trained model.
        np.array: Forecasted / predicted values of length forecast_horizon
    """
    forecast = np.full(forecast_horizon, np.mean(train_data[-window:]))
    return None, forecast


def ses_model(train_data, forecast_horizon):
    """
    Train a Simple Exponential Smoothing model for time series forecasting

    Args:
        train_data: Historical training data.
        forecast_horizon: Number of steps to forecast
        window: Number of recent observations to average (default: 21 for month)

    Returns:
        None: No trained model.
        np.array: Forecasted / predicted values of length forecast_horizon
    """
    model = SimpleExpSmoothing(train_data).fit()
    return None, model.forecast(forecast_horizon)


def arima_model(train_data, forecast_horizon):
    """
    Train an ARIMA model on differenced data (1,0,1) for time series forecasting

    Args:
        train_data: Historical training data.
        forecast_horizon: Number of steps to forecast
        window: Number of recent observations to average (default: 21 for month)

    Returns:
        model: Trained ARIMA model object.
        np.array: Forecasted / predicted values of length forecast_horizon
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(train_data, order=(1, 0, 1)).fit()

    preds = model.forecast(steps=forecast_horizon)

    return model, np.array(preds)


def create_more_features(window):
    """
    Create lag-based and statistical features from a sliding window.

    Args:
        window: Series of dates to extract features from.

    Returns:
        pd.DataFrame: Single-row df with newly created feature columns.
    """
    data = list(window)
    mean = np.mean(window)
    std = np.std(window)
    min_ = np.min(window)
    max_ = np.max(window)
    roc = (window[-1] - window[0]) / (abs(window[0]) + 1e-8)
    data += [mean, std, min_, max_, roc]

    col_names = [f"lag_{i}" for i in range(len(window))] + [
        "mean",
        "std",
        "min",
        "max",
        "roc",
    ]
    return pd.DataFrame([data], columns=col_names)


best_params = {
    "subsample": 0.8,
    "n_estimators": 100,
    "min_child_samples": 20,
    "max_depth": 4,
    "learning_rate": 0.005,
    "colsample_bytree": 0.6,
    "random_state": 42,
    "verbose": -1,
}


def find_best_params(tickers, df_diff, training_window=252, forecast_horizon=21):
    """
    Perform randomized hyperparameter search for LGBM MultiOutput regressor.

    Args:
        tickers: List of asset tickers.
        df_diff: Differenced series dictionary keyed by ticker.
        training_window: Training window length  (Default: 252 / 1 trading year).
        forecast_horizon: Forecast horizon (default: 21 / 1 trading month).

    Returns:
        dict: Best parameter dictionary from RandomizedSearchCV.
    """
    param_grid = {
        "estimator__n_estimators": [100, 200, 300, 500],
        "estimator__max_depth": [2, 3, 4, 5],
        "estimator__learning_rate": [0.005, 0.01, 0.05, 0.1],
        "estimator__subsample": [0.6, 0.7, 0.8, 1.0],
        "estimator__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "estimator__min_child_samples": [5, 10, 20],
    }

    search = RandomizedSearchCV(
        MultiOutputRegressor(LGBMRegressor(random_state=42, verbose=-1)),
        param_distributions=param_grid,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )

    all_X, all_y = [], []

    for ticker in tickers:
        series = df_diff[ticker]["diff"]
        # X = last 21 values, y = next 21 values
        X = pd.concat(
            [
                create_more_features(series[i - forecast_horizon : i])
                for i in range(forecast_horizon, len(series) - forecast_horizon)
            ],
            ignore_index=True,
        )
        y = np.array(
            [
                series[i : i + forecast_horizon]
                for i in range(forecast_horizon, len(series) - forecast_horizon)
            ]
        )
        all_X.append(X)
        all_y.append(y)

    X_df = pd.concat(all_X, ignore_index=True)
    y_array = np.vstack(all_y)

    search.fit(X_df, y_array)
    print(f"Best params: {search.best_params_}")
    return search.best_params_


def lgbm_model(train_data, forecast_horizon, params=best_params):
    """
    Train LGB model on differenced data for time series forecasting

    Args:
        train_data: Historical training data.
        forecast_horizon: Number of steps to forecast
        params: Dict of best found hyperparameters

    Returns:
        model: Fitted LGB model object.
        np.array: Forecasted / predicted values of length forecast_horizon
    """
    train = np.array(train_data).flatten()

    X_train = []
    y_train = []

    for i in range(forecast_horizon, len(train) - forecast_horizon):
        window = train[i - forecast_horizon : i]
        features = create_more_features(window)
        X_train.append(features)
        y_train.append(train[i : i + forecast_horizon])

    X_train_df = pd.concat(X_train, ignore_index=True)
    y_train = np.array(y_train)

    model = MultiOutputRegressor(LGBMRegressor(**params, n_jobs=1))
    model.fit(X_train_df, y_train)

    x_input = create_more_features(train[-forecast_horizon:])
    preds = model.predict(x_input)[0]

    return model, np.array(preds)


def score_predictions(
    actuals: list,
    preds: list,
    rmse_scores: list,
    model_name: str,
    ticker: str,
    results: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score predictions and append to results DataFrame.

    Args:
        actuals: True values.
        preds: Predicted values.
        rmse_scores: RMSE for each fold/segment.
        model_name: Model identifier.
        ticker: Asset ticker.
        results: Existing results DataFrame.

    Returns:
        pd.DataFrame: Updated results DataFrame with new row appended.
    """

    metrics = ["RMSE", "MAE", "MAPE", "Forecast Bias"]

    actuals = np.array(actuals)
    preds = np.array(preds)

    row = {
        "Ticker": ticker,
        "Model": model_name,
        "RMSE": np.mean(rmse_scores),
        "MAE": mean_absolute_error(actuals, preds),
        "MAPE": np.mean(np.abs((actuals - preds) / np.where(actuals == 0, 1, actuals)))
        * 100,
        "Forecast Bias": np.mean(actuals - preds),
    }

    new_row = pd.DataFrame([row])

    if results is None or not isinstance(results, pd.DataFrame) or results.empty:
        results = new_row
    else:
        results = pd.concat([results, new_row], ignore_index=True)

    results = results.convert_dtypes()
    for metric in metrics:
        results[metric] = results[metric].round(4)
    results[metrics] = results[metrics].astype(float)

    return results


def plot_model_forecast(
    ticker,
    model_type,
    ax,
    df_close,
    final_models,
    forecast_df=None,
    logo_palette=None,
    forecast_horizon=21,
    history_length=252,
):
    """
    Plot forecast for a single ticker and model on the provided axis based on parameters.
    """
    # historical prices
    history = df_close[ticker].values[-history_length:]
    anchor = df_close[ticker].iloc[-1]
    history_dates = df_close.index[-history_length:]
    last_date = history_dates[-1]
    future_dates = pd.bdate_range(start=last_date, periods=forecast_horizon + 1)[1:]

    if model_type.lower() == "arima":
        saved_model = final_models[ticker]["arima"]
        preds = anchor + np.cumsum(saved_model.forecast(steps=forecast_horizon))
        title = f"{ticker}: ARIMA Forecast"

    elif model_type.lower() == "lgbm":
        saved_model = final_models[ticker]["lgbm"]
        x_input = create_more_features(forecast_df[ticker]["diff"][-forecast_horizon:])
        preds = anchor + np.cumsum(saved_model.predict(x_input)[0])
        title = f"{ticker}: LGBM Forecast"

    else:
        raise ValueError("model_type must be 'arima' or 'lgbm'")

    ax.plot(
        history_dates, history, color=logo_palette[ticker], linewidth=1, label="History"
    )
    ax.plot(
        future_dates,
        preds,
        color="purple",
        linewidth=1.5,
        linestyle="--",
        label="Forecast",
    )
    ax.axvline(x=last_date, color="gray", linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.legend(fontsize=8)


def get_actual_close(tickers, date="2026-03-23"):
    """
    Use yfinance API to fetch actual closing prices for given tickers on a specific trading date.

    Args:
        tickers: list of asset tickers.
        date: Date of interest. (Default: "2026-03-23" / one day forward from data)

    Returns:
        pd.Series: Closing prices of tickers on given date.
    """
    date = pd.Timestamp(date)

    date_str = date.strftime("%Y-%m-%d")
    next_str = (date + pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    data = yf.download(tickers, start=date_str, end=next_str, progress=False)
    return data["Close"].iloc[-1]


def create_features_rank(
    df, target_cols=["Return", "Volatility"], lags=5, roll_windows=[3, 5]
):
    """
    Create lag and rolling features for ranking or predictive modeling.

    Args:
        df: DataFrame with columns like 'Return', 'Volatility', etc.
        target_cols: Columns to generate features for (default: ['Return', 'Volatility']).
        lags: Maximum lag to create (default: 5, creates lag_1 to lag_5).
        roll_windows: List of rolling window sizes for mean features (default: [3, 5]).

    Returns:
        pd.DataFrame: DataFrame with original columns plus lag and rolling features.
    """
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    for col in target_cols:
        # Lag features
        for lag in range(1, lags + 1):
            df[f"{col}_lag{lag}"] = df.groupby("Ticker", sort=False)[col].shift(lag)

        # Rolling mean features (shift by 1 to avoid lookahead bias)
        for w in roll_windows:
            df[f"{col}_roll{w}"] = df.groupby("Ticker", sort=False)[col].transform(
                lambda x, w=w: x.shift(1).rolling(w).mean()
            )

    # Drop rows with NaNs from lag/rolling computation
    feature_cols = [
        f"{col}_lag{lag}" for col in target_cols for lag in range(1, lags + 1)
    ] + [f"{col}_roll{w}" for col in target_cols for w in roll_windows]

    return df.dropna(subset=feature_cols), feature_cols


def compute_ndcg(group, k=5):
    y_true = group["rank_for_model"].values.reshape(1, -1)
    y_score = group["Score"].values.reshape(1, -1)
    return ndcg_score(y_true, y_score, k=k)
