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


def sharpe_rank(df, window=252, date_range=252, confidence=0.95, time="latest"):
    returns = df["Return"]

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(date_range)

    results = []
    for ticker in returns.columns:
        s = rolling_sharpe[ticker].dropna()
        n = len(s)

        # Report most recent rolling Sharpe ratios or overall according to time param
        sharpe = s.iloc[-1] if time == "latest" else s.mean()

        # Sharpe CI via standard error: se = sqrt((1 + 0.5*sharpe^2) / n or window)
        se = np.sqrt((1 + 0.5 * sharpe**2) / window)

        ci_low, ci_high = stats.t.interval(
            confidence=confidence, df=n - 1, loc=sharpe, scale=se
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
    stat_results["Rank"] = stat_results["Sharpe"].rank(ascending=False).astype(int)
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
) -> tuple[pd.DataFrame, pd.DataFrame, list, list, list]:
    test_rmse_scores = []
    test_predictions, test_actuals = [], []
    fold_starts = []

    for start in range(
        0, len(series) - training_window - forecast_horizon, forecast_horizon
    ):
        train_data = series[start : start + training_window]
        testing_data = series[
            start + training_window : start + training_window + forecast_horizon
        ]

        try:
            _, test_preds = model_fn(train_data, testing_data, forecast_horizon)
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

    results = score_predictions(
        actuals=test_actuals,
        preds=test_predictions,
        rmse_scores=test_rmse_scores,
        model_name=model_name,
        ticker=ticker,
        results=results,
    )

    return results, test_actuals, test_predictions, fold_starts


def ma_baseline_model(train_data, testing_data, forecast_horizon, window=21):
    forecast = np.full(forecast_horizon, np.mean(train_data[-window:]))
    return None, forecast


def ses_model(train_data, testing_data, forecast_horizon):
    model = SimpleExpSmoothing(train_data).fit()
    return None, model.forecast(forecast_horizon)


def arima_model(train_data, testing_data, forecast_horizon):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(train_data, order=(1, 0, 1)).fit()
    return np.array(model.fittedvalues), np.array(
        model.forecast(steps=forecast_horizon)
    )


best_params = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.005,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
}


def find_best_params(tickers, df_diff, training_window=252, forecast_horizon=21):
    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.005, 0.01, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_samples": [5, 10, 20],
    }

    search = RandomizedSearchCV(
        LGBMRegressor(random_state=42, verbose=-1),
        param_distributions=param_grid,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )

    all_X, all_y = [], []

    for ticker in tickers:
        series = df_diff[ticker].dropna().values[:training_window]
        X = np.array(
            [
                series[i : i + forecast_horizon]
                for i in range(len(series) - forecast_horizon)
            ]
        )
        y = series[forecast_horizon:]
        all_X.append(X)
        all_y.append(y)

    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)

    search.fit(pd.DataFrame(all_X), all_y)
    print(f"Best params: {search.best_params_}")

    return search.best_params_


def lgbm_model(train_data, testing_data, forecast_horizon, params=best_params):
    train = np.array(train_data).flatten()

    X_train = np.array(
        [
            train[i : i + forecast_horizon]
            for i in range(len(train) - forecast_horizon * 2)
        ]
    )
    y_train = np.array(
        [
            train[i + forecast_horizon : i + forecast_horizon * 2]
            for i in range(len(train) - forecast_horizon * 2)
        ]
    )

    col_names = [f"lag_{i}" for i in range(forecast_horizon)]
    X_train_df = pd.DataFrame(X_train, columns=col_names)

    model = MultiOutputRegressor(LGBMRegressor(**params, n_jobs=1))
    model.fit(X_train_df, y_train)

    x_input = pd.DataFrame([train[-forecast_horizon:]], columns=col_names)
    preds = model.predict(x_input)[0]

    return None, np.array(preds)


def score_predictions(
    actuals: list,
    preds: list,
    rmse_scores: list,
    model_name: str,
    ticker: str,
    results: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    metric_cols = ["RMSE", "MAE", "MAPE", "Forecast Bias"]

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
    results[metric_cols] = results[metric_cols].astype(float)

    return results


def get_actual_close(tickers, date="2026-03-23"):
    date = pd.Timestamp(date)

    date_str = date.strftime("%Y-%m-%d")
    next_str = (date + pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    data = yf.download(tickers, start=date_str, end=next_str, progress=False)
    return data["Close"].iloc[-1]
