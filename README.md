# 🏦 Stock Price Predictions

## Overview
This repository focuses on building machine learning models capable of **forecasting stock prices, calculating time-to-event returns, and ranking assets**. Historical finance data covers 2010 to 2026 stock pricing taken from the [yfinance API](https://pypi.org/project/yfinance/) 


## 🎯 Objectives:
1. **Time-series forecasting** to predict future asset behavior
    - Test different types of forecasting models
    - Forecast next month's stock prices for all 5 equities
2. **Survival Analysis** to model the time until significant price movement occurs.
    - Determine amount of days until ≥5% daily increase in stock price
3. **Asset Ranking** to list which assets to prioritize
    - Rank the 5 stocks according to possible daily gains

## Key Findings:
1. Time-Series Forecasting
    - Out of 4 different types of models, ARIMA performs best 4/5 times and is less biased 3/5 times than the 2nd best performer, LGBM.
    - Predicted close prices on the next day March 23 are -0.30% and no more than -2.17% off.
2. Survival Analysis
    - The Cox Proportional Hazards model can effectively distinguish between shorter and longer time for 5% return increase events.  
    - Its Concordance Indices are up to the ~0.89 range which are closer to 1.0 indicating strong ranking performance. 
    - Stocks with higher volatilty tend to experience 5% increase in returns sooner as seen with META (only 19 days until event)
    - Event timing for stable stocks like MSFT, GOOGL, and AAPL having the strongest signals and C-indices. 
3. Ranking
    - NDCG per year per ticker sits at around the ~0.80 mark
    - LGB with lambdarank pairwise shows good ranking potential and some room for improvement
    - Ranking of the 5 stocks so far this year is closely-tied and is as follows:
    1. META
    2. MSFT
    3. AAPL
    4. AMZN
    5. GOOGL


## 📦 Libraries used:
* pandas
* numpy
* yfinance
* seaborn
* matplotlib
* scikit-learn
* statsmodels
* lightgbm
* lifelines
1. Full list of libraries needed for the notebook demo are in `requirements.txt`

### 🔎 Viewing / Installation:
* *Viewing Option:* For complete analysis and demonstration, simply view the notebook file `StockPredictions.ipynb`.
* *Full Installation Option:* To develop on all the code firsthand, clone this repo\
    ```git clone https://github.com/TuringCollegeSubmissions/cgarci-DS.v2.5.3.3.5.git```

## 📠 Author and Contact Information
Developed by: Christine Garcia 

Have any questions? Contact details may be found on my profile.
