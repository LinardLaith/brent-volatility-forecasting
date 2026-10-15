# Forecasting Brent Crude: Returns vs Volatility

A small time-series project testing whether daily Brent crude moves can be forecast, using public data and out-of-sample testing against honest baselines.

## Question

Can simple statistical and machine-learning models forecast (a) the direction/size of tomorrow's Brent return, or (b) the *size* of tomorrow's move (volatility)?

## Data

- Brent crude spot price (`DCOILBRENTEU`) and Henry Hub natural gas spot price (`DHHNGSP`), daily, from FRED (Federal Reserve Bank of St. Louis).
- Common history from January 1997 to early October 2026 (about 6,900 trading days after cleaning).
- Cleaning: aligned the two series, forward-filled gaps of up to 3 days (market holidays), dropped remaining missing values.

## Method

1. **Stationarity.** Augmented Dickey-Fuller tests: Brent prices are not stationary (p = 0.14); Brent log returns are (p ~ 0). All modelling uses log returns.
2. **Time-based split.** Train on 1997 to end-2022 (5,949 days); test on 2023 to October 2026 (988 days). No shuffling.
3. **Baselines.** Naive zero-change forecast, training mean, previous-day return.
4. **Models.** Linear regression, ARIMA(2,0,2) (order chosen by AIC on training data), gradient boosting. Features: five lagged Brent returns, lagged gas return (clipped at +/-30% to limit one-day spikes), 5- and 20-day average absolute return.
5. **Volatility task.** Target is the absolute daily return. Features: lagged absolute return and 5-, 20- and 60-day averages, plus lagged gas absolute return. Compared against the training mean and simple volatility baselines.

## Results

**Forecasting daily returns (test RMSE, lower is better)**

| Model | RMSE | Direction correct |
|---|---|---|
| Naive (zero change) | 0.02600 | n/a |
| Linear regression | 0.02603 | 47.2% |
| ARIMA (2,0,2) | 0.02607 | 48.4% |
| Gradient boosting | 0.02860 | 48.4% |

No model beat the naive baseline. Gradient boosting overfit the training period.

**Forecasting volatility (size of the daily move)**

| Model | RMSE | R-squared |
|---|---|---|
| Training mean | 0.01896 | -0.002 |
| Yesterday's absolute return | 0.02378 | -0.577 |
| 20-day average | 0.01775 | 0.122 |
| Linear regression | 0.01760 | 0.136 |
| Gradient boosting | 0.01964 | -0.076 |

A simple linear model gives about 7% lower error than the training-mean baseline and explains roughly 14% of the variation in the next day's absolute move. Most of the gain comes from volatility clustering (the 20-day average alone reaches R-squared 0.12).

## Conclusions

- Daily Brent return direction is not forecastable from its own history, recent volatility or lagged gas returns in this setup, consistent with efficient-market intuition.
- The *size* of the next move is modestly predictable, because volatility clusters. This is the more useful quantity for risk management.
- Simple models beat complex ones out of sample.

## Limitations

- The forecast tracks volatility regimes with a lag and underestimates the largest spikes (e.g. early 2026).
- Single train/test split; models were not refit through the test period (no walk-forward validation).
- ARIMA order chosen from four candidates only; gradient boosting hyperparameters were not tuned.
- Brent and Henry Hub only; no fundamentals such as inventories. Gas one-day spikes were clipped, not investigated individually.
- No trading strategy, costs or risk-adjusted returns are evaluated.

## Possible extensions

- GARCH-family volatility models as a stronger baseline.
- Walk-forward re-estimation and tuned boosting.
- Add EIA weekly crude inventory data and other fundamentals.
- Test sequence models (LSTM) against the simple baselines.

## Tools

Python, pandas, NumPy, statsmodels, scikit-learn, matplotlib.
