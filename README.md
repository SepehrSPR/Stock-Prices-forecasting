# Stock-Prices-forecasting
This repository contains a comprehensive analysis and forecasting of the stock prices. Using various statistical and machine learning techniques, the project aims to decompose the time series data, identify trends and seasonal components, and make accurate forecasts.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Dependencies](#dependencies)
4. [Usage](#usage)
5. [Analysis and Forecasting](#analysis-and-forecasting)
6. [Error Metrics](#error-metrics)
7. [Results](#results)
8. [License](#license)

## Project Overview

The goal of this project is to analyze the historical stock prices of Petroul and forecast future prices. This includes:

- Data collection and preprocessing
- Time series decomposition
- Statistical analysis (ADF test)
- Forecasting using various models such as Simple Moving Average (SMA), Random Forest, and LSTM
- Combining forecasts and evaluating the performance

## Data

The data used in this project is fetched from the Tehran Stock Exchange (TSE) using the `finpy_tse` library. The dataset includes the adjusted low prices of Petroul stock from `1399-01-01` to `1402-01-01`.

## Dependencies

The following Python libraries are required to run the code:

- numpy
- pandas
- finpy_tse
- mplfinance
- scipy
- matplotlib
- statsmodels
- scikit-learn
- keras

You can install the necessary libraries using the following command:

```bash
pip install numpy pandas finpy_tse mplfinance scipy matplotlib statsmodels scikit-learn keras
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SepehrSPR/Stock-Prices-forecasting
   ```

2. **Run the analysis script**:
   The main script performs the entire analysis and forecasting process. Ensure you have the necessary libraries installed.
   ```bash
   python Stock_prices_forescasting.py
   ```

## Analysis and Forecasting

The analysis and forecasting process includes:

1. **Data Fetching**:
   Fetch historical stock price data using `finpy_tse`.

2. **Data Preprocessing**:
   - Drop unnecessary columns
   - Convert dates to datetime format
   - Set the date as the index

3. **ADF Test**:
   Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity.

4. **Seasonal Decomposition**:
   Decompose the time series into trend, seasonal, and residual components using `seasonal_decompose`.

5. **Simple Moving Average (SMA)**:
   Apply SMA to the trend component for forecasting.

6. **Random Forest**:
   Use Random Forest to forecast the seasonal component.

7. **LSTM Model**:
   Build and train an LSTM model for the residual component.

8. **Combined Forecasting**:
   Combine the forecasts from SMA, Random Forest, and LSTM to generate the final forecast.

## Error Metrics

The performance of the models is evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## Results

The integrated forecasting results are compared with the actual data for the last 20% of the dataset. The error metrics for the combined forecast are also calculated and displayed.
Feel free to customize any section as needed, especially the repository URL and other project-specific details.

**Additional Note**
Its obvious that you can use any other data instead of Tehran stockprices and remember that this code is better to use when adf result is non-stationary
