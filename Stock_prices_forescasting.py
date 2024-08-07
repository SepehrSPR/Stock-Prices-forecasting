import numpy as np
import pandas as pd
import finpy_tse as tse
import mplfinance as mplf
import scipy.stats as stt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


DF4 = tse.Get_Price_History(stock='پترول',
                            start_date='1399-01-01',
                            end_date='1402-01-01',
                            ignore_date=False,
                            adjust_price=True,
                            show_weekday=True,
                            double_date=True)
DropList = ['Open', 'High', 'Low', 'Close', 'Final',
            'No', 'Ticker', 'Name', 'Adj Open','Adj High','Adj Close','Adj Final']
DF4.drop(columns=DropList, axis=1, inplace=True)
DF4.to_csv('F:\Arshad\Database-BigData\Last assignment\petroulstock.csv', index=False)
DF4['Date'] = pd.to_datetime(DF4['Date'])
DF4.set_index('Date', inplace=True)
DF4['Adj Low'].plot(title='Stock Price Over Time', xlabel='Date', ylabel='Price', figsize=(10, 6))
plt.show()
result = adfuller(DF4['Adj Low'],autolag='AIC')
print('ADF Statistic:', (result[0]))
print('p-value:', (result[1]))
print('the num of lags used :',result[2])
print('num of observation used',result[3])
for key, value in result[4].items():
    print('Critical Values:')
    print(f'   {key}: {value}')

decompose_model= seasonal_decompose(DF4['Adj Low'], period=5,model='multiplicative')
#Plot the original time series, trend, seasonal and random components
fig, axarr= plt.subplots(4, sharex=True)
fig.set_size_inches(5.5, 5.5)
DF4['Adj Low'].plot(ax=axarr[0], color='b', linestyle='-')
axarr[0].set_title('Adjust Low')
pd.Series(data=decompose_model.trend, index=DF4.index).plot(color='r', linestyle='-', ax=axarr[1])
axarr[1].set_title('Trend component in Adjust Low')
pd.Series(data=decompose_model.seasonal, index=DF4.index).plot(color='g', linestyle='-', ax=axarr[2])
axarr[2].set_title('Seasonal component in Adjust Low')
pd.Series(data=decompose_model.resid, index=DF4.index).plot(color='k', linestyle='-', ax=axarr[3])
axarr[3].set_title('Irregular variations in Adjust Low')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
plt.xticks(rotation=10)
plt.show()


# Extract the trend component and drop NaN values
trend_component = decompose_model.trend.fillna(method='bfill').fillna(method='ffill')

# Apply Simple Moving Average (SMA) for forecasting
window_size = 5  # You can adjust this window size
sma_forecast = trend_component.rolling(window=window_size).mean()

# Extend the SMA forecast to the length of the trend component
sma_forecast = sma_forecast.fillna(method='ffill').fillna(method='bfill')

# Calculate error metrics
mae = mean_absolute_error(trend_component, sma_forecast)
mse = mean_squared_error(trend_component, sma_forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((trend_component - sma_forecast) / trend_component)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(trend_component.index, trend_component, label='Actual Trend')
plt.plot(trend_component.index, sma_forecast, label=f'SMA Forecast (window={window_size})', color='red')
plt.legend()
plt.title('SMA Forecast vs Actual Trend')
plt.show()


seasonal_component = decompose_model.seasonal.fillna(method='bfill').fillna(method='ffill').reset_index()
seasonal_component.columns = ['ds', 'y']

# Prepare the features (lags)
seasonal_component['lag1'] = seasonal_component['y'].shift(1)
seasonal_component['lag2'] = seasonal_component['y'].shift(2)
seasonal_component['lag3'] = seasonal_component['y'].shift(3)
seasonal_component.dropna(inplace=True)

# Split the data into training and testing sets
X = seasonal_component[['lag1', 'lag2', 'lag3']]
y = seasonal_component['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Forecast using the trained Random Forest model
y_pred = rf.predict(X_test)

# Calculate error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_train.index, y_train, label='Actual Seasonal Component')
plt.plot(y_test.index, y_test, label='Actual Seasonal Component')
plt.plot(y_test.index, y_pred, label='Random Forest Forecast', color='red')
plt.legend()
plt.title('Random Forest Forecast vs Actual Seasonal Component')
plt.xlabel('Date')
plt.ylabel('Seasonal Component')
plt.show()

residual_component = decompose_model.resid.fillna(method='bfill').fillna(method='ffill').values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(residual_component)
residual_component_scaled = scaler.transform(residual_component)

# Train-test split
train_size = int(len(residual_component_scaled) * 0.8)
test_size=len(residual_component_scaled) -train_size
train, test=residual_component_scaled[0:train_size,:], residual_component_scaled[train_size:len(residual_component_scaled),:]

# Function to create dataset
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Prepare the data for LSTM
time_step = 5  # Adjust time_step based on your data
trainX, trainY=create_dataset(train, time_step)
testX, testY=create_dataset(test, time_step)

trainX=np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX=np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model=Sequential()
model.add(LSTM(50, input_shape=(1, time_step)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)

# make predictions
trainPredict=model.predict(trainX)
testPredict=model.predict(testX)
# invert predictions
# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY.reshape(-1, 1))


# Calculate MSE, MAE, and MAPE for the LSTM model
train_mse = mean_squared_error(trainY, trainPredict)
train_mae = mean_absolute_error(trainY, trainPredict)
train_mape = np.mean(np.abs((trainY - trainPredict) / trainY)) * 100
 
if np.isnan(testY).any():
    # Forward fill
    testY = pd.Series(testY.flatten()).fillna(method='ffill').values.reshape(-1, 1)
    # If forward fill doesn't fill all nulls, use backward fill as well
    testY = pd.Series(testY.flatten()).fillna(method='bfill').values.reshape(-1, 1)
    
if np.isnan(testPredict).any():
    # Forward fill
    testPredict = pd.Series(testY.flatten()).fillna(method='ffill').values.reshape(-1, 1)
    # If forward fill doesn't fill all nulls, use backward fill as well
    testPredict = pd.Series(testY.flatten()).fillna(method='bfill').values.reshape(-1, 1)
    
    
test_mse = mean_squared_error(testY, testPredict)
test_mae = mean_absolute_error(testY, testPredict)
test_mape = np.mean(np.abs((testY - testPredict) / testY)) * 100

print(f"Train MSE: {train_mse:.2f}")
print(f"Train MAE: {train_mae:.2f}")
print(f"Train MAPE: {train_mape:.2f}%")

print(f"Test MSE: {test_mse:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test MAPE: {test_mape:.2f}%")

# shift train predictions for plotting
trainPredictPlot=np.empty_like(residual_component_scaled)
trainPredictPlot[:, :] =np.nan
trainPredictPlot[time_step:len(trainPredict)+time_step, :] =trainPredict
# shift test predictions for plotting
testPredictPlot=np.empty_like(residual_component_scaled)
testPredictPlot[:, :] =np.nan
testPredictPlot[len(trainPredict)+(time_step*2)+1:len(residual_component_scaled)-1, :] =testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(residual_component_scaled))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

forecast_size = int((len(DF4) * 0.2)-5)
print(len(sma_forecast),len(y_pred),len(testPredict))
# Combine the forecasts for the test set
combined_forecast = sma_forecast[-forecast_size:].values * y_pred[-forecast_size:] * testPredict[-forecast_size:].reshape(-1)

# Plot the integrated results for the test set
plt.figure(figsize=(10, 6))
plt.plot(DF4.index, DF4['Adj Low'].values, label='Actual Data')
plt.plot(DF4.index[-forecast_size:], combined_forecast, label='Combined Forecast', color='red')
plt.legend()
plt.title('Integrated Forecast Comparison with Actual Data for 20% of Data')
plt.xlabel('Date')
plt.ylabel('Adjusted Low Price')
plt.show()

# Calculate error metrics for the combined forecast
actual = DF4['Adj Low'].iloc[-forecast_size:]
mae = mean_absolute_error(actual, combined_forecast)
rmse = np.sqrt(mean_squared_error(actual, combined_forecast))
mape = np.mean(np.abs((actual - combined_forecast) / actual)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")




