### DEVELOPED BY : NIVETHA A
### REGISTER NO : 212222230101
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('MentalHealthSurvey.csv')
data['Time'] = pd.date_range(start='2023-01-01', periods=len(data), freq='M')
data.set_index('Time', inplace=True)
time_series = data['anxiety']
plt.figure(figsize=(10, 6))
plt.plot(time_series)
plt.title('Anxiety Time Series Plot')
plt.xlabel('Time')
plt.ylabel('Anxiety Level')
plt.show()
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    if result[1] < 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is not stationary.")

test_stationarity(time_series)
time_series_diff = time_series.diff().dropna()
print("\nAfter Differencing:")
test_stationarity(time_series_diff)
# Set initial SARIMA parameters (p, d, q, P, D, Q, m)
p, d, q = 1, 1, 1
P, D, Q, m = 1, 1, 1, 12  # m = 12 for monthly seasonality if applicable
model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(P, D, Q, m), enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = model.fit(disp=False)
print(sarima_fit.summary())
forecast_steps = 12  # Number of periods to forecast
forecast = sarima_fit.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Historical Data')
plt.plot(forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMA Forecast of Anxiety Levels')
plt.xlabel('Time')
plt.ylabel('Anxiety Level')
plt.legend()
plt.show()
from sklearn.metrics import mean_absolute_error
test_data = time_series[-forecast_steps:]
pred_data = forecast.predicted_mean[:len(test_data)]
mae = mean_absolute_error(test_data, pred_data)
print('Mean Absolute Error:', mae)

```

### OUTPUT:

![383801391-710cc018-2083-4b5e-9ded-f67ce45f5136](https://github.com/user-attachments/assets/ecbe436d-e4ef-4d5a-beaf-f420f343e08a)
![383801473-4e4afb4c-c738-4125-95e8-787cdd7c1258](https://github.com/user-attachments/assets/cf346a5d-8076-4a39-bb0e-c5b2252e888e)
![383801555-a311cfc8-079f-4e8f-a54f-009f3f83cf3e](https://github.com/user-attachments/assets/f13a7e48-342f-43b7-9e1d-98107db6bc60)



### RESULT:
Thus the program run successfully based on the SARIMA model.
