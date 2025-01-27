import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set_style('darkgrid')

from statsmodels.tsa.arima_process import ArmaProcess, arma_acovf, arma_acf, arma_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acovf, acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

#import pmdarima as pm

data_1 = pd.read_csv('powerconsumption.csv', usecols=['PowerConsumption_Zone1'], nrows=4465)#nrows=4465)
data = data_1.iloc[::5] 
n = len(data)
x_axis = np.linspace(1, n, n)
plt.plot(x_axis, data)
#plt.show()

decomposition = seasonal_decompose(data, model='multiplicative', period=2, extrapolate_trend=2)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.plot(data, label='Original')
plt.title('Surowe dane')

plt.subplot(2, 2, 2)
plt.plot(trend, label='Trend', color='red')
plt.title('Trend')

plt.subplot(2, 2, 3)
plt.plot(seasonal, label='Seasonal', color='green', linewidth=0.2)
plt.title('Sezonowość')

plt.subplot(2, 2, 4)
plt.plot(residual, label='Residual', color='orange')
plt.title('Dane po usunięciu trendu i sezonowości')

plt.tight_layout()
plt.show()



stl_decomposition = STL(data, period=30).fit()

trend_stl = stl_decomposition.trend
seasonal_stl = stl_decomposition.seasonal
residual_stl = stl_decomposition.resid

plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.plot(data, label='Original')
plt.title('Surowe dane')

plt.subplot(2, 2, 2)
plt.plot(trend_stl, label='Trend', color='red')
plt.title('Trend')

plt.subplot(2, 2, 3)
plt.plot(seasonal_stl, label='Seasonal', color='green')
plt.title('Sezonowość')

plt.subplot(2, 2, 4)
plt.plot(residual_stl, label='Residual', color='orange')
plt.title('Dane po usunięciu trendu i sezonowości')

plt.tight_layout()
plt.show()

time = np.arange(len(data))

linear_reg = LinearRegression()
linear_reg.fit(time.reshape(-1, 1), data)
trend_predicted = linear_reg.predict(time.reshape(-1, 1))

arma_trajectory_without_linear_trend = data - trend_predicted

periodogram = np.abs(np.fft.fft(arma_trajectory_without_linear_trend))**2 / len(arma_trajectory_without_linear_trend)
frequencies = np.fft.fftfreq(len(arma_trajectory_without_linear_trend))

plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(arma_trajectory_without_linear_trend) // 2], periodogram[:len(arma_trajectory_without_linear_trend) // 2])
plt.title('Periodogram')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.grid(True)
plt.show()

peak_indices = np.argsort(periodogram)[::-1][:6] 
peaks_frequency = frequencies[peak_indices]
peaks_period = 1 / peaks_frequency

