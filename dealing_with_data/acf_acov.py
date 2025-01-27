from statsmodels.tsa.stattools import acovf, acf, pacf, adfuller
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess, arma_acovf, arma_acf, arma_pacf


data = pd.read_csv('dealing_with_data\powerconsumption.csv')
data = data[:4465]

  
# drop function which is used in removing or deleting rows or columns from the CSV files 



data_train = data["PowerConsumption_Zone1"]

stl_decomposition = STL(data_train, period = 144).fit()

trend_stl = stl_decomposition.trend
seasonal_stl = stl_decomposition.seasonal
residual_stl = stl_decomposition.resid

periodogram = np.abs(np.fft.fft(seasonal_stl))**2 / len(seasonal_stl)
frequencies = np.fft.fftfreq(len(seasonal_stl))

plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(seasonal_stl) // 2], periodogram[:len(seasonal_stl) // 2])
plt.title('Periodogram')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.grid(True)
plt.show()

peak_indices = np.argsort(periodogram)[::-1][:10] 
peaks_frequency = frequencies[peak_indices]
peaks_period = 1 / peaks_frequency

print("Top 5 Peaks (Frequency, Period):")
for i in range(len(peaks_frequency)):
    print(f"Peak {i+1}: {peaks_frequency[i]:.4f}, {peaks_period[i]:.2f}")



plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.plot(data_train, label='Original')
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

def adfuller_test(series, sig = 0.05):
    res = adfuller(series, autolag='AIC')    
    p_value = round(res[1], 3) 
    stats   = round(res[0], 3) 

    if p_value <= sig:
        print(f"Statystyka testowa = {stats}, p-Value = {p_value} => Stationary. ")
    else:
        print(f"Statystyka testowa = {stats}, p-value = {p_value} => Non-stationary.")

adfuller_test(residual_stl)
adfuller_test(residual_stl + seasonal_stl)
adfuller_test(residual_stl + seasonal_stl + trend_stl)
adfuller_test(residual_stl + trend_stl)

plt.figure(figsize=(10, 3))
 

plt.subplot(2, 1, 1)
h_max = 50
ar_coef = np.array([1, 0.9520])
autokow_teor = arma_acovf(ar_coef, np.array([1]), nobs = h_max, sigma2=6.484e+04)
autokow_emp = acovf(residual_stl, demean=False, fft=False, nlag=h_max)

plt.stem(autokow_emp, basefmt='')
plt.plot(autokow_teor, '-', label='Teoretyczna ACVF')
plt.xlabel('Opóźnienie')
plt.ylabel('Autokowariancja')
plt.title('Autokowariancja danych po dekompozycji')
#plt.legend()

plt.subplot(2, 1, 2)



autokow_teor_t = arma_acf(ar_coef, np.array([1]), lags = h_max)
autokor_emp = acf(residual_stl, fft=False, nlags=h_max)

plt.stem(autokor_emp, basefmt='')
plt.plot(autokow_teor_t, '-', label='Teoretyczna ACVF')
plt.xlabel('Opóźnienie')
plt.ylabel('Autokorelacja')
plt.title('Autokorealcja danych po dekompozycji')
#plt.legend()

plt.tight_layout(pad=0.05)
plt.show()


model = pm.auto_arima(residual_stl, 
                       d=0,
                       start_p=0,
                       start_q=0,
                       max_d=0,
                       max_p=5, 
                       max_q=5, 
                       max_order=None,
                       trace=True, 
                       seasonal=False, 
                       stepwise = True)

print(model.summary())
model.plot_diagnostics(figsize=(10, 8))
plt.tight_layout()
plt.show()