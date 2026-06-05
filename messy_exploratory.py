# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
import scipy.fft as fft
import scipy.optimize as opt
column_names = {"Datetime", "O2 Concentration (%)", "Dissolved Oxygen (mg/L)",\
                "DO Sensor Temperature (C) ", "Well Salinity (PPT)", "Flood plain water level in BGS (cm)",\
                "WEATHER TIMESTAMP TS", "SlrFD_kW_Avg", "Precip (mm) over 5 minutes","AirT_C_Avg",\
                "BP_hPa", "SlrTF_MJ_Tot",\
                "WS_ms_S_WVT", "VP_hPa_Avg", "RH"}

data = pd.read_csv('ess_dive_9925a1a27161e64_20250522T191344828/data/2019_06_26_to_2024_09_30_Beaver_Creek_DO_saln_BGS_temp_weather.csv')

col = data.columns.values
col = col[col != 'Datetime']
col = col[col != 'WEATHER TIMESTAMP']
#col = col[col != 'Flood plain water level in BGS (cm)']
print(col)
'''
for k in col:
    print(k)
    l = 2
    x =data[k].values[l:]
    d = data.index.values[l:]
    lcut = 497387 - l
    ucut  = 501421 - l
    m = d.shape[-1]
    indices = list(range(0, lcut))+list(range(ucut, m))
    x = x[indices]
    d = d[indices]
    #x = x.delete(lcut, ucut)
    x = np.array(x, dtype='double')    
    plt.figure(k)
    conv_min = 5.0/(24*60*365)
    plt.plot(d*conv_min, x, label = k)
    plt.xlabel("years")
    plt.legend()
    plt.show()
'''

model = lr()
l = 2
d = data.index.values[l:]
lcut = 497387 - l
ucut  = 501421 - l
m = d.shape[-1]
indices = list(range(0, lcut))+list(range(ucut, m))
conv_min = 5.0/(24*60*365)
i = data.index.values[l:]*conv_min
O2 =data["O2 Concentration (%)"].values[l:]
Sal =data["Well Salinity (PPT)"].values[l:]
Rain =data["Precip (mm) over 5 minutes"].values[l:]
T =data["DO Sensor Temperature (C) "].values[l:]
i = i[indices]
O2 = O2[indices]
Sal = Sal[indices]
Rain = Rain[indices]
T = T[indices]
i = np.array(i[np.isnan(T) == False], dtype='double')
O2 = np.array(O2[np.isnan(T) == False], dtype='double')    
Sal = np.array(Sal[np.isnan(T) == False], dtype='double')    
Rain = np.array(Rain[np.isnan(T) == False], dtype='double')    
T = np.array(T[np.isnan(T) == False], dtype='double')
model.fit(T.reshape(-1, 1), Sal)
Sal_pred = model.predict(T.reshape(-1, 1))
ds = Sal - Sal_pred

plt.scatter(i, T, marker='o', s=10)
plt.ylabel("Temperature (C)")
plt.xlabel("years")
plt.show()

plt.scatter(i, Sal, marker='o', s=10)
plt.plot(i, Sal_pred, c = 'g')
plt.plot(i, ds, c = 'r')
plt.xlabel("years")
plt.ylabel("Salinity (PPT)")
plt.show()


plt.scatter(T, Sal, marker='o', s=10)
plt.plot(T, Sal_pred, c = 'r')
plt.xlabel("Temperature (C)")
plt.ylabel("Salinity (PPT)")
plt.show()


plt.figure()
plt.scatter(ds[O2 != 0.0], Rain[O2 != 0.0], c=O2[O2 != 0.0], cmap='coolwarm', marker='o', s=10)
plt.yscale('log')
plt.ylabel("Precip (mm) over 5 minutes")
plt.xlabel("Salinity - Seasonal Changes (PPT)")
plt.colorbar(label='O2 Concentration (%)')
plt.show()

plt.figure()
plt.scatter(Sal[O2 != 0.0], Rain[O2 != 0.0], c=O2[O2 != 0.0], cmap='coolwarm', marker='o', s=10)
plt.yscale('log')
plt.ylabel("Precip (mm) over 5 minutes")
plt.xlabel("Salinity")
plt.colorbar(label='O2 Concentration (%)')
plt.show()

plt.scatter(i[O2 != 0.0], O2[O2 != 0.0], marker='o', s=10)
plt.scatter(i[O2 != 0.0], ds[O2 != 0.0], marker='o', s=10)
plt.scatter(i[O2 != 0.0], Sal[O2 != 0.0], marker='o', s=10)
plt.scatter(i[O2 != 0.0], Rain[O2 != 0.0], marker='o', s=10)
plt.ylabel("O2 Concentration (%)")
plt.xlabel("years")
plt.xlim([0, 1])
#plt.ylim([0, 0.6])
plt.show()

k = Sal.shape[0]
print(k)
xf = fft.fftfreq(k, 5.0)
Sal_ff = fft.fft(Sal)
T_ff = fft.fft(T)
plt.scatter(xf, Sal_ff, marker='o', s=10)
plt.ylabel("Fequencies of Salinity")
plt.xlabel("1/min")
#plt.yscale('log')
plt.xlim([0, 0.01])
plt.ylim([0, 5000])
plt.show()

plt.scatter(xf, T_ff, marker='o', s=10)
plt.ylabel("Fequencies of Temperature")
plt.xlabel("1/min")
#plt.yscale('log')
plt.xlim([0, 0.01])
plt.ylim([0, 1000])
plt.show()

def f_sine(x, a, b, c, d):
    return a*np.sin(2*np.pi*(x + d)/b) + c

Sal_sin = opt.curve_fit(f_sine, i, Sal, p0 = [10, 1, 10, 0])
print(Sal_sin)


plt.scatter(i, Sal, marker='o', s=10)
plt.plot(i, f_sine(i, Sal_sin[0][0], Sal_sin[0][1], Sal_sin[0][2], Sal_sin[0][3]), c = 'g')
plt.xlabel("years")
plt.ylabel("Salinity (PPT)")
plt.show()
ds_fit = Sal - f_sine(i, Sal_sin[0][0], Sal_sin[0][1], Sal_sin[0][2], Sal_sin[0][3])
plt.scatter(i, ds_fit, marker='o', s=10)
plt.xlabel("years")
plt.ylabel("Salinity (PPT)")
plt.show()


ds_ff = fft.fft(ds_fit)
plt.scatter(xf, ds_ff, marker='o', s=10)
plt.ylabel("Fequencies of Salinity")
plt.xlabel("1/min")
plt.xlim([0, 0.01])
#plt.ylim([0, 5000])
plt.show()