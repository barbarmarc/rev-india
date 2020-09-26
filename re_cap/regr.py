import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pd.read_excel("Installed_RE_Capacity_China.xlsx", index_col=0)
solar_cagr = df['Solar CAGR'].dropna()
wind_cagr = df['Wind CAGR'].dropna()

def func(x, a, c):
    return np.log((a + x)**2 / (x - c)**2)

def regr(x, y):

	popt, _ = curve_fit(func, x, y)

	R2 = np.sum((func(x, *popt) - y.mean())**2) / np.sum((y - y.mean())**2)
	print("r^2 = %10.6f" % R2)

	print(popt)

	plt.figure()
	plt.plot(x, y, 'ko', label="Original Data")
	plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
	plt.legend()
	plt.show()

	return popt

x = solar_cagr.index.to_numpy()
y = solar_cagr.to_numpy()
solar_popt = regr(x, y)

x = wind_cagr.index.to_numpy()
y = wind_cagr.to_numpy()
wind_popt = regr(x, y)

wind_energy_india_2019 = 37.505
solar_energy_india_2019 = 33.731

df_india = pd.DataFrame(index=range(2019,2051))
wind_growth = []
for i in range(2020, 2051):
	if i <= 2025:
		wind_growth.append(func(2025, *wind_popt))
	else:
		wind_growth.append(func(i, *wind_popt))
wind_india = [wind_energy_india_2019]
for i in range(len(wind_growth)):
	wind_india.append((1+wind_growth[i])*wind_india[i])
df_india['wind_gw'] = wind_india

solar_growth = []
for i in range(2020, 2051):
	if i <= 2025:
		solar_growth.append(func(2025, *solar_popt))
	else:
		solar_growth.append(func(i, *solar_popt))
solar_india = [solar_energy_india_2019]
for i in range(len(solar_growth)):
	solar_india.append((1+solar_growth[i])*solar_india[i])
df_india['solar_gw'] = solar_india

df_india.to_csv("India_Renewables_Forecast.csv")