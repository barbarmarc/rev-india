import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pd.read_excel("Installed_RE_Capacity_China.xlsx", index_col=0)

solar = df['China Solar PV SP (GW)']
wind = df['China Wind energy SP (GW)']

def guess_lag(x, y):
    """Given two axes returns a guess of the lag point.

    The lag point is defined as the x point where the difference in y
    with the next point is higher then the mean differences between
    the points plus one standard deviation. If such point is not found
    or x and y have different lengths the function returns zero.
    """
    if len(x) != len(y):
        return 0

    diffs = []
    indexes = range(len(x))

    for i in indexes:
        if i + 1 not in indexes:
            continue
        diffs.append(y[i + 1] - y[i])
    diffs = np.array(diffs)

    flex = x[-1]
    for i in indexes:
        if i + 1 not in indexes:
            continue
        if (y[i + 1] - y[i]) > (diffs.mean() + (diffs.std())):
            flex = x[i]
            break

    return flex


def guess_plateau(x, y):
    """Given two axes returns a guess of the plateau point.

    The plateau point is defined as the x point where the y point
    is near one standard deviation of the differences between the y points to
    the maximum y value. If such point is not found or x and y have
    different lengths the function returns zero.
    """
    if len(x) != len(y):
        return 0

    diffs = []
    indexes = range(len(y))

    for i in indexes:
        if i + 1 not in indexes:
            continue
        diffs.append(y[i + 1] - y[i])
    diffs = np.array(diffs)

    ymax = y[-1]
    for i in indexes:
        if y[i] > (ymax - diffs.std()) and y[i] < (ymax + diffs.std()):
            ymax = y[i]
            break

    return ymax

def gompertz(x, A, u, d, v, y0):
    """Gompertz growth model.

    Proposed in Zwietering et al., 1990 (PMID: 16348228)
    """
    y = (A * np.exp(-np.exp((((u * np.e) / A) * (d - x)) + 1))) + y0

    return y

def fit(function, x, y):
    """Fit the provided function to the x and y values.

    The function parameters and the parameters covariance.
    """
    # Compute guesses for the parameters
    # This is necessary to get significant fits
    p0 = [guess_plateau(x, y), 4.0, guess_lag(x, y), 0.1, min(y)]

    popt, pcov = curve_fit(function, x, y, p0=p0)

    return popt, pcov

def pplot(function, popt, pcov, title):

	R2 = np.sum((function(x, *popt) - y.mean())**2) / np.sum((y - y.mean())**2)
	print(title+" R^2 = %10.6f" % R2)

	plt.figure()
	plt.plot(x, y, 'ko', label="Original Data")
	plt.plot(x, function(x, *popt), 'r-', label="Fitted Curve")
	plt.plot(list(range(2001, 2101)), function(list(range(2001,2101)), *popt), 'y--', label="Full Curve")
	plt.title(title+ " Trend in China under IEA WEO 2018 Sustainable Policy Scenario Projections")
	plt.legend()
	plt.show()

	return R2

x = solar.index.to_numpy()
y = solar.to_numpy()
solar_popt, solar_pcov = fit(gompertz, x, y)
solar_r2 = pplot(gompertz, solar_popt, solar_pcov, "Solar")

x = wind.index.to_numpy()
y = wind.to_numpy()
wind_popt, wind_pcov = fit(gompertz, x, y)
wind_r2 = pplot(gompertz, wind_popt, wind_pcov, "Wind")

wind_energy_india_2019 = 37.505
solar_mapping_year = 2014
solar_energy_india_2019 = 33.731
wind_mapping_year = 2010

df_india = pd.DataFrame(index=range(2019,2051))
wind_india = [wind_energy_india_2019]
for i in range(wind_mapping_year, (2050-2019+wind_mapping_year)):
	wind_india.append(gompertz(i, *wind_popt))
df_india['wind_gw'] = wind_india

solar_india = [solar_energy_india_2019]
for i in range(solar_mapping_year, (2050-2019+solar_mapping_year)):
	solar_india.append(gompertz(i, *solar_popt))
df_india['solar_gw'] = solar_india

df_india.plot(title="Solar and Wind Forecast for India using China's Gompertz growth curve")
plt.show()

df_india.to_csv("India_Renewables_Forecast.csv")