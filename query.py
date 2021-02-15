import pandas as pd
import geopandas as gpd
from functools import partial
import pyproj
from shapely.ops import transform
from shapely.geometry import Point, Polygon

proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')

lulc = gpd.read_file('supplycurve/lulc_tech_tr_lcoe.shp')
points = pd.read_csv('query/Location_Dharik.csv')

wind_cf = pd.read_csv('hr_sampling/wind/wind_cf.csv')
solar_cf = pd.read_csv('hr_sampling/solar/solar_cf.csv')

def geodesic_point_buffer(lat, lon):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(50000)
    return transform(project, buf).exterior.coords[:]

# Example
polies = []
for index, row in points.iterrows():
    pointList = geodesic_point_buffer(float(row['Latitude']), float(row['Longitude']))
    polies.append(Polygon(pointList))

df = pd.DataFrame(polies, columns=['geometry'])
gdf = gpd.GeoDataFrame(df, geometry='geometry')


lulc['geometry'] = lulc['geometry'].centroid

points_within = gpd.sjoin(lulc, gdf, op='within')

interconnection, solar_potential, wind_potential = [], [], []
solarDF = pd.DataFrame()
windDF = pd.DataFrame()

for i in range(len(gdf)):
    pointList = points_within[points_within.index_right == i]

    interconnection.append(pointList.IX_MW.mean())
    solar_potential.append(pointList.mw_solar.sum())
    wind_potential.append(pointList.mw_wind.sum())

    windDF[i] = wind_cf[pointList.gid_wind.tolist()].mean(axis=1)
    solarDF[i] = solar_cf[pointList.gid_solar.tolist()].mean(axis=1)

points['IXMW'] = interconnection
points['SolarMW'] = solar_potential
points['WindMW'] = wind_potential

points.to_csv('query/Location_Dharik.csv')
windDF.to_csv('query/windCF.csv')
solarDF.to_csv('query/solarCF.csv')
