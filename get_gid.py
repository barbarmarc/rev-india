from rex import Resource
import pandas as pd
import pyproj

SOLAR = "/nrel/nsrdb/india/nsrdb_india_2000.h5"
WIND = "/nrel/wtk/india/wtk_india_2014.h5"

points = pd.read_excel('query/Loc_Ind_Wind_Run.xlsx')

with Resource(WIND, hsds=True) as file:
    wind = file.meta

wind.index.name = "gid"
wind["config"] = "default"

with Resource(SOLAR, hsds=True) as file:
    solar = file.meta

solar.index.name = "gid"
solar["config"] = "default"

import pandas as pd
import geopandas as gpd
import pyproj
from functools import partial
from shapely.ops import transform
from shapely.geometry import Point, Polygon
proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')

wind = pd.read_csv('wind_gid.csv')

geometry = [Point(xy) for xy in zip(wind.longitude, wind.latitude)]
wind_gdf = gpd.GeoDataFrame(wind, crs="EPSG:4326", geometry=geometry)

geometry = [Point(xy) for xy in zip(solar.longitude, solar.latitude)]
solar_gdf = gpd.GeoDataFrame(solar, crs="EPSG:4326", geometry=geometry)

def geodesic_point_buffer(lat, lon):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(50000)
    return transform(project, buf).exterior.coords[:]

polies = []
for index, row in points.iterrows():
    pointList = geodesic_point_buffer(float(row['Latitude']), float(row['Longitude']))
    polies.append(Polygon(pointList))

df = pd.DataFrame(polies, columns=['geometry'])
gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry='geometry')

solar_considered = gpd.sjoin(solar_gdf, gdf, op='within')
wind_considered = gpd.sjoin(wind_gdf, gdf, op='within')

solar_considered.to_csv('solar_points_full.csv', index=False)
wind_considered.to_csv('wind_points_full.csv', index=False)

solar_considered = solar_considered.reset_index()
solar_considered = solar_considered.drop_duplicates('gid')

# optional
solar_considered['tilt'] = solar_considered.latitude.round()
unique_tilt = solar_considered.tilt.unique().tolist()
for tilt in unique_tilt:
    solar_tilt = solar_considered[solar_considered.tilt == tilt]
    print(len(solar_tilt))
    solar_tilt.to_csv('solar_points'+str(int(tilt))+'.csv', index=False)


wind_considered = wind_considered.drop_duplicates('gid')
wind_considered = wind_considered.sample(7000)

solar_considered.to_csv('solar_points.csv', index=False)
wind_considered.to_csv('wind_points.csv', index=False)


