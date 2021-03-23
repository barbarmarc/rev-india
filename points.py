#%% Step 1
from rex import Resource


SOLAR = "/nrel/nsrdb/india/nsrdb_india_2000.h5"
WIND = "/nrel/wtk/india/wtk_india_2014.h5"

with Resource(WIND, hsds=True) as file:
    points = file.meta

points.index.name = "gid"
points["config"] = "default"

points.to_csv("points_projects.csv")

#%% Step 2
# select by location in qgis

#%% sample points
from rasterstats import zonal_stats, point_query
import geopandas as gpd
path = 'genx/wind_shp/'

regions = ['NR', 'SR', 'WR', 'ER', 'NER']

for r in regions:
    gdf = gpd.read_file(path+r+'_points.shp')
    gdf['wind'] = point_query(gdf.geometry.tolist(), 'genx/IND_wind-speed_100m.tif')

    gdf = gpd.GeoDataFrame(gdf, crs='epsg:4326', geometry = gdf.geometry)
    gdf.to_file(path+r+'_wind_points.shp')

#%% Step 3
import geopandas as gpd
path = 'genx/wind_shp/'

regions = ['NR', 'SR', 'WR', 'ER', 'NER']

for r in regions:
    gdf = gpd.read_file(path+r+'_wind_points.shp')

    reduced_gdf = gdf[gdf.wind >= 4]

    df = reduced_gdf[['gid', 'latitude', 'longitude', 'elevation', 'offshore', 'wrf_region', 'config']]

    df.to_csv('genx/points/wind/'+r+'_points.csv', index=False)

#%% Step 4
import pandas as pd
path = 'genx/points/wind/'

regions = ['NR', 'SR', 'WR', 'ER', 'NER']

for r in regions:
    df = pd.read_csv(path+r+'_points.csv')
    reduced_df = df.sample(int(df.size/10))

    reduced_df.to_csv(path+r+'_reduced_points.csv')