import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import pyproj
from shapely.geometry import Polygon, Point

path = 'D:/india-dem/SLP/'

slp_dir = os.listdir(path)

for slp in slp_dir:
    src = rio.open(path+slp)


india = gpd.read_file('genx/tiff/india.shp')


xrange = np.arange(india.bounds.minx[0], india.bounds.maxx[0], 0.05)
yrange = np.arange(india.bounds.miny[0], india.bounds.maxy[0], 0.05)

cnt = 0
polygon = []
for x in range(len(xrange)-1):
    for y in range(len(yrange)-1):
        polygon.append(Polygon([[xrange[x], yrange[y]],[xrange[x+1], yrange[y]],[xrange[x+1], yrange[y+1]],[xrange[x], yrange[y+1]],[xrange[x], yrange[y]]]))
        print(cnt/(len(xrange)*len(yrange))*100)
        cnt += 1

df = pd.DataFrame(polygon, columns=['geometry'])

gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.to_file('genx/tiff/india_grid.shp')

slope = pd.read_csv('file_coords.csv')

gdf = gpd.read_file('genx/tiff/indiagrid.shp')

cnt = 1
geometry_2 = []
for geom in gdf.geometry:

    points = [Point(geom.bounds[0], geom.bounds[1]),
        Point(geom.bounds[2], geom.bounds[1]),
        Point(geom.bounds[2], geom.bounds[3]),
        Point(geom.bounds[0], geom.bounds[3])]

    slope_files = []
    for point in points:
        slope_file = slope[(slope.Left<=point.x)&(slope.Right>=point.x)&(slope.Top>=point.y)&(slope.Bottom<=point.y)]
        if slope_file.file.iloc[0] not in slope_files:
            slope_files.append(slope_file.file.iloc[0])

    if len(slope_files) == 1:
        s = rio.open('D:/india-dem/SLP/'+slope_file.file.iloc[0])
        
        proj_points = []
        for point in points:
            wgs84=pyproj.CRS("EPSG:4326") 
            src_crs = pyproj.CRS(str(s.crs))

            xx, yy = pyproj.transform(wgs84, src_crs, point.y, point.x)
            proj_points.append([xx, yy])
        geometry_2.append(Polygon(proj_points))
    else:
        geometry_2.append(np.nan)
    print(cnt, cnt/(len(gdf)*100))
    cnt+=1
gdf['geometry_2'] = geometry_2
