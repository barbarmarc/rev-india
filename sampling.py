import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from rasterstats import zonal_stats, point_query
import random
import pyproj

lulc = gpd.read_file('genx/tiff/india_lulc.shp')
regions = gpd.read_file('genx/tiff/regions.shp')
slope = pd.read_csv('asf/file_coords.csv')

tech = 'solar'
pointtype = '/WR_points.csv'
points = pd.read_csv('genx/points/'+tech+pointtype)
lulc = rasterio.open('genx/tiff/india.tif')

colors = [[255,0,0],
[255, 209,0],
[255,158,0],
[107,120,31],
[0,94,0],
[0,204,0],
[245,245,219],
[94,209,242],
[209,181,133],
[0,158,222],
[199,191,191],
[255,191,196],
[184,235,120],
[158,31,235],
[115,184,43],
[5,130,94],
[158,207,31],
[158,82,43]]

atr = ['building', 'crop1', 'crop2', 'forest1', 'forest2', 'plantation', 'crop3','water1','wasteland','water2','rann','snow','grassland','cultivation','forest3','swamp','crop4','crop5']

dfcol = pd.DataFrame(colors, index=atr, columns=['red','green','blue'])

def read_rstr(dataset, point, dfcol):
    band1 = dataset.read(1)
    band2 = dataset.read(2)
    band3 = dataset.read(3)

    coords = dataset.index(point.x, point.y)

    r = band1[coords[0],coords[1]]
    g = band2[coords[0],coords[1]]
    b = band3[coords[0],coords[1]]

    pixel = dfcol[(dfcol.red==r)&(dfcol.green==g)&(dfcol.blue==b)]
    if not pixel.empty:
        if pixel.index[0] in ['crop1', 'crop2', 'plantation', 'crop3','wasteland','rann','grassland','cultivation','crop4','crop5']:
            return True
        else:
            return False
    else:
        return False

def sample(number, lulc, points, tech):
    print(tech)
    print(pointtype)
    pointList = []
    i = 0
    chosen = 0
    while i < number:
        index = random.randint(0,len(points)-1)
        if index not in pointList:
            point = Point(points.loc[index]['longitude'], points.loc[index]['latitude'])

            slope_file = slope[(slope.Left<=point.x)&(slope.Right>=point.x)&(slope.Top>=point.y)&(slope.Bottom<=point.y)]
            slope_file.file.iloc[0]

            slp = rasterio.open('asf/SLP/'+slope_file.file.iloc[0])

            wgs84=pyproj.CRS("EPSG:4326") 
            src_crs = pyproj.CRS(str(slp.crs))

            xx, yy = pyproj.transform(wgs84, src_crs, point.y, point.x)
            proj_point = Point(xx, yy)

            slp_val = point_query([proj_point], 'asf/SLP/'+slope_file.file.iloc[0])

            if slp_val[0] <= 5 and tech == 'wind':
                if read_rstr(lulc, point, dfcol):
                    ws = point_query([point], 'genx/tiff/IND_wind-speed_100m.tif')
                    if ws[0] > 4:
                        pointList.append(index)
                        i += 1
            elif slp_val[0] <= 14:
                if read_rstr(lulc, point, dfcol):
                    pointList.append(index)
                    i += 1

        chosen += 1

        if chosen == len(points):
            break

    result_points = points.loc[pointList]

    result_points.to_csv('genx/points/hr_sampling/'+tech+pointtype, index=False)