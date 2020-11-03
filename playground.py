import geopandas as gpd
import pandas as pd

red = gpd.read_file('genx/tiff/red.shp')
green = gpd.read_file('genx/tiff/green.shp')
blue = gpd.read_file('genx/tiff/blue.shp')

rgb = []
for i, j, k in zip(red.red.tolist(), green.green.tolist(), blue.blue.tolist()):
    rgb.append([i,j,k])

df = pd.DataFrame(rgb, columns=['red', 'green', 'blue'])
df1 = pd.DataFrame(rgb, columns=['red', 'green', 'blue'])
df1 = df1.drop_duplicates()


df['geometry'] = red.geometry

df_zero = df[(df.red == 0)&(df.green==0)&(df.blue==0)]

df = df.drop(df_zero.index)

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

lulc = []
for index, row in df1.iterrows():
    foo = dfcol[(dfcol.red == row.red)&(dfcol.green == row.green)&(dfcol.blue == row.blue)]
    if not foo.empty:
        lulc.append(foo.index[0])
    else:
        lulc.append(None)

df1['lulc'] = lulc

lulc = []
for index, row in df.iterrows():
    foo = df1[(df1.red == row.red)&(df1.green == row.green)&(df1.blue == row.blue)]
    if not foo.empty:
        lulc.append(foo.lulc.iloc[0])
    else:
        print(index)
        lulc.append(None)

df['lulc'] = lulc

gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.to_file('genx/tiff/lulc.shp')


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import elevation
import richdem as rd

dem_path = os.path.join(os.getcwd(), 'Shasta-30m-DEM.tif')
elevation.clip(bounds=(-122.4, 41.2, -122.1, 41.5), output=dem_path)

lulc = gpd.read_file('genx/tiff/lulc.shp')

lulc.drop(lulc[lulc.lulc == 'building'].index, inplace=True)
lulc.drop(lulc[lulc.lulc == 'forest1'].index, inplace=True)
lulc.drop(lulc[lulc.lulc == 'forest2'].index, inplace=True)
lulc.drop(lulc[lulc.lulc == 'forest3'].index, inplace=True)
lulc.drop(lulc[lulc.lulc == 'swamp'].index, inplace=True)
lulc.drop(lulc[lulc.lulc == 'water1'].index, inplace=True)
lulc.drop(lulc[lulc.lulc == 'water2'].index, inplace=True)
lulc.drop(lulc[lulc.lulc == 'snow'].index, inplace=True)

gdf = gpd.GeoDataFrame(lulc, geometry='geometry')
gdf.to_file('genx/tiff/lulc_permissible.shp')