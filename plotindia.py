import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import cKDTree

tech = input('Input renewable energy technology: ')
path = 'supplycurve/hr_sampling/'+tech+'/'
files = os.listdir(path)
file_df = pd.DataFrame(files, columns=['files'])

df = pd.DataFrame(columns=['lat', 'lon', 'cf_mean', 'gid', 'geometry'])

for region in ['ER', 'NER', 'NR', 'SR', 'WR']:

	region_df = file_df[file_df.files.str.contains(region)]

	if region == 'ER':
		region_df = region_df[~region_df.files.str.contains('NER')]

	file = region+'_points.csv'
	region_df = region_df[~region_df.files.str.contains(file)]

	file_csv = pd.read_csv(path+file)

	df_mean_cf = pd.DataFrame(columns=['lat','lon','cf_mean'])

	if len(file_csv) == 2000:
		for i in range(4):
			f_csv = file_csv.loc[500*i:500*i+499]
			f_csv = f_csv.sort_values('gid')
			cf_df = pd.read_csv(path+file[:-4]+'_'+str(i)+'.csv', index_col=0)
			mean_cf = cf_df.mean()
			df_mean = pd.DataFrame(zip(f_csv.latitude.tolist(), f_csv.longitude.tolist(), mean_cf.tolist()), columns=['lat','lon','cf_mean'])
			df_mean_cf = df_mean_cf.append(df_mean, ignore_index=True)
	else:
		rangeval = int(len(file_csv)/500)+1
		if rangeval > 1:
			for i in range(rangeval):
				if i < rangeval-1:
					f_csv = file_csv.loc[500*i:500*i+499]
					f_csv = f_csv.sort_values('gid')
					cf_df = pd.read_csv(path+file[:-4]+'_'+str(i)+'.csv', index_col=0)
					mean_cf = cf_df.mean()
					df_mean = pd.DataFrame(zip(f_csv.latitude.tolist(), f_csv.longitude.tolist(), mean_cf.tolist()), columns=['lat','lon','cf_mean'])
					
				else:
					f_csv = file_csv.loc[500*i:]
					f_csv = f_csv.sort_values('gid')
					cf_df = pd.read_csv(path+file[:-4]+'_'+str(i)+'.csv', index_col=0)
					mean_cf = cf_df.mean()
					df_mean = pd.DataFrame(zip(f_csv.latitude.tolist(), f_csv.longitude.tolist(), mean_cf.tolist()), columns=['lat','lon','cf_mean'])
				df_mean_cf = df_mean_cf.append(df_mean, ignore_index=True)
		else:
			f_csv = file_csv
			f_csv = f_csv.sort_values('gid')
			cf_df = pd.read_csv(path+file[:-4]+'_0.csv', index_col=0)
			mean_cf = cf_df.mean()
			df_mean = pd.DataFrame(zip(f_csv.latitude.tolist(), f_csv.longitude.tolist(), mean_cf.tolist()), columns=['lat','lon','cf_mean'])
			df_mean_cf = df_mean_cf.append(df_mean, ignore_index=True)

	df_mean_cf['gid'] = file_csv.gid
		
	geometry = []
	for index, row in df_mean_cf.iterrows():
		geometry.append(Point(row.lon, row.lat))
	df_mean_cf['geometry'] = geometry

	df = df.append(df_mean_cf, ignore_index=True)

gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.to_file(path+tech+'_cf.shp')

def ckdnearest(gdA, gdB):
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdf = pd.concat(
        [gdA.reset_index(drop=True), gdB.loc[idx, gdB.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf

india = gpd.read_file('supplycurve/india/indiagrid.shp')
tech_cf = gdf

centroids = []
for g in india.geometry:
	centroids.append(g.centroid)
india['centroid'] = centroids
india = india.rename(columns={"geometry": "polygon", "centroid": "geometry"})

india = ckdnearest(india,tech_cf)

india = india.rename(columns={"geometry": "centroid","polygon": "geometry"})
del india['lat']
del india['lon']
del india['gid']
del india['dist']
del india['centroid']

india.to_file(path+tech+'_plot.shp')

plot = india
norm = colors.Normalize(vmin=plot.cf_mean.min(), vmax=plot.cf_mean.max())
cbar = plt.cm.ScalarMappable(norm=norm, cmap='viridis_r')

# plot
fig, ax = plt.subplots(1,1)
plot.plot(column='cf_mean', cmap='viridis_r', legend=False, ax=ax) 
x_axis = ax.axes.get_xaxis()
x_axis.set_visible(False)
y_axis = ax.axes.get_yaxis()
y_axis.set_visible(False)

# add colorbar
ax_cbar = fig.colorbar(cbar, ax=ax)
ax_cbar.set_label(label='Mean Capacity Factor', size='xx-large')
ax_cbar.ax.tick_params(labelsize=16)
plt.show()