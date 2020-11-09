import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
from matplotlib import colors

tech = input('Input renewable energy technology: ')
path = 'hr_sampling/'+tech+'/'
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
			cf_df = pd.read_csv(path+file[:-4]+'_'+str(i)+'.csv', index_col=0)
			mean_cf = cf_df.mean()
			df_mean = pd.DataFrame(zip(file_csv.latitude[500*i:500*i+499].tolist(), file_csv.longitude[500*i:500*i+499].tolist(), mean_cf.tolist()), columns=['lat','lon','cf_mean'])
			df_mean_cf = df_mean_cf.append(df_mean, ignore_index=True)
	else:
		rangeval = int(len(file_csv)/500)+1
		if rangeval > 1:
			for i in range(rangeval):
				if i < rangeval-1:
					cf_df = pd.read_csv(path+file[:-4]+'_'+str(i)+'.csv', index_col=0)
					mean_cf = cf_df.mean()
					df_mean = pd.DataFrame(zip(file_csv.latitude[500*i:500*i+499].tolist(), file_csv.longitude[500*i:500*i+499].tolist(), mean_cf.tolist()), columns=['lat','lon','cf_mean'])
					
				else:
					cf_df = pd.read_csv(path+file[:-4]+'_'+str(i)+'.csv', index_col=0)
					mean_cf = cf_df.mean()
					df_mean = pd.DataFrame(zip(file_csv.latitude[500*i:].tolist(), file_csv.longitude[500*i:].tolist(), mean_cf.tolist()), columns=['lat','lon','cf_mean'])
				df_mean_cf = df_mean_cf.append(df_mean, ignore_index=True)
		else:
			cf_df = pd.read_csv(path+file[:-4]+'_0.csv', index_col=0)
			mean_cf = cf_df.mean()
			df_mean = pd.DataFrame(zip(file_csv.latitude.tolist(), file_csv.longitude.tolist(), mean_cf.tolist()), columns=['lat','lon','cf_mean'])
			df_mean_cf = df_mean_cf.append(df_mean, ignore_index=True)

	df_mean_cf['gid'] = file_csv.gid
		
	geometry = []
	for index, row in df_mean_cf.iterrows():
		geometry.append(Point(row.lon, row.lat))
	df_mean_cf['geometry'] = geometry

	df = df.append(df_mean_cf, ignore_index=True)

gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.to_file(path+tech+'_cf.shp')

# build voronoi

input('voronoi polygon built? (y/n): ')

india = gpd.read_file('genx/tiff/india.shp')
voronoi = gpd.read_file(path+tech+'_voronoi.shp')

polygon = gpd.clip(voronoi, india)


for index, row in polygon.iterrows():
	if type(row.geometry) != Polygon:
		print('collection ', index)
		for i in row.geometry:
			if type(i) == Polygon:
				polygon.at[index, 'geometry'] == i

polygon.to_file(path+tech+'_plot.shp')

plot = gpd.read_file(path+tech+'_plot.shp')
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