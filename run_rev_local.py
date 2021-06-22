#%% packages
import os
import pandas as pd
import numpy as np
#import geopandas as gpd
from reV.config.project_points import ProjectPoints, PointsControl
from reV.generation.generation import Gen
from rex import init_logger
init_logger("reV.generation", log_level="DEBUG", log_file="./rev.log")
init_logger("reV.config", log_level="DEBUG", log_file="./rev.log")
from create_json import *
from os import walk

#%% Path
SOLAR = "/nrel/nsrdb/india/nsrdb_india_2000.h5"
WIND = "/nrel/wtk/india/wtk_india_2014.h5"

renewable_choice = input("Enter renewable resource: ")

if renewable_choice == 'solar':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'solar_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_solar(points_df.timezone.unique()[0], 0, 0.5)
	write_jsonfile(sam_config, 'solar')
	sam_file = os.path.expanduser('json_files/sam_config_solar.json')
elif renewable_choice == 'solar1':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'solar_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_solar(points_df.timezone.unique()[0], 0, 0.75)
	write_jsonfile(sam_config, 'solar1')
	sam_file = os.path.expanduser('json_files/sam_config_solar1.json')
elif renewable_choice == 'solar2':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'solar_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_solar(points_df.timezone.unique()[0], 0, 1)
	write_jsonfile(sam_config, 'solar2')
	sam_file = os.path.expanduser('json_files/sam_config_solar2.json')
elif renewable_choice == 'solar3':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'solar_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_solar(points_df.timezone.unique()[0], 0, 1.3)
	write_jsonfile(sam_config, 'solar3')
	sam_file = os.path.expanduser('json_files/sam_config_solar3.json')
elif renewable_choice == 'solar4':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'solar_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_solar(points_df.timezone.unique()[0], 0, 1.5)
	write_jsonfile(sam_config, 'solar4')
	sam_file = os.path.expanduser('json_files/sam_config_solar4.json')
elif renewable_choice == 'solar5':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'solar_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_solar(points_df.timezone.unique()[0], 2, 0.5)
	write_jsonfile(sam_config, 'solar5')
	sam_file = os.path.expanduser('json_files/sam_config_solar5.json')
elif renewable_choice == 'solar6':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'solar_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_solar(points_df.timezone.unique()[0], 2, 0.75)
	write_jsonfile(sam_config, 'solar6')
	sam_file = os.path.expanduser('json_files/sam_config_solar6.json')
elif renewable_choice == 'solar7':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'solar_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_solar(points_df.timezone.unique()[0], 2, 1)
	write_jsonfile(sam_config, 'solar7')
	sam_file = os.path.expanduser('json_files/sam_config_solar7.json')
elif renewable_choice == 'solar8':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'solar_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_solar(points_df.timezone.unique()[0], 2, 1.3)
	write_jsonfile(sam_config, 'solar8')
	sam_file = os.path.expanduser('json_files/sam_config_solar8.json')
elif renewable_choice == 'solar9':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'solar_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_solar(points_df.timezone.unique()[0], 2, 1.5)
	write_jsonfile(sam_config, 'solar9')
	sam_file = os.path.expanduser('json_files/sam_config_solar9.json')
else:
	res_file = WIND
	renewable = 'windpower'
	points_path = 'wind_points.csv'
	points_df = pd.read_csv(points_path)
	sam_config = create_g126_84(0)
	write_jsonfile(sam_config, 'g126_102')
	sam_file = os.path.expanduser('json_files/sam_config_g126_102.json')

def run_rev(df, res_file, sam_file, renewable, j):
	lat_lon = []
	for index, row in df.iterrows():
		lat_lon.append([row.latitude, row.longitude])

	pp = ProjectPoints.lat_lon_coords(lat_lon, res_file, sam_file, tech=renewable)
	pc = PointsControl(pp, sites_per_split=1)
	gen = Gen.reV_run(tech=renewable, points=pc, sam_files=sam_file,
						res_file=res_file, max_workers=1, fout=None,
						output_request=("cf_mean","cf_profile"))

	profile_df = pd.DataFrame(gen.out['cf_profile'])

	if renewable == 'windpower':
		result_df = pd.DataFrame()
		for col, item in profile_df.iteritems():
			mean_wind = []
			for i in range(8760):
				lst = item.loc[i*12:i*12+11]
				mean_wind.append(lst.mean()) 
			result_df[col] = mean_wind
	else:
		result_df = profile_df
    
	if not os.path.exists('output/'+renewable_choice):
		os.makedirs('output/'+renewable_choice)

	result_df.to_csv('output/'+renewable_choice+'/'+str(j)+'.csv')

rangeval = int(len(points_df)/500)+1

for i in range(rangeval):
    if i < rangeval-1:
        df = points_df.iloc[500*i:500*i+500]
    else:
        df = points_df.iloc[500*i:]
    run_rev(df, res_file, sam_file, renewable, i)

renewable_choice = 'solar9'
_, _, filenames = next(walk('output/'+renewable_choice+'/'))

dfs = []
for file in filenames:
	dfs.append(pd.read_csv('output/'+renewable_choice+'/'+file,index_col=0))

result = pd.concat(dfs,axis=1)

for lat in list(range(8,36)):
	lat = str(lat)
	for dcac in ['1.3','1.5']:
		result = pd.read_csv('data/0_'+lat+'_'+dcac+'.csv',index_col=0)
		points_path = 'solar_points'+lat+'.csv'
		points_df = pd.read_csv(points_path)
		#points_df = points_df[:4500]
		#points_df.to_csv('wind_points_reduced.csv')

		result.columns = points_df.gid.tolist()

		capacityFactorDF = pd.DataFrame()
		for point in points_df.index_right.unique():
			capacityFactorDF[point] = np.roll(result[points_df[points_df.index_right == point].gid.tolist()].mean(axis=1),5)

		capacityFactorDF.to_csv('cf_0_'+lat+'_'+dcac+'.csv')

