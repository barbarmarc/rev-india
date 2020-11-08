#%% packages
import os
import pandas as pd
#import geopandas as gpd
from reV.config.project_points import ProjectPoints, PointsControl
from reV.generation.generation import Gen
from rex import init_logger
init_logger("reV.generation", log_level="DEBUG", log_file="./rev.log")
init_logger("reV.config", log_level="DEBUG", log_file="./rev.log")
from create_json import *

#%% Path
SOLAR = "K:/Data/NREL/NSRDB/nsrdb_india_2014.h5"
WIND = "K:/Data/NREL/WTK/wtk_india_2014.h5"

renewable_choice = input("Enter renewable resource: ")
region = input("Enter a region: ")

if renewable_choice == 'solar':
	res_file = SOLAR
	renewable = 'pvwattsv7'
	points_path = 'genx/points/hr_sampling/solar/'+region+'_points.csv'
	points_df = pd.read_csv(points_path)

	sam_config = create_solar(points_df.timezone.unique()[0])
	write_json(sam_config, 'solar', region)
	sam_file = os.path.expanduser('genx/json/sam_config_'+'solar'+'_'+region+'.json')
else:
	res_file = WIND
	renewable = 'windpower'
	points_path = 'genx/points/hr_sampling/wind/'+region+'_points.csv'
	points_df = pd.read_csv(points_path)

	sam_config = create_g126_84(0)
	write_json(sam_config, 'g126_84', region)
	sam_file = os.path.expanduser('genx/json/sam_config_'+'g126_84'+'_'+region+'.json')

print(renewable, points_path)

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

	result_df.to_csv(points_path[:-4]+'_'+str(j)+'.csv')
	
if len(points_df) == 2000:
	for i in range(4):

		df = points_df.loc[500*i:500*i+499]
		
		run_rev(df, res_file, sam_file, renewable, i)

else:
	rangeval = int(len(points_df)/500)+1
	if rangeval > 1:
		for i in range(rangeval):
			if i < rangeval-1:
				df = points_df.loc[500*i:500*i+499]
			else:
				df = points_df.loc[500*i:]
			run_rev(df, res_file, sam_file, renewable, i)
	else:
		df = points_df
		run_rev(df, res_file, sam_file, renewable, 0)
		
