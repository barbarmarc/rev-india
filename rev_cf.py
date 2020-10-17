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

# regions = ['NR', 'WR', 'SR', 'ER', 'NER']
windpath = 'genx/points/sample/wind/'
samplepath = 'genx/points/sample/'
resultpath = 'genx/results/'
jsonpath = 'genx/json/sam_config.json'
solarpath = 'genx/points/sample/solar/'

tech = ['w0', 'w1', 'w2', 'g126_84', 'g126_102']

#%% Sampling

point_files = os.listdir('genx/points/wind/')
for p in point_files:
    full_gid = pd.read_csv('genx/points/wind/'+p)
    sample_gid = full_gid.sample(int(len(full_gid)/10))
    sample_gid.to_csv(windpath+p)
    #gdf = gpd.GeoDataFrame(sample_gid, geometry=gpd.points_from_xy(sample_gid.longitude, sample_gid.latitude))
    #gdf.to_file('genx/wind_shp/sample/'+p[:-4]+'_sample.shp')
    
"""
point_files = os.listdir('genx/points/solar/')
for p in point_files:
    full_gid = pd.read_csv('genx/points/solar/'+p)
    sample_gid = full_gid.sample(100)
    sample_gid.to_csv(solarpath+p)
    #gdf = gpd.GeoDataFrame(sample_gid, geometry=gpd.points_from_xy(sample_gid.longitude, sample_gid.latitude))
    #gdf.to_file('genx/solar_shp/sample/'+p[:-4]+'_sample.shp')    
""" 


#%% Wind

point_files = os.listdir(windpath)
for p in point_files:
    for t in tech:

        if os.path.exists("log.txt"):
            with open("log.txt") as f:
                log = f.read().splitlines()
        else:
            log = []
        
        if resultpath+p[:-10]+t not in log:
            points_df = pd.read_csv(windpath+p)

            lat_lon = []
            for index, row in points_df.iterrows():
                lat_lon.append([row.latitude, row.longitude])

            if t == 'w0':
                res_file = WIND
                sam_config = create_w0(0)
                write_json(sam_config)
                sam_file = os.path.expanduser(jsonpath)
                renewable = 'windpower'
            elif t == 'w1':
                res_file = WIND
                sam_config = create_w1(0)
                write_json(sam_config)
                sam_file = os.path.expanduser(jsonpath)
                renewable = 'windpower'
            elif t == 'w2':
                res_file = WIND
                sam_config = create_w2(0)
                write_json(sam_config)
                sam_file = os.path.expanduser(jsonpath)
                renewable = 'windpower'
            elif t == 'g126_84':
                res_file = WIND
                sam_config = create_g126_84(0)
                write_json(sam_config)
                sam_file = os.path.expanduser(jsonpath)
                renewable = 'windpower'
            elif t == 'g126_102':
                res_file = WIND
                sam_config = create_g126_102(0)
                write_json(sam_config)
                sam_file = os.path.expanduser(jsonpath)
                renewable = 'windpower'

            pp = ProjectPoints.lat_lon_coords(lat_lon, res_file, sam_file, tech=renewable)
            pc = PointsControl(pp, sites_per_split=1)
            gen = Gen.reV_run(tech=renewable, points=pc, sam_files=sam_file,
                                res_file=res_file, max_workers=1, fout=None,
                                output_request=("cf_mean","cf_profile"))
			
            profile_df = pd.DataFrame(gen.out['cf_profile'])
        
            df_mean = []
            for index, row in profile_df.iterrows():
                df_mean.append(row.mean())
            df_mean = pd.DataFrame(df_mean)
            df_mean.columns = [p[:-11]]
            df_mean.to_csv(resultpath+p[:-10]+t+'_cf.csv', index=False)
		
	        #profile_df.columns = points_df.gid.tolist()
            #profile_df.to_csv(resultpath+p[:-10]+t+'_cf.csv')

            file = open('log.txt','a')
            file.write(resultpath+p[:-10]+t+"\n")
            file.close()     

#%% Clean Wind
result_files = os.listdir('genx/results/')
for rf in result_files:
    rdf = pd.read_csv('genx/results/'+rf)
    if len(rdf) > 8760:
        mean_wind = []
        for i in range(8760):
            lst = rdf.loc[i*12:i*12+11]
            mean_wind.append(lst.mean()[0])
        df = pd.DataFrame(mean_wind, columns=rdf.columns.tolist())
        df.to_csv('genx/results/'+rf)

#%% Solar
"""
point_files = os.listdir(solarpath)
for p in point_files:

    if os.path.exists("log.txt"):
        with open("log.txt") as f:
            log = f.read().splitlines()
    else:
        log = []

    if resultpath+p[:-10]+'s' not in log:
        points_df = pd.read_csv(solarpath+p)

        lat_lon = []
        for index, row in points_df.iterrows():
            lat_lon.append([row.latitude, row.longitude])

        res_file = SOLAR
        sam_config = create_solar(points_df.timezone.unique()[0])
        write_json(sam_config)
        sam_file = os.path.expanduser(jsonpath)
        renewable = 'pvwattsv7'

        pp = ProjectPoints.lat_lon_coords(lat_lon, res_file, sam_file, tech=renewable)
        pc = PointsControl(pp, sites_per_split=1)
        gen = Gen.reV_run(tech=renewable, points=pc, sam_files=sam_file,
                            res_file=res_file, max_workers=1, fout=None,
                            output_request=("cf_profile"))

        profile_df = pd.DataFrame(gen.out['cf_profile'])
        
        df_mean = []
        for index, row in profile_df.iterrows():
            df_mean.append(row.mean())
        df_mean = pd.DataFrame(df_mean)
        df_mean.columns = [p[:-11]]
        df_mean.to_csv(resultpath+p[:-10]+'s_cf.csv', index=False)
		
	    #profile_df.columns = points_df.gid.tolist()
        #profile_df.to_csv(resultpath+p[:-10]+t+'_cf.csv')

        file = open('log.txt','a')
        file.write(resultpath+p[:-10]+'s'+"\n")
        file.close()      
"""