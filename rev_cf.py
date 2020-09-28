#%% packages
import os
import pandas as pd
from reV.config.project_points import ProjectPoints
from reV.generation.generation import Gen
from create_json import *

#%% Path
SOLAR = "/nrel/nsrdb/india/nsrdb_india_2000.h5"
WIND = "/nrel/wtk/india/wtk_india_2014.h5"

windpath = 'genx/points/wind/'
samplepath = 'genx/points/sample/'
resultpath = 'genx/results/'
jsonpath = 'genx/json/sam_config.json'
solarpath = 'genx/points/solar/'

#%% Wind
point_files = os.listdir(windpath)
regions = ['NR', 'WR', 'SR', 'ER', 'NER']
tech = ['w0', 'w1', 'w2', 'g126_84', 'g126_102']

for p in point_files:
    for t in tech:
        full_gid = pd.read_csv(windpath+p)
        for _ in range(int(len(full_gid)/100)):
            sample_gid = full_gid.sample(25)
            sample_gid.to_csv(samplepath+p)

            points_df = pd.read_csv(samplepath+p)
            points = os.path.expanduser(samplepath+p)
            lat_lon = []
            for index, row in points_df.iterrows():
                lat_lon.append([row.latitude, row.longitude])

            if t == 'w0':
                res_file = WIND
                sam_config =create_w0(0)
                write_json(sam_config)
                sam_file = os.path.expanduser(jsonpath)
                renewable = 'windpower'
            elif t == 'w1':
                res_file = WIND
                sam_file = create_w1(0)
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
            elif t == 'solar':
                res_file = SOLAR
                sam_config = create_solar(points_df.timezone.unique()[0])
                write_json(sam_config)
                sam_file = os.path.expanduser(jsonpath)
                renewable = 'pvwattsv7'

            pp = ProjectPoints.lat_lon_coords(lat_lon, res_file, sam_file)
            gen = Gen.reV_run(tech=renewable, points=pp, sam_files=sam_file,
                                res_file=res_file, max_workers=1, fout=None,
                                output_request=("cf_mean", "cf_profile"))

            profile_df = pd.DataFrame(gen.out['cf_profile'])
            df_mean = []
            for index, row in profile_df.iterrows():
                df_mean.append(row.mean())
            df_mean = pd.DataFrame(df_mean)
            try:
                results_df = pd.read_csv(resultpath+p[:-10]+t+'_cf.csv')
                results_df = results_df.append(df_mean, ignore_index=True)
                results_df.to_csv(resultpath+p[:-10]+t+'_cf.csv')
            except:
                df_mean.to_csv(resultpath+p[:-10]+t+'_cf.csv')

#%% Solar
point_files = os.listdir(solarpath)

for p in point_files:
    full_gid = pd.read_csv(solarpath+p)
    for _ in range(int(len(full_gid)/100)):
        sample_gid = full_gid.sample(25)
        sample_gid.to_csv(samplepath+p)

        points_df = pd.read_csv(samplepath+p)
        points = os.path.expanduser(samplepath+p)
        lat_lon = []
        for index, row in points_df.iterrows():
            lat_lon.append([row.latitude, row.longitude])

        res_file = SOLAR
        sam_config = create_solar(points_df.timezone.unique()[0])
        write_json(sam_config)
        sam_file = os.path.expanduser(jsonpath)
        renewable = 'pvwattsv7'

        pp = ProjectPoints.lat_lon_coords(lat_lon, res_file, sam_file)
        gen = Gen.reV_run(tech=renewable, points=pp, sam_files=sam_file,
                            res_file=res_file, max_workers=1, fout=None,
                            output_request=("cf_mean", "cf_profile"))

        profile_df = pd.DataFrame(gen.out['cf_profile'])
        df_mean = []
        for index, row in profile_df.iterrows():
            df_mean.append(row.mean())
        df_mean = pd.DataFrame(df_mean)
        try:
            results_df = pd.read_csv(resultpath+p[:-10]+t+'_cf.csv')
            results_df = results_df.append(df_mean, ignore_index=True)
            results_df.to_csv(resultpath+p[:-10]+t+'_cf.csv')
        except:
            df_mean.to_csv(resultpath+p[:-10]+t+'_cf.csv')