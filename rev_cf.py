import os
import numpy as np
import pandas as pd
from reV import TESTDATADIR
from reV.config.project_points import ProjectPoints
from reV.generation.generation import Gen

SOLAR = "/nrel/nsrdb/india/nsrdb_india_2000.h5"
WIND = "/nrel/wtk/india/wtk_india_2014.h5"

regions = ['NR', 'WR', 'SR', 'ER', 'NER']
tech = ['w0', 'w1', 'w2', 'g126_84', 'g126_102', 'solar']

for r in regions:
    for t in tech:
        points = pd.read_csv('india_points/'+r+'_points.csv')
        points = os.path.expanduser('india_points/'+r+'_points.csv')
        lat_lon = []
        for index, row in points.iterrows():
            lat_lon.append([row.latitude, row.longitude])

        if t == 'w0':
            res_file = WIND
            sam_file = {"default": os.path.join(TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json')}
            renewable = 'windpower'
        elif t == 'w1':
            res_file = WIND
            sam_file = {"default":os.path.join(TESTDATADIR, 'SAM/wind_gen_standard_losses_1.json')}
            renewable = 'windpower'
        elif t == 'w2':
            res_file = WIND
            sam_file = {"default": os.path.join(TESTDATADIR, 'SAM/wind_gen_standard_losses_2.json')}
            renewable = 'windpower'
        elif t == 'g126_84':
            res_file = WIND
            sam_file = {"default": os.path.join(TESTDATADIR, 'SAM/wind_gen_g126_2500_84.json')}
            renewable = 'windpower'
        elif t == 'g126_102':
            res_file = WIND
            sam_file = {"default": os.path.join(TESTDATADIR, 'SAM/wind_gen_g126_2500_102.json')}
            renewable = 'windpower'
        elif t == 'solar':
            res_file = WIND
            sam_file = {"default": os.path.join(TESTDATADIR, 'SAM/naris_pv_1axis_inv13.json')}
            renewable = 'pvwattsv5'

        profile_df = pd.DataFrame()

        #pp = ProjectPoints.lat_lon_coords(np.array(lat_lon), sam_file)
        gen = Gen.reV_run(tech=renewable, points=points, sam_files=sam_file,
                            res_file=res_file, max_workers=1, fout=None,
                            output_request=("cf_mean", "cf_profile"))

        #gen.out['cf_profile']
        
        for index, row in profile_df.iterrows():
            df_mean = []
            df_mean.append(row.mean())
            df_mean = pd.DataFrame(df_mean)
            df_mean.to_csv('india_points/mean/'+r+'_'+t+'_cf.csv')

        df_mean.to_csv('india_points/results/'+r+'_'+t+'_cf.csv')