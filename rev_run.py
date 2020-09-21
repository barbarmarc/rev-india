import os
import numpy as np
import pandas as pd
import geopandas as gpd
from reV import TESTDATADIR
from reV.config.project_points import ProjectPoints
from reV.generation.generation import Gen

regions = ['NR', 'WR', 'SR', 'ER', 'NER']

for r in regions:
    gdf = gpd.read_file('india_points/'+r+'_points.shp')

    gdf = gpd.read_file('india_points/'+r+'_points.shp')
    lat_lons = list(zip(gdf.geometry.x, gdf.geometry.y))

    # wind
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2013.h5')
    sam_file = os.path.join(TESTDATADIR, 'SAM/wind_gen_g126_2500.json')

    # solar
    #res_file = os.path.join(TESTDATADIR, 'nsrdb/', 'ri_100_nsrdb_2013.h5')
    #sam_file = os.path.join(TESTDATADIR, 'SAM/naris_pv_1axis_inv13.json')

    profile_df = pd.DataFrame()
    mean = []

    for i in range(len(lat_lons)):

        pp = ProjectPoints.lat_lon_coords(np.array(lat_lons[i]), res_file, sam_file)
        # tech = windpower or pvwattsv7
        gen = Gen.reV_run(tech='windpower', points=pp, sam_files=sam_file,
                        res_file=res_file, max_workers=1, fout=None,
                        output_request=('cf_mean', 'cf_profile'))

        mean.append(gen.out['cf_mean'][0])
        profile_df[i] = gen.out['cf_profile'].flatten()
        
    mean_df = pd.DataFrame(mean)

    mean_df.to_csv('india_points/'+r+'_g126_102_cf_mean.csv')
    profile_df.to_csv('india_points/'+r+'_g126_102_cf_profile.csv')
