import os
import numpy as np
from reV import TESTDATADIR
from reV.config.project_points import ProjectPoints
from reV.generation.generation import Gen

lat_lons = np.array([[ 41.25, -71.66],
                     [ 41.05, -71.74],
                     [ 41.45, -71.66],
                     [ 41.97, -71.78],
                     [ 41.65, -71.74],
                     [ 41.53, -71.7 ],
                     [ 41.25, -71.7 ],
                     [ 41.05, -71.78],
                     [ 42.01, -71.74],
                     [ 41.45, -71.78]])

res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
sam_file = os.path.join(TESTDATADIR,
                         'SAM/wind_gen_standard_losses_0.json')

pp = ProjectPoints.lat_lon_coords(lat_lons, res_file, sam_file)
gen = Gen.reV_run(tech='windpower', points=pp, sam_files=sam_file,
                  res_file=res_file, max_workers=1, fout=None,
                  output_request=('cf_mean', 'cf_profile'))
print(gen.out['cf_profile'])