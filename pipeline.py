import os

#%% Gen
from reV.generation.generation import Gen

NSRDB_SAMPLE = "/nrel/nsrdb/india/nsrdb_india_2000.h5"
SAM_SAMPLE = os.path.expanduser("input/sample_sam_config.json")
POINT_SAMPLE = os.path.expanduser("input/project_points.csv")

sam_files = {
    "default": SAM_SAMPLE
}

gen = Gen.reV_run(tech="pvwattsv7", points=POINT_SAMPLE, sam_files=sam_files, res_file=NSRDB_SAMPLE, max_workers=1, fout=None, output_request=("cf_mean", "cf_profile"))

gen.out

#%% Collect


#%% Multi-year

#%% Aggregation

#%% Supply curve
from reV.supply_curve.supply_curve import SupplyCurve

supply_curve = SupplyCurve.simple(sc_points=POINT_SAMPLE, trans_table="trans.csv", fcr=0.096)

#%% Rep profiles

#%% qa-qc

