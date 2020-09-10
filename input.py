import json
import os
from rex import Resource

sam_config = {
    "array_type":2,
    "azimuth":180,
    "dc_ac_ratio":1.3,
    "gcr":0.4,
    "inv_eff":96,
    "losses":14.07566,
    "module_type":0,
    "system_capacity":20000,
    "variable_operating_cost":0
}

config_path = os.path.expanduser("input/sample_sam_config.json")
with open(config_path, "w") as file:
    file.write(json.dumps(sam_config, indent=4))

NSRDB_SAMPLE = "/nrel/nsrdb/india/nsrdb_india_2000.h5"

with Resource(NSRDB_SAMPLE, hsds=True) as file:
    points = file.meta
points = points[(points["latitude"]>37)&(points["longitude"]>97)]

points.index.name = "gid"
points = points.reset_index()
points["config"] = "default"

points.to_csv("input/project_points.csv", index=False)

