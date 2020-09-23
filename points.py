from rex import Resource

NSRDB_SAMPLE = "/nrel/nsrdb/india/nsrdb_india_2000.h5"

with Resource(NSRDB_SAMPLE, hsds=True) as file:
    points = file.meta

points.index.name = "gid"
points = points.reset_index()
points["config"] = "default"

points.to_csv("points_projects.csv")