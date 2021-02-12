import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import networkx as nx
import math
import pyproj
from rasterio.mask import mask
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial import cKDTree
import shapely.ops as ops
from functools import partial
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import colors
from shapely.geometry import mapping
from area import area
from pyproj import Geod


def clean(df1, no_data, tech=None, clip=False):
    left = df1[0]
    right = df1[len(df1.columns)-1]
    upper = df1.loc[0]
    lower = df1.loc[len(df1.index)-1]

    if upper.unique()[0] == no_data and len(upper.unique())==1:
        df1 = df1[1:]

    if lower.unique()[0] == no_data and len(lower.unique())==1:
        df1 = df1[:-1]

    if left.unique()[0] == no_data and len(left.unique())==1:
        df1 = df1[df1.columns.tolist()[1:]]   

    if right.unique()[0] == no_data and len(right.unique())==1:
        df1 = df1[df1.columns.tolist()[:-1]]   

    if clip == True:
        df1 = df1.where(df1 < no_data, other=0)

    if tech == 'solar':
        df1 = df1[df1.isin([14,8,5,13])] 
        df1 = df1.fillna(0)
    elif tech == 'wind':
        df1 = df1[df1.isin([14,8,2,5,6,7,13])] 
        df1 = df1.fillna(0)
    else:
        pass
    return df1

def get_potential():

    all_solar, all_wind = [], []
    shapefile = gpd.read_file("supplycurve/india/indiagrid.shp")

    areas = []
    for index, row in shapefile.iterrows():

        # transform to GeJSON format
        geoms = [mapping(row.geometry)]

        geod = Geod(ellps="WGS84")
        area = abs(geod.geometry_area_perimeter(row.geometry)[0])
        areas.append(area/1e6)
        # extract the raster values values within the polygon 

        with rio.open("supplycurve/output.tif") as src:
            out_image, out_transform = mask(src, geoms, crop=True)
            no_data = src.nodata
        df1solar = clean(pd.DataFrame(out_image[0]),no_data, tech='solar',clip=True)
        df1wind = clean(pd.DataFrame(out_image[0]), no_data, tech='wind',clip=True)
        df2 = clean(pd.DataFrame(out_image[1]), no_data) # solar
        df3 = clean(pd.DataFrame(out_image[2]), no_data) # wind

        df4 = df1solar.mul(df2) # solar
        df5 = df1wind.mul(df3) # wind

        df1_solar = df1solar[df4!=0]
        df1_solar = df1_solar.where(~np.isnan(df1_solar), other=no_data) 
        solar_area = df1_solar.to_numpy()

        df1_wind = df1wind[df5!=0]
        df1_wind = df1_wind.where(~np.isnan(df1_wind), other=no_data)
        wind_area = df1_wind.to_numpy()


        unique_solar, counts_solar = np.unique(solar_area, return_counts=True)
        unique_wind, counts_wind = np.unique(wind_area, return_counts=True)

        solar_value_counts = dict(zip(unique_solar, [round(100*i,2) for i in counts_solar/counts_solar.sum()]))
        
        if no_data in solar_value_counts.keys():
            solar_percent = round(100-solar_value_counts[no_data],2)
        else:
            solar_percent = 100

        wind_value_counts = dict(zip(unique_wind, [round(100*i,2) for i in counts_wind/counts_wind.sum()]))

        if no_data in wind_value_counts.keys():
            wind_percent = round(100-wind_value_counts[no_data],2)
        else:
            wind_percent = 100

        all_solar.append(solar_percent)
        all_wind.append(wind_percent)

        print(index/len(shapefile))

    shapefile['solar'] = all_solar
    shapefile['wind'] = all_wind
    shapefile['area'] = areas

    gdf = gpd.GeoDataFrame(shapefile, geometry='geometry')
    gdf.to_file('supplycurve/lulc_tech.shp')


def hed(point1, point2):

    R = 6372800  # Earth radius in meters

    lat1 = point1.x
    lon1 = point1.y
    lat2 = point2.x
    lon2 = point2.y

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return (2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a)))/1000  # in kilometers


def transmission_cost():
    lulc = gpd.read_file('supplycurve/lulc_tech.shp')
    transmission_nodes = gpd.read_file(
        'supplycurve/nx/nodes.shp')
    net = nx.read_shp(
        'supplycurve/nx/edges.shp')
    nodea = nx.get_edge_attributes(net, 'NodeA')
    nodeb = nx.get_edge_attributes(net, 'NodeB')
    voltage = nx.get_edge_attributes(net, 'voltage')
    name = nx.get_edge_attributes(net, 'name')
    atr = pd.DataFrame(zip(list(nodea.values()), list(nodeb.values()), list(
        voltage.values()), list(name.values())), columns=['NodeA', 'NodeB', 'voltage', 'name'])
    atr = atr[atr.voltage > 0]

    cost_df = pd.DataFrame([[554456.63, 16647.24, 68468.01], [242, 295, 312]], columns=[
                           765, 400, 220], index=['cc_km', 'cc_mw_km'])
    cost_df = cost_df.T

    node_points, node_voltage = [], []
    for index, row in atr.iterrows():
        if row.NodeA not in node_points:
            node_points.append(row.NodeA)
            node_voltage.append(row.voltage)
        if row.NodeB not in node_points:
            node_points.append(row.NodeB)
            node_voltage.append(row.voltage)

    node_df = pd.DataFrame(zip(node_points, node_voltage),
                           columns=['Node', 'voltage'])

    cc_lulc = []
    cnt = 1
    for g in lulc.geometry:
        centroid = g.centroid

        distance = []
        for i in transmission_nodes.geometry:
            distance.append(hed(i, centroid))

        transmission_nodes['distance'] = distance

        cc_mw = []
        for index, row in transmission_nodes.iterrows():
            if row.Node in node_df.Node.tolist():
                voltage_level = node_df[node_df.Node ==
                    row.Node].voltage.iloc[0]
                if voltage_level == 400:
                    cc_mw.append(cost_df['cc_mw_km'][400]*row.distance)
                elif voltage_level == 500:
                    cc_mw.append(cost_df['cc_mw_km'][400]*row.distance)
                else:
                    cc_mw.append(cost_df['cc_mw_km'][765]*row.distance)
            else:
                cc_mw.append(np.inf)

        transmission_nodes['cc_mw'] = cc_mw

        val = transmission_nodes[transmission_nodes.cc_mw ==
            transmission_nodes.cc_mw.min()]

        cc_lulc.append(val.cc_mw.iloc[0])

        print(100*(cnt/len(lulc)))
        cnt += 1

    lulc['IX_MW'] = cc_lulc

    gdf = gpd.GeoDataFrame(lulc, geometry='geometry')
    gdf.to_file('supplycurve/lulc_tech_tr.shp')


def calc_lcoe(mw, cf, tech):
    if tech == 'solar':
        cc = 633.51 * mw * 1000  # $/kW
        fom = 15.2 * mw * 1000  # $/kW-yr
    else:
        cc = 1072.47 * mw * 1000  # $/kW
        fom = 42.1 * mw * 1000  # $/kW-yr
    fcr = 0.09
    if mw != 0:
        return ((cc*fcr)+fom)/(8760*mw*cf)
    else:
        return 0


def calc_lcoe_tr(mw, cf, tr, tech):
    if tech == 'solar':
        cc = 633.51 * mw * 1000  # $/kW
        cc += tr * mw
        fom = 15.2 * mw * 1000  # $/kW-yr
    else:
        cc = 1072.47 * mw * 1000  # $/kW
        cc += tr * mw
        fom = 42.1 * mw * 1000  # $/kW-yr
    fcr = 0.09
    if mw != 0:
        return ((cc*fcr)+fom)/(8760*mw*cf)
    else:
        return 0


def ckdnearest(gdA, gdB):
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdf = pd.concat(
        [gdA.reset_index(drop=True), gdB.loc[idx, gdB.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf


def get_lcoe():
    lulc = gpd.read_file('supplycurve/lulc_tech_tr.shp')
    solar_cf = gpd.read_file('supplycurve/hr_sampling/solar/solar_cf.shp')
    wind_cf = gpd.read_file('supplycurve/hr_sampling/wind/wind_cf.shp')

    centroids = []
    for g in lulc.geometry:
        centroids.append(g.centroid)
    lulc['centroid'] = centroids
    lulc = lulc.rename(columns={"geometry": "polygon", "centroid": "geometry"})

    lulc = ckdnearest(lulc, solar_cf)
    lulc = lulc.rename(columns={"cf_mean": "cf_solar", "gid": "gid_solar"})
    lulc = ckdnearest(lulc, wind_cf)
    lulc = lulc.rename(
        columns={"geometry": "centroid", "polygon": "geometry", "cf_mean": "cf_wind", "gid": "gid_wind"})
    del lulc['lat']
    del lulc['lon']
    del lulc['dist']
    del lulc['centroid']

    solar_pd = 32  # mw/km2
    wind_pd = 4  # mw/km2

    lulc['mw_solar'] = lulc.apply(
        lambda x: x['area']*solar_pd*(x['solar']/100), axis=1)
    lulc['mw_wind'] = lulc.apply(
        lambda x: x['area']*wind_pd*(x['wind']/100), axis=1)

    lulc['lcoe_w'] = lulc.apply(lambda x: calc_lcoe(
        x['mw_wind'], x['cf_wind'], 'wind'), axis=1)

    lulc['lcoe_s'] = lulc.apply(lambda x: calc_lcoe(
        x['mw_solar'], x['cf_solar'], 'solar'), axis=1)

    lulc['lcoe_w_tr'] = lulc.apply(lambda x: calc_lcoe_tr(
        x['mw_wind'], x['cf_wind'], x['IX_MW'], 'wind'), axis=1)

    lulc['lcoe_s_tr'] = lulc.apply(lambda x: calc_lcoe_tr(
        x['mw_solar'], x['cf_solar'], x['IX_MW'], 'solar'), axis=1)

    gdf = gpd.GeoDataFrame(lulc, geometry='geometry')
    gdf.to_file('supplycurve/lulc_tech_tr_lcoe.shp')

def get_cf_profiles():

    tech = input('Input renewable energy technology: ')
    path = 'supplycurve/hr_sampling/'+tech+'/'
    files = os.listdir(path)
    file_df = pd.DataFrame(files, columns=['files'])

    for region in ['ER', 'NER', 'NR', 'SR', 'WR']:

        region_df = file_df[file_df.files.str.contains(region)]

        if region == 'ER':
            region_df = region_df[~region_df.files.str.contains('NER')]

        file = region+'_points.csv'
        region_df = region_df[~region_df.files.str.contains(file)]

        file_csv = pd.read_csv(path+file)

        if len(file_csv) == 2000:
            columns = []
            df_cf = pd.DataFrame()
            for i in range(4):
                f_csv = file_csv.loc[500*i:500*i+499]
                f_csv = f_csv.sort_values('gid')
                cf_df = pd.read_csv(path+file[:-4]+'_'+str(i)+'.csv', index_col=0)
                df_cf = pd.concat([df_cf, cf_df], axis=1)
                columns.append(f_csv.gid.tolist())
            columns = [item for sublist in columns for item in sublist]
            df_cf.columns = columns
        else:
            rangeval = int(len(file_csv)/500)+1
            if rangeval > 1:
                columns = []
                df_cf = pd.DataFrame()
                for i in range(rangeval):
                    if i < rangeval-1:
                        f_csv = file_csv.loc[500*i:500*i+499]
                        f_csv = f_csv.sort_values('gid')
                        cf_df = pd.read_csv(path+file[:-4]+'_'+str(i)+'.csv', index_col=0)
                        df_cf = pd.concat([df_cf, cf_df], axis=1)
                        columns.append(f_csv.gid.tolist())
                    else:
                        f_csv = file_csv.loc[500*i:]
                        f_csv = f_csv.sort_values('gid')
                        cf_df = pd.read_csv(path+file[:-4]+'_'+str(i)+'.csv', index_col=0)
                        df_cf = pd.concat([df_cf, cf_df], axis=1)
                        columns.append(f_csv.gid.tolist())
                columns = [item for sublist in columns for item in sublist]
                df_cf.columns = columns
            else:
                df_cf = pd.DataFrame()
                f_csv = file_csv
                f_csv = f_csv.sort_values('gid')
                cf_df = pd.read_csv(path+file[:-4]+'_0.csv', index_col=0)
                df_cf = pd.concat([df_cf, cf_df], axis=1)
                df_cf.columns = f_csv.gid.tolist()


        df_cf.to_csv(path+tech+'_'+region+'_cf.csv')

def build_sc():
    tech = input('Input renewable energy technology: ')
    trans = input('Include transmission cost? (y/n): ')
    if trans == 'y':
        tr = True
    else:
        tr = False
    lulc = gpd.read_file('supplycurve/lulc_tech_tr_lcoe.shp')
    
    lulc = lulc[['cf_solar', 'cf_wind', 'mw_solar', 'mw_wind',
        'lcoe_w', 'lcoe_s', 'lcoe_w_tr', 'lcoe_s_tr', 'IX_MW', 'gid_wind', 'gid_solar']]
    if tr == False:
        if tech == 'solar':
            lcoe = 'lcoe_s'
        else:
            lcoe = 'lcoe_w'
    else:
        if tech == 'solar':
            lcoe = 'lcoe_s_tr'
        else:
            lcoe = 'lcoe_w_tr'
    # {1: 'north', 2: 'west', 3: 'south', 4: 'east', 5: 'ne'}
    for region, r in zip(['ER', 'NER', 'NR', 'SR', 'WR'], ['4', '5', '1', '3', '2']):
        cf_profiles = pd.read_csv('hr_sampling/'+tech+'/'+tech+'_'+region+'_cf.csv', index_col = 0)
        gid = cf_profiles.columns.tolist()
        supply_curve = lulc[['cf_'+tech, 'mw_'+tech, 'gid_'+tech, lcoe, 'IX_MW']]
        supply_curve = supply_curve[supply_curve['gid_'+tech].isin(gid)]
        supply_curve = supply_curve[supply_curve['mw_'+tech] > 0]
        supply_curve = supply_curve.sort_values(lcoe)
        supply_curve = supply_curve.reset_index()

        x = supply_curve[lcoe].to_numpy().reshape(-1, 1)
        clusterer = KMeans(n_clusters=3)
        clusters = clusterer.fit(x)
        supply_curve['cluster'] = clusters.labels_

        centers = list(np.sort(clusters.cluster_centers_.flatten()))
        w_avg_cf, w_avg_cc, avg_mw = [], [], []
        cluster_profiles = pd.DataFrame()
        for n in [0,1,2]:
            df = supply_curve[supply_curve.cluster == n]
            w_avg_cf.append(
                round((df['cf_'+tech] * df['mw_'+tech]).sum() / df['mw_'+tech].sum(), 2))
            w_avg_cc.append(
                round((df['IX_MW'] * df['mw_'+tech]).sum() / df['mw_'+tech].sum(), 2))
            avg_mw.append(round(df['mw_'+tech].sum(), 2))
            cluster_profiles[n] = cf_profiles[df['gid_'+tech].unique()].mean(axis=1)

        simple_sc = pd.DataFrame(zip(centers, w_avg_cf, w_avg_cc, avg_mw), columns=[
                                'LCOE', 'CF', 'IX', 'MW'])
        simple_sc.index = [1,2,3]
        #ax = simple_sc.plot.bar(x='MW', y='LCOE', title=tech+' supply curve')
        if tr == True:
            #ax.set_xlabel('MW')
            #ax.set_ylabel('LCOE $/kWh (including TX)')
            simple_sc.to_csv('supplycurve/sc/sc_'+tech+'_'+r+'.csv')
            cluster_profiles.columns = [1,2,3]
            cluster_profiles.to_csv('supplycurve/cf/cf_profiles_'+tech+'_'+r+'.csv', index=False)
        else:
            #ax.set_xlabel('MW')
            #ax.set_ylabel('LCOE $/kWh (excluding TX)')
            simple_sc.to_csv('supplycurve/sc/sc_'+tech+'_'+r+'_wo_tr.csv', index=False)

        print(region)
        print(simple_sc)
        #plt.show()
