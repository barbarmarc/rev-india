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

path = 'genx/tiff/'

# lulc = gpd.read_file(path+'lulc_permissible.shp')


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def get_potential(path, elevation, lulc):
    elevation = rio.open(path+'ELE.tif')
    geo = lulc[200:201]
    cnt = 1
    solar, wind = [], []
    for k in range(len(lulc)-1):
        geo = lulc[k:k+1]

        coords = getFeatures(geo)

        out_img, out_transform = mask(
            dataset=elevation, shapes=coords, crop=True)

        df = pd.DataFrame(out_img[0])
        length = int(df.size/len(df))
        if all(df[0] == 0):
            del df[0]
        if all(df[length-1] == 0):
            del df[length-1]
        if all(df.loc[len(df)-1] == 0):
            df = df.drop(len(df)-1)
        if all(df.loc[0] == 0):
            df = df.drop(0)

        slope_lateral = []
        for index, row in df.iterrows():
            slope_1 = []
            for p in range(len(row.tolist())-1):
                slope_1.append(abs((row.tolist()[p+1]-row.tolist()[p])/925))
            slope_lateral.append(slope_1)

        slope_vertical = []
        for index, row in df.iteritems():
            slope_1 = []
            for p in range(len(row.tolist())-1):
                slope_1.append(abs((row.tolist()[p+1]-row.tolist()[p])/925))
            slope_vertical.append(slope_1)

        import networkx as nx

        G = nx.Graph()

        for i in range(df.size):
            G.add_node(i)

        for j in range(len(df)-1):
            for i in range(len(df.columns)):
                s = i+j*len(df)
                t = s+len(df.columns)
                if s < df.size:
                    w = slope_vertical[i][j]
                    G.add_edge(s, t, weight=w)

        for j in range(len(df)):
            for i in range(len(df.columns)-1):
                s = i+j*len(df.columns)
                t = i+j*len(df.columns)+1
                if s != t:
                    w = slope_lateral[j][i]
                    G.add_edge(s, t, weight=w)

        slope_weight = []
        for i in range(df.size):
            weight = nx.get_edge_attributes(G, 'weight')
            weights = []
            for e in G.edges(i):
                try:
                    weights.append(weight[e])
                except:
                    weights.append(weight[(e[1], e[0])])
            slope_weight.append(np.mean(weights))

        slope_weights = []
        for i in range(len(df.columns)):
            slope_weights.append(
                slope_weight[i*len(df.columns):i*len(df.columns)+len(df.columns)])

        slope_df = pd.DataFrame(slope_weights)
        slope_df = slope_df*100

        slope_solar = slope_df[slope_df <= 5]
        solar_land_percentage = 100*(slope_solar.count().sum()/slope_df.size)

        slope_wind = slope_df[slope_df <= 20]
        wind_land_percentage = 100*(slope_wind.count().sum()/slope_df.size)

        solar.append(solar_land_percentage)
        wind.append(wind_land_percentage)

        print(100*(cnt/len(lulc)))
        cnt += 1

    lulc['solar'] = solar
    lulc['wind'] = wind

    gdf = gpd.GeoDataFrame(lulc, geometry='geometry')
    gdf.to_file(path+'lulc_permissible_slope.shp')


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
    lulc = gpd.read_file(path+'lulc_permissible_slope.shp')
    transmission_nodes = gpd.read_file(
        'C:/Users/marcb/Dropbox (MIT)/FofS/India/Data/GIS/nx/nodes.shp')
    net = nx.read_shp(
        'C:/Users/marcb/Dropbox (MIT)/FofS/India/Data/GIS/nx/edges.shp')
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
    gdf.to_file(path+'lulc_permissible_slope_tr.shp')


def calc_lcoe(mw, cf, tech):
    if tech == 'solar':
        cc = 1566 * mw * 1000  # $/kW
        fom = 19 * mw * 1000  # $/kW-yr
    else:
        cc = 1712 * mw * 1000  # $/kW
        fom = 43 * mw * 1000  # $/kW-yr
    fcr = 0.09
    if mw != 0:
        return ((cc*fcr)+fom)/(8760*mw*cf)
    else:
        return 0


def calc_lcoe_tr(mw, cf, tr, tech):
    if tech == 'solar':
        cc = 1566 * mw * 1000  # $/kW
        cc += tr * mw
        fom = 19 * mw * 1000  # $/kW-yr
    else:
        cc = 1712 * mw * 1000  # $/kW
        cc += tr * mw
        fom = 43 * mw * 1000  # $/kW-yr
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
    lulc = gpd.read_file(path+'lulc_permissible_slope_tr.shp')
    solar_cf = gpd.read_file('hr_sampling/solar/solar_cf.shp')
    wind_cf = gpd.read_file('hr_sampling/wind/wind_cf.shp')

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
    area = 20.7  # km2

    lulc['mw_solar'] = lulc.apply(
        lambda x: area*solar_pd*(x['solar']/100), axis=1)
    lulc['mw_wind'] = lulc.apply(
        lambda x: area*wind_pd*(x['wind']/100), axis=1)

    lulc['lcoe_w'] = lulc.apply(lambda x: calc_lcoe(
        x['mw_wind'], x['cf_wind'], 'wind'), axis=1)

    lulc['lcoe_s'] = lulc.apply(lambda x: calc_lcoe(
        x['mw_solar'], x['cf_solar'], 'solar'), axis=1)

    lulc['lcoe_w_tr'] = lulc.apply(lambda x: calc_lcoe_tr(
        x['mw_wind'], x['cf_wind'], x['IX_MW'], 'wind'), axis=1)

    lulc['lcoe_s_tr'] = lulc.apply(lambda x: calc_lcoe_tr(
        x['mw_solar'], x['cf_solar'], x['IX_MW'], 'solar'), axis=1)

    gdf = gpd.GeoDataFrame(lulc, geometry='geometry')
    gdf.to_file(path+'lulc_permissible_slope_tr_lcoe.shp')

def get_cf_profiles():

    tech = input('Input renewable energy technology: ')
    path = 'hr_sampling/'+tech+'/'
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
    path = 'genx/tiff/'
    lulc = gpd.read_file(path+'lulc_permissible_slope_tr_lcoe.shp')
    
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
            simple_sc.to_csv('genx/supplycurve/sc_'+tech+'_'+r+'.csv')
            cluster_profiles.columns = [1,2,3]
            cluster_profiles.to_csv('genx/supplycurve/cf_profiles_'+tech+'_'+r+'.csv', index=False)
        else:
            #ax.set_xlabel('MW')
            #ax.set_ylabel('LCOE $/kWh (excluding TX)')
            simple_sc.to_csv('genx/supplycurve/sc_'+tech+'_'+r+'_wo_tr.csv', index=False)

        print(region)
        print(simple_sc)
        #plt.show()
