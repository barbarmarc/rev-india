import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import networkx as nx
import math
import pyproj
from rasterio.mask import mask
from shapely.geometry import Polygon, Point

path = 'genx/tiff/'

elevation = rio.open(path+'ELE.tif')
lulc = gpd.read_file(path+'lulc_permissible.shp')

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def get_potential(path, elevation, lulc):
    geo = lulc[200:201]
    cnt = 1
    solar, wind = [], []
    for k in range(len(lulc)-1):
        geo = lulc[k:k+1]

        coords = getFeatures(geo)

        out_img, out_transform = mask(dataset=elevation, shapes=coords, crop=True)

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
                    G.add_edge(s,t,weight=w)

        for j in range(len(df)):
            for i in range(len(df.columns)-1):
                s = i+j*len(df.columns)
                t = i+j*len(df.columns)+1
                if s != t:
                    w = slope_lateral[j][i]
                    G.add_edge(s,t,weight=w)

        slope_weight = []
        for i in range(df.size):
            weight=nx.get_edge_attributes(G,'weight')
            weights = []
            for e in G.edges(i):
                try:
                    weights.append(weight[e])
                except:
                    weights.append(weight[(e[1],e[0])])
            slope_weight.append(np.mean(weights))

        slope_weights = []
        for i in range(len(df.columns)):
            slope_weights.append(slope_weight[i*len(df.columns):i*len(df.columns)+len(df.columns)])

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


lulc = gpd.read_file(path+'lulc_permissible_slope.shp')
transmission_nodes = gpd.read_file('C:/Users/marcb/Dropbox (MIT)/FofS/India/Data/GIS/nx/nodes.shp')
net = nx.read_shp('C:/Users/marcb/Dropbox (MIT)/FofS/India/Data/GIS/nx/edges.shp')
nodea = nx.get_edge_attributes(net, 'NodeA') 
nodeb = nx.get_edge_attributes(net, 'NodeB') 
voltage = nx.get_edge_attributes(net, 'voltage')
name = nx.get_edge_attributes(net, 'name')
atr = pd.DataFrame(zip(list(nodea.values()),list(nodeb.values()),list(voltage.values()),list(name.values())), columns=['NodeA','NodeB','voltage','name'])
atr = atr[atr.voltage >0]

cost_df = pd.DataFrame([[554456.63, 16647.24, 68468.01],[242,295,312]],columns=[765,400,220], index=['cc_km','cc_mw_km'])
cost_df = cost_df.T

node_points, node_voltage = [], []
for index, row in atr.iterrows():
    if row.NodeA not in node_points:
        node_points.append(row.NodeA)
        node_voltage.append(row.voltage)
    if row.NodeB not in node_points:
        node_points.append(row.NodeB)
        node_voltage.append(row.voltage)

node_df = pd.DataFrame(zip(node_points, node_voltage),columns=['Node','voltage'])

def haversine_euclidean_distance(point1,point2):
    
    R = 6372800  # Earth radius in meters

    lat1 = point1.x
    lon1 = point1.y
    lat2 = point2.x
    lon2 = point2.y

    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return (2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a)))/1000 # in kilometers

cc_lulc, cc_mw_lulc = [], []
cnt = 1
for g in lulc.geometry:
    centroid = g.centroid

    distance = []
    for i in transmission_nodes.geometry:
        distance.append(haversine_euclidean_distance(i,centroid))
    
    transmission_nodes['distance'] = distance

    cc_km, cc_mw_km = [], []
    for index, row in transmission_nodes.iterrows():
        if row.Node in node_df.Node.tolist():
            voltage_level = node_df[node_df.Node==row.Node].voltage.iloc[0]
            if voltage_level == 400:
                cc_km.append(cost_df[400]['cc_km']*row.distance)
                cc_mw_km.append(cost_df[400]['cc_mw_km'])
            elif voltage_level == 500:
                cc_km.append(cost_df[400]['cc_km']*row.distance)
                cc_mw_km.append(cost_df[400]['cc_mw_km'])
            else:
                cc_km.append(cost_df[765]['cc_km']*row.distance)
                cc_mw_km.append(cost_df[765]['cc_mw_km'])
        else:
            cc_km.append(np.inf)
            cc_mw_km.append(np.inf)
    
    transmission_nodes['capital_cost'] = cc_km
    transmission_nodes['cc_mw_km'] = cc_mw_km

    val = transmission_nodes[transmission_nodes.capital_cost == transmission_nodes.capital_cost.min()]

    cc_lulc.append(val.capital_cost.iloc[0])
    cc_mw_lulc.append(val.cc_mw_km.iloc[0])

    print(100*(cnt/len(lulc)))
    cnt += 1   

lulc['CC'] = cc_lulc
lulc['CC_MW_KM'] = cc_mw_lulc

gdf = gpd.GeoDataFrame(lulc, geometry='geometry')
gdf.to_file(path+'lulc_permissible_slope_tr.shp')