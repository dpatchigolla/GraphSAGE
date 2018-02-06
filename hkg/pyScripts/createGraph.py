#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:14:31 2018

@author: dileeppatchigolla
"""


import csv
import StringIO
import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np


### set seedNodes according to whether we want hq users and their whole network or only hq users.
#seedNodes='hq_and_their_links'
#seedNodes='hq_only'
seedNodes='hq_only_clusteringFeatures'

embeddingFolder = '/Users/dileeppatchigolla/git_repo/GraphSAGE/'
rawDataFolder = '/Users/dileeppatchigolla/Dropbox/Work/datascience/network_embedding/highEngagedUsers/data/'

if seedNodes=='hq_and_their_links':
    #### Config for Whole graph of hq users and their circle
    print("Creating Graph for Whole HQ users and their contacts. Total 2.38M nodes")
    dumpFilesPath = embeddingFolder + 'hqUsers/hqGraph'
    edgesInfoFile = rawDataFolder+'edgeInfo.csv'
    nodeInfoFile= rawDataFolder+'nodeInfo.csv'
elif seedNodes=='hq_only':
    #### Config for graph of only hq users and not their circle
    print("Creating Graph for only HQ users. Total 886k nodes")
    dumpFilesPath = embeddingFolder + 'hqUsers/hqsubGraph'
    edgesInfoFile = rawDataFolder+'edgeInfo.csv'
    nodeInfoFile= rawDataFolder+'nodeInfo2.csv'
elif seedNodes=='hq_only_clusteringFeatures':
    print("creating graph for only HQ Users. Total 886k nodes. Uses only a few features for clustering")
    dumpFilesPath = embeddingFolder + 'hqUsers/hqsubGraphClustering'
    edgesInfoFile = rawDataFolder+'edgeInfo.csv'
    nodeInfoFile= rawDataFolder+'nodeInfoClusteringSubsetFeatures.csv'    
elif seedNodes=='hq_extended_large':
    print("creating graph for HQ and their extended contacts. Total 3.3M nodes")
    dumpFilesPath = embeddingFolder +'hqExtendedUsers/hqextendedGraph'
    edgesInfoFile= rawDataFolder+'edgeInfoLargeGraph.csv'
    nodeInfoFile= rawDataFolder+'nodeInfoLargeGraph.csv'



def valTestGenerator(test_n,val_n):
    ## test_n: Probability that test = True
    ## val_n: probability that val = True
    r = np.random.rand()
    if r < test_n:
        d={'val':False,'test':True}
    elif r < test_n + val_n:
        d={'val':True,'test':False}
    else:
        d={'val':False,'test':False}
    return d


G = nx.Graph()

## Adding Nodes:
nodeFile = open(nodeInfoFile,'rb')
nodeReader = csv.reader(nodeFile)
nodes=[]
for row in nodeReader:
    if row[0]=='user_id':
        continue
    else:
        d=valTestGenerator(0.22,0.12)
        G.add_node(row[0],{'test':d['test'],'val':d['val'],
                           'feature': [round(float(x),2) if x else 0.0 for x in row[2:]] ,
                           'label': [int(float(row[1]))]})
    
    
## Adding Links:

linkFile = open(edgesInfoFile,'rb')
linkReader = csv.reader(linkFile)
for link in linkReader:
    G.add_edge(link[0],link[1])
    
    
## To test whether the Graph has any invalid nodes
t=0
f=0
nodes=[]
for n in G.nodes():
    try:
        G.node[n]['val']
        t+=1
    except:
        f+=1
        nodes.append(n)
print("Error in "+ str(f) + " nodes")

#print(nodes)
#G.remove_node('1')

G.remove_nodes_from(nodes)

print("Total nodes: "+str(len(G.nodes())))
print("Total edges: "+str(len(G.edges()))) 


## Degree of node:
print("Degree of graph: "+str(np.mean(nx.degree(G).values())))


### Creating supporting files - classmap, idmap, feats npy array
idMap = dict(zip(G.nodes(),range(len(G.nodes()))))


classMap={}
for val in G.nodes():
    classMap[val]=G.node[val]['label']


## Copying data into json files
data = json_graph.node_link_data(G)
with open(dumpFilesPath+'-G.json', 'w') as outfile:
    json.dump(data, outfile)

### Took nearly 15 mins to write for HQ and its edges graph.

### Took nearly 5 mins to write for HQ only.
    
del data


with open(dumpFilesPath+'-class_map.json','w') as outfile:
    json.dump(classMap,outfile)

del classMap

with open(dumpFilesPath+'-id_map.json','w') as outfile:
    json.dump(idMap,outfile)


    





idMapInv = {v: k for k, v in idMap.iteritems()}
del idMap

feat = np.empty((0,len(G.node[G.nodes()[0]]['feature'])))


L=[]
for val in range(len(G.nodes())):
    f =G.node[idMapInv[val]]['feature']
    L.append(f)
    

feat=np.asarray(L)
### Took 15 sec

# =============================================================================
# for val in range(len(G.nodes())):
#     f =G.node[idMapInv[val]]['feature']
#     feat = np.append(feat,[f],axis=0)
# 
# =============================================================================
del G

with open(dumpFilesPath+'-feats.npy','w') as outfile:
    np.save(outfile, feat)