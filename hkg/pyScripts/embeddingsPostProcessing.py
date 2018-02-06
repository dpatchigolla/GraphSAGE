#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:54:29 2018

@author: dileeppatchigolla
Creates a csv file with node-embeddings-class mapping from the embeddings that are generated
using GraphSAGE

"""


import json
import numpy as np
import pandas as pd




### Config for embeddings and Graph that are generated on HQ Only users, with 9 features -using graphsage_mean and 1k iters
modelFolder = 'hqUsers'
modelName = 'hqsubGraphClustering'
dumpFilesPath = '/Users/dileeppatchigolla/git_repo/GraphSAGE/'+modelFolder+'/'+modelName
#dumpFilesPath = '/Users/dileeppatchigolla/git_repo/GraphSAGE/example_data/sampleABData'
unsupVals='/Users/dileeppatchigolla/git_repo/GraphSAGE/unsup-'+modelFolder+'/graphsage_mean_small_0.000010/val.npy'
dumpEmbeddings ='../data/'+modelName+'_embeddings_node_class_map.csv'



### Config for embeddings and Graph that are generated on HQ Only users, with 9 features - 
### Using graphsage_maxpool and 1k iters
modelFolder = 'hqClusteringUsers'
modelName = 'hqsubGraphClustering'
resultFolder='/Users/dileeppatchigolla/git_repo/GraphSAGE/unsup-'+modelFolder+\
'/graphsage_maxpool_small_0.000010/identity1kItersMaxPool/'
unsupVals = resultFolder + 'val.npy'
unsupIndex = resultFolder + 'val.txt'
dumpEmbeddings ='../data/'+modelName+'_maxpool_1k_embeddings_node_class_map.csv'








classes = 0 
## if classes = 0, it uses val.txt to get uid and val.npy for embeddings. it just joins both these

if classes == 1:
    ## To create a dataframe with uid, features, classes
    ## first create dictionary with uid as keys and embed features as values
    vals = np.load(unsupVals)
    idMap = json.load(open(dumpFilesPath+'-id_map.json'))
    classMap = json.load(open(dumpFilesPath+'-class_map.json'))
    feats={}
    for key in idMap:
        feats[key]=vals[idMap[key]]
        
    classDict = pd.DataFrame.from_dict(classMap,orient='index')
    classDict.columns=['hqFlag']
    featDict = pd.DataFrame.from_dict(feats,orient='index')
    featDict.columns = ['f'+str(x) for x in range(featDict.shape[1])]
    featDict.reset_index(inplace=True)
    classDict.reset_index(inplace=True)
    classDict.rename(columns={'index':'user_id'},inplace=True)
    featDict.rename(columns={'index':'user_id'},inplace=True)
    rawData = pd.merge(featDict,classDict,how='inner',on='user_id')
    rawData.to_csv(dumpEmbeddings,index=False)


if classes == 0:
    ## To create a dataframe with just uid and features:
    vals = np.load(unsupVals)
    uidFile = open(unsupIndex, "r")
    lines = uidFile.read().split('\n')
    uids = np.array(lines)
    del lines
    uids = pd.DataFrame(uids)
    feats = pd.DataFrame(vals)
    del vals
    uids.columns=['user_id']
    feats.columns=['f'+str(i) for i in range(feats.shape[1])]
    feats.reset_index(inplace=True)
    uids.reset_index(inplace=True)
    rawData = pd.merge(uids,feats,how='inner',on='index')
    del rawData['index']
    del feats
    del uids
    print(rawData.shape)
    rawData.to_csv(dumpEmbeddings,index=False)


    



### Runing kmeans clustering on vals:

from sklearn.cluster import KMeans

## clusters with mean 1k iters, not using nodes as features
dumpClustersFile='../data/'+modelName+'_clusters.csv'

## Clusters with nodes as features - maxpool with 1k iters
dumpClustersFile='../data/nodesAsFeatures/'+modelName+'_clusters.csv' 

kmeans = KMeans(n_clusters=9, random_state=0).fit(vals)
clusters = kmeans.labels_
print("Sum of Squared: "+str(kmeans.inertia_))
clusterCentres = kmeans.cluster_centers_

kmeans2 = KMeans(n_clusters=1, random_state=0).fit(vals)
print("Sum of Squared: "+str(kmeans2.inertia_))


print("Drop in Sum of Squared due to clustering: "+str(1 - kmeans.inertia_/kmeans2.inertia_))
## Result for embeddings on HQ Only, with 9 features and nodes also as features
## node_dim: 64, iters: 1000, max_pool
## 0.3666 is the drop in sum of squared due to kmeans clustering
## original : 852151 after kmeans: 539747


## To Create dataframe with uid and their cluster
## create dict with uid as keys and cluster as values
uidClust ={}
for key in range(len(uids)):
    uidClust[uids[key]]=clusters[key]

uidClust = pd.DataFrame.from_dict(uidClust, orient='index')
uidClust.columns=['cluster']
uidClust.reset_index(inplace=True)
uidClust.rename(columns={'index':'user_id'},inplace=True)

uidClust.to_csv(dumpClustersFile,index=False)



### Profiling users based on clusters
nodeInfoFile='../data/nodeInfo2.csv'

nodeInfo = pd.read_csv(nodeInfoFile)
nodeInfo = pd.merge(nodeInfo,uidClust,on='user_id',how='inner')

flagVars = [i for i in list(nodeInfo) if i.startswith('is_')]

flagVarsSum = nodeInfo.groupby('cluster')[flagVars].mean()

allVars=list(nodeInfo.ix[:,2:-1])
allVarsSum = nodeInfo.groupby('cluster')[allVars].mean()

