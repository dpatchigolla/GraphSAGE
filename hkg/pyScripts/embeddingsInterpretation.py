#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:17:13 2018

@author: dileeppatchigolla
Embeddings - Intepretation
"""



import pandas as pd
import json
from networkx.readwrite import json_graph
import networkx as nx
import random
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.cm as cm



### For HQ Only Users, with embeddings using mean on 1k iterations
modelFolder = 'hqUsers'
modelName = 'hqsubGraphClustering'
dumpFilesPath = '/Users/dileeppatchigolla/git_repo/GraphSAGE/'+modelFolder+'/'+modelName
embeddingsFile ='../data/'+modelName+'_embeddings_node_class_map.csv'
graphFile = dumpFilesPath+'-G.json'


### For HQ Only users, with embeddings using maxpooling on 1k iterations
### it also uses the nodes as features
modelFolder = 'hqClusteringUsers'
modelName = 'hqsubGraphClustering'
dumpFilesPath = '/Users/dileeppatchigolla/git_repo/GraphSAGE/'+modelFolder+'/'+modelName
embeddingsFile ='../data/'+modelName+'_maxpool_1k_embeddings_node_class_map.csv'
graphFile = dumpFilesPath+'-G.json'
similarityDumpFile = '../data/'+modelName+'_maxpool_1k_embeddings_similarity.csv'


embeddings = pd.read_csv(embeddingsFile)

'''
Check similarty of embeddings based on nearness:
    Take 1000 nodes from graph
    At different distances, take samples.
    node1: 
        distance 1 - 5 nodes
        distance 2 - 5 nodes
        distance 3 - 5 nodes
        .
        .
        distance 6 - 5 nodes
        disjoint   - 5 nodes
    total: 1000 nodes * (6 distances + disjoint) * 5 = max 35000 node embeddings
    each distance: 5000 pairs -> 5000 cosine similarities
    7 sets of cosine similarities -> 7 scatters of cosine similarities
    BEWARE: distance between two nodes in subgraph is not same as in original graph. This needs to be handled
'''

G_data = json.load(open(graphFile))
G = json_graph.node_link_graph(G_data)
del G_data

# =============================================================================
# 
# rNodeIndex = random.sample(range(len(G.nodes())),1000)
# 
# rNodes = [G.nodes()[i] for i in rNodeIndex]
# 
# 
# =============================================================================


deg = G.degree()

hqNodes=[]
for n in deg:
    if deg[n]>4:
        hqNodes.append(n)
        
        

rNodeIndex = random.sample(range(len(hqNodes)),200)
rNodes = [hqNodes[i] for i in rNodeIndex]


nodeDistances={}
i=0
%%time
for node in rNodes:
    i+=1
    nodeDistances[node]={}
    G1=nx.ego_graph(G,node,radius=5)
    for n in G1.nodes():
        d = nx.shortest_path_length(G1,node,n)
        if d not in nodeDistances[node].keys():
            nodeDistances[node][d] = [n]
        elif len(nodeDistances[node][d])<=5:
            nodeDistances[node][d].append(n)
        else:
            continue
    if i%10==0:
        print(i)
        

del G1
print(len(nodeDistances.keys()))
## Total 200 nodes with their neighbors at different distances


## node neighborhood list
nnList=[]

%%time
for rNode in nodeDistances.keys():
    for dist in nodeDistances[rNode]:
        for neigh in nodeDistances[rNode][dist]:
            nnList.append([rNode,neigh,dist])
    disjNodeIndex = random.sample(range(10000),10)
    disjNodes=[deg.keys()[i] for i in disjNodeIndex]
    for n in disjNodes:
        try:
            d = nx.shortest_path_length(G,node,n)
            if d>10:
                nnList.append([node,n,11])
        except:
            nnList.append([node,n,100])


# =============================================================================
# for node in nodeDistances.keys():
#     disjNodeIndex = random.sample(range(1000),10)
#     disjNodes=[rNodes[i] for i in disjNodeIndex]
#     for n in disjNodes:
#         try:
#             nx.shortest_path_length(G,node,n)
#         except:
#             nnList.append([node,n,100])
# =============================================================================
# =============================================================================
#             
# nnDF = pd.DataFrame(nnList)
# 
# nnDF.columns=['node','neighbor','distance']
# 
# nnDF.shape
# nnDF['distance'].value_counts()
# 
# ## Dropping the rows where node and neighbor are same
# nnDF=nnDF.drop(nnDF[nnDF['distance']==0].index)
# 
# nnDF.shape
# nnDF['distance'].value_counts()
# 
# =============================================================================

nodesList=[]
for i in nnList:
    nodesList.append(i[0])
    nodesList.append(i[1])

## Deduplicating nodeslist 
nodesList = list(set(nodesList))

embeddings2=embeddings[embeddings['user_id'].isin(nodesList)]

nnList2=[]
i=0
##%%time
for val in nnList:
    if val[2]==0:
        continue
    else:
        e1=embeddings2[embeddings2['user_id']==val[0]].ix[:,1:-1].values.tolist()
        e2=embeddings2[embeddings2['user_id']==val[1]].ix[:,1:-1].values.tolist()
        s=cosine_similarity(e1,e2)
        nnList2.append([val[0],val[1],val[2],s[0][0]])
    i+=1
    if i%50==0:
        print(i)
        
        
nnDF2 = pd.DataFrame(nnList2)
nnDF2.columns=['node','neighbor','distance','similarity']


colors = cm.rainbow(np.linspace(0, 1, 8))

## Replace 11 with6 and 100 with 7
## so 6 = >10 degree
## 100 = disjoint nodes
nnDF2['distance'].replace(11,6,inplace=True)
nnDF2['distance'].replace(100,7,inplace=True)


nnDF2.to_csv(similarityDumpFile,index=False)
plt.scatter(nnDF2['distance'],nnDF2['similarity'])

plt.hist(nnDF2[nnDF2['distance']==1]['similarity'], label='1')
plt.title('distance:1')
plt.hist(nnDF2[nnDF2['distance']==7]['similarity'])
plt.title('unrelated')

plt.hist(nnDF2[nnDF2['distance']==2]['similarity'])
plt.title('distance:2')




plt.subplot(3, 3, 1)
plt.hist(nnDF2[nnDF2['distance']==1]['similarity'], label='1')
plt.title('distance:1')

plt.subplot(3, 3, 2)
plt.hist(nnDF2[nnDF2['distance']==2]['similarity'], label='1')
plt.title('distance:2')


plt.subplot(3, 3, 3)
plt.hist(nnDF2[nnDF2['distance']==3]['similarity'], label='1')
plt.title('distance:3')

plt.subplot(3, 3, 7)
plt.hist(nnDF2[nnDF2['distance']==4]['similarity'], label='1')
plt.title('distance:4')

plt.subplot(3, 3, 8)
plt.hist(nnDF2[nnDF2['distance']==5]['similarity'], label='1')
plt.title('distance:5')

plt.subplot(3, 3, 9)
plt.hist(nnDF2[nnDF2['distance']==6]['similarity'], label='1')
plt.title('distance>10')

plt.savefig('../data/nodesAsFeatures/cosine_similarity.png')


nnDF2['absSim']=nnDF2['similarity'].abs()

q = nnDF2.groupby('distance')['similarity'].quantile([0.1,0.25,0.5,0.75,0.9,\
                 0.95]).reset_index().pivot('distance','level_1','similarity')

qAbs = nnDF2.groupby('distance')['absSim'].quantile([0.1,0.25,0.5,0.75,0.9,\
                    0.95]).reset_index().pivot('distance','level_1','absSim')

    
nnDF2.groupby('distance')['similarity'].mean()
nnDF2['distance'].value_counts()




