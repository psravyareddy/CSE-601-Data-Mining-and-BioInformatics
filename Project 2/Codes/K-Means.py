import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import random

file_path = input("enter file path :")
k = int(input("enter the number of clusters :"))
max_iterations = int(input("enter maximum iterations :"))
centroids_index = eval(input("enter the list of centroids [1,2,3,..]:"))
df = pd.read_csv(file_path, sep='\t', header=None)
gene_id = df[df.columns[0]]
ground_truth = df[df.columns[1]]
df = df.iloc[:, 2:]
rows = df
newCentroids = []
countloops = 0
clusterallocated = []
centroids = []
if len(centroids_index)==k:
   for i in centroids_index :
      centroids.append(rows.iloc[i-1].values.tolist())
else:
    print("first k values have been assigned as centroids")
    centroids = df.head(k)
    centroids=centroids.values.tolist()
while sorted(centroids) != sorted(newCentroids) and countloops < max_iterations:
    countloops = countloops+1
    clusters = []
    i = 0
    while i < k:
        clusters.append([])
        i += 1
    clusterallocated = []
    for index, row in rows.iterrows():
        dist = []
        for index_c, row_c in enumerate(centroids):
            dist.append(np.linalg.norm(row_c-row))
        minpos = dist.index(min(dist))
        clusterallocated.append(minpos+1)
        clusters[minpos].append(row.values.tolist())
    for index_c, cluster_c in enumerate(clusters):
        means = [round(float(sum(col))/len(col), 2) for col in zip(*cluster_c)]
        newCentroids.insert(index_c, means)
    if sorted(centroids) == sorted(newCentroids):
        print("oldcentroids==newcentroids")
        # return
    else:
        centroids = newCentroids
        newCentroids = []

tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(rows)):
    for j in range(len(rows)):
        if ground_truth[i] == ground_truth[j]:
            if clusterallocated[i] == clusterallocated[j]:
                tp = tp+1
            else:
                fn = fn+1
        elif ground_truth[i] != ground_truth[j]:
            if clusterallocated[i] == clusterallocated[j]:
                fp = fp+1
            else:
                tn = tn+1
jv = (tp)/(tp+fp+fn)
riv = (tp+tn)/(tp+tn+fp+fn)
print("jaccard value is ", jv)
print("random index value is ", riv)

def visualizepca(title, result):
    labels = result.Y.unique()
    nrof_labels = len(pd.unique(result['Y']))
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(nrof_labels)]

    fig, ax = plt.subplots(figsize=[15, 10])
    label_color = dict(zip(labels, color))
    label_set = set()
    for index, row in result.iterrows():
        if row['Y'] in label_set:
            ax.scatter(x=row['PCA1'], y=row['PCA2'],
                       color=label_color[row['Y']], s=75)
        else:
            label_set.add(row['Y'])
            ax.scatter(x=row['PCA1'], y=row['PCA2'],
                       color=label_color[row['Y']], label=row['Y'], s=75)

    plt.title(title)
    plt.legend()
    plt.show()
if(rows.shape[1] > 2):
    pca = PCA(n_components=2)
    dataforpca = np.matrix(rows)
    data_pca = pca.fit_transform(dataforpca)
    result = pd.DataFrame(list(data_pca[:, 0]), columns=['PCA1'])
    result['PCA2'] = list(data_pca[:, 1])
    result['Y'] = clusterallocated
    visualizepca('Clustering using K-Means', result)

