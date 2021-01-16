
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 


input_file_Path  = input("enter file path :")
#input("enter file path :")
sigma = int(input("enter sigma value :"))
#int(input("enter sigma value :"))
k = int(input("enter the number of clusters :"))
#int(input("enter number of clusters :"))
max_iterations = int(input("enter the maximum number of iterations :"))
#int(input("enter the maximum number of iterations :"))
init = eval(input("enter the list of centroids [1,2,3,..]:"))
#eval(input("enter the list of centroids [1,2,3,..]:"))
filedata = np.loadtxt(input_file_Path, delimiter="\t")

 
data = filedata[:,2:]
print(type(data))
def ComputeSimilarity(features):
    r = len(features)
    res = np.zeros((r,r))
    for i in range(0,r):
        for j in range(0, r):
            diff = np.matrix(abs(np.subtract(features[i], features[j])))
            diff_squared = (np.square(diff).sum(axis=1))
            res[i][j] = np.exp(-(diff_squared)/(sigma**2))
    return res

def ComputeDegree(w):
    diag = np.array(w.sum(axis=1)).ravel()
    return np.diag(diag)

def ComputeLaplacian(w, d): 
    return d - w


w = ComputeSimilarity(data)
d = ComputeDegree(w)
l = ComputeLaplacian(w, d)


vals, vecs = np.linalg.eig(l)

vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]
maxi = 0
index = 0
for i in range(1,len(vals.tolist())):
  if(vals[i]-vals[i-1] > maxi):
      maxi = vals[i]-vals[i-1]
      index = i

data = np.array(vecs[:,0:index+1])
truelabels = filedata[:,1]
truelabels = np.reshape(truelabels, (len(truelabels),1))


if(len(init)==k):
    init=[i-1 for i in init]
    centroids= data[init,:]
else:
    print("Using random centroids.")
    perm=np.random.permutation(len(data))
    centroids= data[perm[0:k]]


y_pred = KMeans(n_clusters = k,init=centroids,max_iter=max_iterations).fit(data)
clusterallocated=[]
clusterallocated=y_pred.labels_
clusterallocated = [x+1 for x in clusterallocated]

tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(data)):
    for j in range(len(data)):
        if truelabels[i] == truelabels[j]:
            if clusterallocated[i] == clusterallocated[j]:
                tp = tp+1
            else:
                fn = fn+1
        elif truelabels[i] != truelabels[j]:
            if clusterallocated[i] == clusterallocated[j]:
                fp = fp+1
            else:
                tn = tn+1
jv = (tp)/(tp+fp+fn)
riv = (tp+tn)/(tp+tn+fp+fn)
print("jaccard value is ", jv)
print("random index value is ", riv)

data = filedata[:,2:]


labels = clusterallocated

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
if(data.shape[1] > 2):
    pca = PCA(n_components=2)
    dataforpca = np.matrix(data)
    data_pca = pca.fit_transform(dataforpca)
    result = pd.DataFrame(list(data_pca[:, 0]), columns=['PCA1'])
    result['PCA2'] = list(data_pca[:, 1])
    result['Y'] = clusterallocated
    visualizepca('Clustering using K-Means', result)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 


input_file_Path  = input("enter file path :")
#input("enter file path :")
sigma = int(input("enter sigma value :"))
#int(input("enter sigma value :"))
k = int(input("enter the number of clusters :"))
#int(input("enter number of clusters :"))
max_iterations = int(input("enter the maximum number of iterations :"))
#int(input("enter the maximum number of iterations :"))
init = eval(input("enter the list of centroids [1,2,3,..]:"))
#eval(input("enter the list of centroids [1,2,3,..]:"))
filedata = np.loadtxt(input_file_Path, delimiter="\t")

 
data = filedata[:,2:]
print(type(data))
def ComputeSimilarity(features):
    r = len(features)
    res = np.zeros((r,r))
    for i in range(0,r):
        for j in range(0, r):
            diff = np.matrix(abs(np.subtract(features[i], features[j])))
            diff_squared = (np.square(diff).sum(axis=1))
            res[i][j] = np.exp(-(diff_squared)/(sigma**2))
    return res

def ComputeDegree(w):
    diag = np.array(w.sum(axis=1)).ravel()
    return np.diag(diag)

def ComputeLaplacian(w, d): 
    return d - w


w = ComputeSimilarity(data)
d = ComputeDegree(w)
l = ComputeLaplacian(w, d)


vals, vecs = np.linalg.eig(l)

vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]
maxi = 0
index = 0
for i in range(1,len(vals.tolist())):
  if(vals[i]-vals[i-1] > maxi):
      maxi = vals[i]-vals[i-1]
      index = i

data = np.array(vecs[:,0:index+1])
truelabels = filedata[:,1]
truelabels = np.reshape(truelabels, (len(truelabels),1))


if(len(init)==k):
    init=[i-1 for i in init]
    centroids= data[init,:]
else:
    print("Using random centroids.")
    perm=np.random.permutation(len(data))
    centroids= data[perm[0:k]]


y_pred = KMeans(n_clusters = k,init=centroids,max_iter=max_iterations).fit(data)
clusterallocated=[]
clusterallocated=y_pred.labels_
clusterallocated = [x+1 for x in clusterallocated]

tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(data)):
    for j in range(len(data)):
        if truelabels[i] == truelabels[j]:
            if clusterallocated[i] == clusterallocated[j]:
                tp = tp+1
            else:
                fn = fn+1
        elif truelabels[i] != truelabels[j]:
            if clusterallocated[i] == clusterallocated[j]:
                fp = fp+1
            else:
                tn = tn+1
jv = (tp)/(tp+fp+fn)
riv = (tp+tn)/(tp+tn+fp+fn)
print("jaccard value is ", jv)
print("random index value is ", riv)

data = filedata[:,2:]


labels = clusterallocated

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
if(data.shape[1] > 2):
    pca = PCA(n_components=2)
    dataforpca = np.matrix(data)
    data_pca = pca.fit_transform(dataforpca)
    result = pd.DataFrame(list(data_pca[:, 0]), columns=['PCA1'])
    result['PCA2'] = list(data_pca[:, 1])
    result['Y'] = clusterallocated
    visualizepca('Clustering using K-Means', result)

