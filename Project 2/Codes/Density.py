{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "K5eqaIJ1XCBR"
   },
   "source": [
    "# **CSE-601 Project Density-Based Clustering**\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import math\n",
    "import matplotlib.cm as cm\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import random\n",
    "import sys\n",
    "from sklearn.decomposition import PCA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def neighbours(i):\n",
    "    \n",
    "    nearbypoints=[]\n",
    "    \n",
    "    for points in range(0, num_rows):\n",
    "        \n",
    "        distance=distancematrix[i][points]\n",
    "        if distance <= epsilon:\n",
    "            nearbypoints.append(points);\n",
    "        else:\n",
    "            continue;\n",
    "    \n",
    "    return nearbypoints"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def dbscan(data,epsilon,min_points):\n",
    "    cluster=0\n",
    "\n",
    "    for i in range(len(data)):\n",
    "\n",
    "        if(visited[i]==0):\n",
    "            visited[i]=1\n",
    "            neighbour_pts = neighbours(i)\n",
    "            if(len(neighbour_pts)>min_points):\n",
    "                cluster=cluster+1;\n",
    "                finalCluster[i]=cluster;\n",
    "                expandCluster(i,neighbour_pts,cluster)\n",
    "            else:\n",
    "                finalCluster[i]=0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def expandCluster(i,neighbour_pts,cluster):\n",
    "     i=0\n",
    "     #print(\"com expand cluster\")\n",
    "     while i < len(neighbour_pts):\n",
    "         #print(i)\n",
    "         \n",
    "        if(visited[neighbour_pts[i]]==0):\n",
    "            visited[neighbour_pts[i]]=1\n",
    "            new_nearbypoints = neighbours(neighbour_pts[i])\n",
    "            if(len(new_nearbypoints) >= minimumpoints):\n",
    "                 neighbour_pts=neighbour_pts+new_nearbypoints\n",
    "            if(finalCluster[neighbour_pts[i]]==0):\n",
    "                 finalCluster[neighbour_pts[i]]=cluster;\n",
    "        i=i+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def euclidean_distance(one,two):\n",
    "    size = len(one)\n",
    "    result = 0.0\n",
    "    for i in range(size):\n",
    "        f1 = float(one[i])   \n",
    "        f2 = float(two[i])   \n",
    "        tmp = f1 - f2\n",
    "        result += pow(tmp, 2)\n",
    "    result = math.sqrt(result)\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_distancematrix(dataset):\n",
    "    result = np.zeros((num_rows,num_rows))\n",
    "    dataset_size = len(dataset)\n",
    "    for i in range(dataset_size-1):   \n",
    "        for j in range(i+1, dataset_size):    \n",
    "            dist = euclidean_distance(dataset[i], dataset[j])\n",
    "            result[i][j]=dist\n",
    "    result = result + result.T - np.diag(np.diag(result))\n",
    "\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculateJacRand(points,ground_truth,labels):\n",
    "    \n",
    "    tp = 0\n",
    "    tn = 0\n",
    "    fp=0\n",
    "    fn=0\n",
    "    for i in range(len(data)):\n",
    "        for j in range(len(data)):\n",
    "            if ground_truth[i]==ground_truth[j]:\n",
    "                if labels[i]==labels[j]:\n",
    "                    tp=tp+1\n",
    "                else:\n",
    "                    fn=fn+1\n",
    "            elif ground_truth[i]!=ground_truth[j]:\n",
    "                if labels[i]==labels[j]:\n",
    "                    fp=fp+1\n",
    "                else:\n",
    "                    tn=tn+1\n",
    "    jaccard_value=(tp)/(tp+fp+fn)\n",
    "    rand_index_value=(tp+tn)/(tp+tn+fp+fn)\n",
    "    return jaccard_value,rand_index_value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plotPCA(points,labels):\n",
    "    \n",
    "    pca_matrix = PCA(n_components=2).fit_transform(points)\n",
    "    plot_unique_labels = list(set(labels))\n",
    "    unique_naming_list_1=[]\n",
    "\n",
    "    colours_unique_vector = cm.Set1(np.linspace(0, 1, len(plot_unique_labels)))\n",
    "\n",
    "    for i in range(len(plot_unique_labels)):\n",
    "        dis_rows_index = np.where(labels==plot_unique_labels[i])\n",
    "        dis_rows = pca_matrix[dis_rows_index]\n",
    "        x_plot =[dis_rows[:,0]]\n",
    "        y_plot = [dis_rows[:,1]]\n",
    "        unique_naming_list_1.append(plt.scatter(x_plot, y_plot, c=colours_unique_vector[i]))\n",
    "\n",
    "        plt.scatter(x_plot,y_plot,c=colours_unique_vector[i])\n",
    "    plot_unique_labels=[-1.0 if x==0 else x for x in plot_unique_labels]\n",
    "    plot_unique_labels=np.array(plot_unique_labels,dtype=int)\n",
    "    plt.legend(unique_naming_list_1,plot_unique_labels,loc=\"best\",ncol=1,markerfirst=True,shadow=True)\n",
    "    plt.xlabel(\"PC 1\")\n",
    "    plt.ylabel(\"PC 2\")\n",
    "    plt.title(\"Density based clustering using PCA for visualisation \"+filelocation,fontweight=\"bold\")\n",
    "    plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "filelocation = input(\"Enter file name: \")\n",
    "\n",
    "with open(filelocation) as textFile:\n",
    "    lines=[line.split() for line in textFile]\n",
    "filedata = np.asarray(lines)\n",
    "data = filedata[:,2:]\n",
    "truelabels = filedata[:,1]\n",
    "truelabels = np.reshape(truelabels, (len(truelabels),1))\n",
    "epsilon = float(input('Enter epsilon(radius) value: '))\n",
    "minimumpoints = int(input('Enter the minimum number of points: '))\n",
    "num_rows=data.shape[0]\n",
    "num_columns=data.shape[1]\n",
    "visited=np.zeros(num_rows,dtype=int)\n",
    "finalCluster =np.zeros(num_rows,dtype=int)\n",
    "distancematrix=compute_distancematrix(data)\n",
    "\n",
    "dbscan(data,epsilon,minimumpoints)\n",
    "\n",
    "#print(finalCluster)\n",
    "points= np.matrix(filedata[:,2:],dtype=float,copy=False)\n",
    "jaccard_value,rand_index_value=calculateJacRand(points,truelabels,finalCluster)\n",
    "plotPCA(points,finalCluster)\n",
    "print(\"Jaccard Coefficient=\",jaccard_value)\n",
    "print(\"Rand Index = \",rand_index_value)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "name": "Project2_Recitation.ipynb",
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
