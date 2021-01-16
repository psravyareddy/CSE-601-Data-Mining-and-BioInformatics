{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "K5eqaIJ1XCBR"
   },
   "source": [
    "# **CSE-601 Project Hierarchical Agglomerative clustering with Min approach**\n",
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
    "from sklearn.decomposition import PCA\n",
    "import matplotlib.cm as cm\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import random\n",
    "import sys"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
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
   "execution_count": 3,
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
    "    return result\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def findCluster(x, par):\n",
    "    if (par[x] == -100):\n",
    "        return x\n",
    "    else:\n",
    "        par[x]=findCluster(par[x],par)\n",
    "        return par[x]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mergeCluster(x, y, par):\n",
    "    #print(\"{}->{}\".format(findCluster(y, par)+1,findCluster(x, par)+1))\n",
    "    par[findCluster(y, par)] = findCluster(x, par)\n",
    "    return\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
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
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def HirerchicalClustering(num_iterations):\n",
    "    \n",
    "    for i in range(num_iterations):\n",
    "        m_distance = float('inf')\n",
    "        m_xcoord = 0\n",
    "        m_ycoord = 0\n",
    "        for j in range(1, num_rows):\n",
    "            for k in range(j):\n",
    "                par_j = findCluster(j, par)\n",
    "                par_k = findCluster(k, par)\n",
    "                if(par_j == par_k):\n",
    "                    continue; \n",
    "                elif((distancematrix[par_j][par_k] < m_distance)):\n",
    "                        m_distance = distancematrix[par_j][par_k]\n",
    "                        m_xcoord = par_j\n",
    "                        m_ycoord = par_k\n",
    "                else:\n",
    "                      continue;\n",
    "\n",
    "        mergeCluster(m_xcoord, m_ycoord, par)\n",
    "\n",
    "        for j in range(0, num_rows):\n",
    "            cur = findCluster(j, par)\n",
    "            if((cur == m_xcoord)):\n",
    "                continue;\n",
    "            else:\n",
    "                minimumDistance=min(distancematrix[cur][m_xcoord], distancematrix[cur][m_ycoord])\n",
    "                distancematrix[cur][m_xcoord] = minimumDistance\n",
    "                distancematrix[m_xcoord][cur] = distancematrix[cur][m_xcoord]\n",
    "\n"
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
    "    plt.title(\"Hierarchical Agglomerative clustering using PCA for visualisation \"+input_file,fontweight=\"bold\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "input_file = input(\"Enter file name: \")\n",
    "num_clusters = int(input(\"Enter number of clusters: \"))\n",
    "with open(input_file) as t:\n",
    "    lines=[line.split() for line in t]\n",
    "filedata = np.asarray(lines)\n",
    "data = filedata[:,2:]\n",
    "truelabels = filedata[:,1]\n",
    "truelabels = np.reshape(truelabels, (len(truelabels),1))\n",
    "num_rows=data.shape[0]\n",
    "c_id = np.zeros((num_rows, 1))\n",
    "num_columns=data.shape[1]\n",
    "par = [-100]*len(data)\n",
    "distancematrix=compute_distancematrix(data)\n",
    "num_iterations = num_rows - num_clusters\n",
    "HirerchicalClustering(num_iterations)\n",
    "c_id_dict = {}\n",
    "\n",
    "count = 0\n",
    "for i in range(len(par)):\n",
    "    if (par[i] == -100):\n",
    "        c_id_dict[i] = count\n",
    "        count += 1\n",
    "\n",
    "for i in range(num_rows):\n",
    "    c_id[i][0] =  c_id_dict[findCluster(i, par)]\n",
    "\n",
    "\n",
    "points= np.matrix(data)\n",
    "labels = []\n",
    "for i in range(len(c_id)):\n",
    "    labels.append(c_id[i][0]+1)\n",
    "\n",
    "jaccard,rand=calculateJacRand(points,truelabels,labels)\n",
    "\n",
    "plotPCA(points,labels)\n",
    "\n",
    "print(\"Jaccard Coefficient=\",jaccard)\n",
    "print(\"Rand Index = \",rand)\n"
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
