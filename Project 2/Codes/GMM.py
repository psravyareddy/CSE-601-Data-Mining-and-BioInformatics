# -*- coding: utf-8 -*-
"""
Project 2 - Gaussian Mixture Model

Team members: Sai Hari Charan, Shravya Pentaparthi, Hemant Koti <br>

In this notebook, we will use the gaussian mixture model to find clusters of genes that exhibit similar expression behavior. <br>
"""

import random
import argparse
import traceback
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

pd.set_option('display.max_rows', 1000)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data Mining Project 1 - Dimensionality Reduction and Association Analysis.")
    parser.add_argument("--filepath", type=str, default="association-rule-test-data.txt",
                        help="Path to the association rules dataset")
    args = parser.parse_args()
    return args


class GMM:
    def __init__(self, k, max_iter=15, smoothing_value=0.0000001):
        self.k = k
        self.max_iter = max_iter
        self.smoothing_value = smoothing_value

    def initialize(self, X, mu=None, sigma=None, pi=None):
        self.shape = X.shape
        self.n, self.m = self.shape

        self.pi = [np.asarray(ele, dtype=float) for ele in pi] if pi else np.full(
            shape=self.k, fill_value=1/self.k)
        self.weights = np.full(shape=self.shape, fill_value=1/self.k)

        row_choice = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [np.asarray(ele, dtype=float) for ele in mu] if mu else [
            X[row_index, :] for row_index in row_choice]

        self.sigma = [np.asarray(ele, dtype=float) for ele in sigma] if sigma else [
            np.cov(X.T) for _ in range(self.k)]

        for i, ele in enumerate(self.sigma):
            np.fill_diagonal(
                self.sigma[i], ele.diagonal() + self.smoothing_value)

    def e_step(self, X):
        self.weights = self.predict_probability(X)
        self.pi = self.weights.mean(axis=0)
        # print('Pi value:', self.pi)

    def m_step(self, X):
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T,
                                   aweights=(weight/total_weight).flatten(),
                                   bias=True)
            for i, ele in enumerate(self.sigma):
                np.fill_diagonal(
                    self.sigma[i], ele.diagonal() + self.smoothing_value)

        # print('Mu value:', self.mu)
        #  print('Sigma value:', self.sigma)

    def fit(self, X, mu=None, sigma=None, pi=None, conv_threshold=0.00000001):
        self.initialize(X, mu, sigma, pi)
        prev_loss = None
        for iteration in range(self.max_iter):
            # print('Iteration: ', iteration)
            self.e_step(X)
            self.m_step(X)
            new_loss = self.calculate_loss(X)
            if prev_loss != None and abs(new_loss - prev_loss) <= conv_threshold:
                break
            prev_loss = new_loss

    def calculate_loss(self, X):
        N = X.shape[0]
        C = self.weights.shape[1]
        self.loss = np.zeros((N, C))

        for c in range(C):
            dist = multivariate_normal(
                self.mu[c], self.sigma[c], allow_singular=True)
            self.loss[:, c] = self.weights[:, c] * (np.log(
                self.pi[c]+0.00000001)+dist.logpdf(X)-np.log(self.weights[:, c]+0.000000001))
        self.loss = np.sum(self.loss)
        return self.loss

    def predict_probability(self, X):
        likelihood = np.zeros((self.n, self.k))
        for i in range(self.k):
            distribution = multivariate_normal(
                mean=self.mu[i],
                cov=self.sigma[i])
            likelihood[:, i] = distribution.pdf(X)

        numerator = likelihood * self.pi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    def predict(self, X):
        return np.argmax(self.predict_probability(X), axis=1)


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


def jaccardRand(df_gene, ground_truth, clusterallocated):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(df_gene)):
        for j in range(len(df_gene)):
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
    return jv, riv


def main():
    try:

        args = parse_args()

        df_gene = pd.read_csv(args.filepath, sep='\t', header=None)

        # Gene dataset
        print(df_gene.shape)
        print(df_gene.info())
        print()

        print('Unique Genes: ', df_gene[1].unique())
        print()

        df_gene_array = df_gene.to_numpy()
        gene_id = df_gene_array[:, 0]
        ground_truth = df_gene_array[:, 1]

        attributes = np.delete(df_gene_array, np.s_[0:2], axis=1)
        dim = np.shape(df_gene_array)[1] - 2
        n_data = np.shape(df_gene_array)[0]

        clusters = int(input("Enter the number of clusters: "))
        max_iters = int(input("Enter the maximum number of iterations: "))

        convergence_threshold = float(
            input("Enter the convergence threshold: "))
        smoothing_value = float(input("Enter the smoothing value: "))

        mu = eval(input("Enter mu value: "))
        sigma = eval(input("Enter sigma value: "))
        pi = eval(input("Enter pi value: "))

        mu = None if len(mu) == 0 else mu
        sigma = None if len(sigma) == 0 else sigma
        pi = None if (len(pi) == 0) else pi

        gmm = GMM(k=clusters, max_iter=max_iters,
                  smoothing_value=smoothing_value)
        gmm.fit(attributes, mu, sigma, pi, convergence_threshold)

        clusterallocated = gmm.predict(attributes)
        clusterallocated = np.asarray(clusterallocated, dtype=int)

        df_gene = df_gene.iloc[:, 2:]
        jaccard, rand_index = jaccardRand(
            df_gene, ground_truth, clusterallocated)
        print("Jaccard index value:", jaccard)
        print("Rand index value:", rand_index)

        data_pca = PCA(n_components=2).fit_transform(attributes)
        result = pd.DataFrame(list(data_pca[:, 0]), columns=['PCA1'])
        result['PCA2'] = list(data_pca[:, 1])
        result['Y'] = clusterallocated
        visualizepca('Clustering using GMM', result)

    except Exception as ex:
        print(traceback.print_exc())


if __name__ == "__main__":
    main()

"""
References
1. http://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/

Code
1. https://stackoverflow.com/a/12186422/6379722


"""
