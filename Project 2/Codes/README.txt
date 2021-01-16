KMeans

1) Run the code using "python3 kmeans.py" command
2)enter the parameters like k, list of centroids file location,max iterations
3)After entering the required inputs, the results are visualized
4)The jaccard coefficient and rand index values are printed on the console

Spectral

1) Run the code using "python3 spectral.py" command
2)enter the parameters like k, list of centroids file location,max iterations,sigma
3)After entering the required inputs, the results are visualized
4)The jaccard coefficient and rand index values are printed on the console

Gaussian Mixture Model
1) Run the code using the command "python GMM.py --filepath <path_to_file>"
2) The command line asks for the inputs - number of clusters, maximum iterations, mu, Sigma, Pi, convergence threshold, smoothing value.
3) Enter all parameters one by one. For Mu, Sigma and Pi enter the values as a list, e.g.
Mu: [[0 , 0] , [1 , 1]]
Sigma: [[[1 , 1],[1 , 1]],[[2 , 2],[2 , 2]]]
4) The Jaccard and rand index are printed along with the chart

Hirarchical Clustering:

•	Place the datasets iyer.txt, cho.txt and the hac.py file in the same directory.
•	Run the python file using the command python Project2_HierarchicalClustering.ipynb.py
•	Prompts you to enter the filename with extension, and the number of clusters.
•	After entering the required inputs, the results are visualized and the plots are displayed where you can identify the clusters

•	Once you close the plot, The jaccard coefficient and rand index values are printed

Density based clustering:

•	Place the datasets iyer.txt, cho.txt and the hac.py file in the same directory.
•	python Project2_Density-basedClustering.py
•	Prompts you to enter the filename with extension, and the Epsillon value,minimum points.
•	After entering the required inputs, the results are visualized and the plots are displayed.
•	The algorithm runs until the points are either put into cluster or marked as noise
•	Once you close the plot, The jaccard coefficient and rand index values are printed
