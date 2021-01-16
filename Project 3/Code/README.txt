Naive Bayes
1) Keep the code and data file in the same directory.
2) Open the file NaiveBayes.ipynb in a Jupyter Notebook.
3) Enter the name of the input file.
4) Give the value of K for K-Fold Cross Validation as input from the command line.
5) The metrics i.e Accuracy, Precision, Recall, F1-Measure are displayed.

KNN
1) Keep the code and data file in the same directory.
2) Open the file K-NN.ipynb in a Jupyter Notebook.
3) Enter the name of the input file.
4) Give the value of K for K-Fold Cross Validation as input from the command line.
5) Give the value of K for No.of neighbors in KNN as input from the command line.
6) The metrics i.e Accuracy, Precision, Recall, F1-Measure are displayed.

Decision Tree
1) Keep the code and data file in the same directory.
2) Open the file DecisionTree.ipynb in a Jupyter Notebook.
3) Change the name of the input file path during read_csv() in the 4th cell.
4) The tree is displayed for the complete dataset.
4) The tree is displayed for the K-fold datasets.
6) The metrics i.e Accuracy, Precision, Recall, F1-Measure are displayed.

Random Forest
1) Keep the code and data file in the same directory.
2) Open the file RandomForest.ipynb in a Jupyter Notebook.
3) Change the name of the input file path during read_csv() in the 4th cell.
4) Change the number of trees (n_trees) as input when the K-fold cross-validation starts.
5) By default the number of features for split is taken as the square root of the number of features. This value can also be customized in the num_features_for_split variable when the K-fold cross-validation starts.
6) The metrics i.e Accuracy, Precision, Recall, F1-Measure are displayed.

Kaggle Competition
1) Keep the code and data file in the same directory.
2) Open the file KaggleRandomForest.ipynb in a Jupyter Notebook.
3) Change the name of the input file path during read_csv() for df_train_features, df_train_labels, df_test_features variables.
4) The optimal hyperparameters are chosen using the GridSearchCV algorithm and are used for the RandomForestClassifier class.
5) The predicted values for df_test_features is saved as output.csv.
6) The metrics i.e Accuracy, Precision, Recall, F1-Measure are displayed for the training dataset.