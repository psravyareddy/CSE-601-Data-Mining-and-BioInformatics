"""
Created on Thu Nov 26 18:04:03 2020

@author: Shravya Pentaparthi
"""
#  C:\Users\svelpur\Desktop\Data Mining\Project3-Data\project3_dataset2.txt

import pandas as pd
import math
import numpy as np
from random import seed
from random import randrange

def compute_lables_NaiveBayes(df,testdata):
        # print(type(df))
        # print(type(testdata))
        # #df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/datamining project-3/project3_dataset2.txt", sep='\t', header=None)
        label = df[df.columns[-1]]
        # print(label)
        features = df.iloc[:,:-1]
        grouped = df.groupby(df[df.columns[-1]])
        label_0 = grouped.get_group(0)
        # print("label_0 is :\n",label_0)
        label_1 = grouped.get_group(1)
        # print("label_1 is :\n",label_1)
        std_1 = label_1.std(axis = 0)
        mean_1 = label_1.mean(axis = 0)
        std_0 = label_0.std(axis=0)
        mean_0 = label_0.mean(axis=0)
        p_0 =  len(label_0.index)/len(df.index)
        p_1 = len(label_1.index)/len(df.index)
        # print(p_0,"p0")
        # print(p_1,"p1")
        # test = df.iloc[[12],:-1]
        # print("test row is ",test)
        testdatapredictedlables = []
        for index, row in testdata.iterrows():
            test = testdata.iloc[[index]]  
            prod0 = []
            prod1 = []          
            for i,val in enumerate(test):
                        if test[i].dtype==float or test[i].dtype==int :
                            # print("inside first loop :",test[i])
                            num0 = math.exp(-((test[i]-mean_0[i])**2/std_0[i]**2))
                            den0 = (44/7*std_0[i]**2)**.5
                            prob0 = num0/den0
                            num1 = math.exp(-((test[i]-mean_1[i])**2/std_1[i]**2))
                            den1 = (44/7*std_1[i]**2)**.5
                            prob1 = num1/den1
                            prod0.append(prob0)
                            prod1.append(prob1)
                        else:
                            # print("inside second loop :",test[i])
                            if len(pd.unique(features.iloc[:,i])) == len(pd.unique(label_0.iloc[:,i])):  
                                prob0= label_0.iloc[:,i].value_counts()[test[i]][0]/len(label_0.index)
                            else:
                                prob0= (1+(label_0.iloc[:,i].value_counts()[test[i]][0]))/(len(label_0.index)+len(pd.unique(features.iloc[:,i])))
                            if len(pd.unique(features.iloc[:,i])) == len(pd.unique(label_1.iloc[:,i])):
                                prob1= label_1.iloc[:,i].value_counts()[test[i]][0]/len(label_1.index)
                            else:
                                prob1= (1+(label_1.iloc[:,i].value_counts()[test[i]][0]))/(len(label_1.index)+len(pd.unique(features.iloc[:,i])))
                            prod1.append(prob1)
                            prod0.append(prob0)
            # print((prod0))
            # print((prod1))
            prob0=np.prod(prod0)*p_0
            prob1=np.prod(prod1)*p_1
            # print("prob0 is :",prob0)
            # print("prob1 is :",prob1)
            if prob0>prob1:
                    testdatapredictedlables.append(0)
            else:
                    testdatapredictedlables.append(1)
        return testdatapredictedlables
    
def performance_metric(test_actual_version, test_predicted_version):

    #samples classification
    true_pos = 0
    false_neg =0
    false_pos=0
    true_neg = 0

    # to identify the performance metrics
    acc = 0
    prec = 0
    recall = 0
    f1_measure = 0

    for i in range(len(test_predicted_version)):
        if test_actual_version[i] == 1 and test_predicted_version[i] == 1:
            true_pos += 1
        elif test_actual_version[i] == 1 and test_predicted_version[i] == 0:
            false_neg += 1
        elif test_actual_version[i] == 0 and test_predicted_version[i] == 1:
            false_pos += 1
        elif test_actual_version[i] == 0 and test_predicted_version[i] == 0:
            true_neg += 1

    acc += (float(true_pos+true_neg)/(true_pos+false_neg+false_pos+true_neg))

    if(true_pos+false_pos != 0):
        prec += (float(true_pos)/(true_pos+false_pos))

    if(true_pos+false_neg != 0):
        recall += (float(true_pos)/(true_pos+false_neg))

    f1_measure += (float(2*true_pos)/((2*true_pos)+false_neg+false_pos))

    return acc, prec, recall, f1_measure


def cross_validation_split(dataset, folds=3):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / folds)
        for i in range(folds):
        	fold = list()
        	while len(fold) < fold_size:
        		index = randrange(len(dataset_copy))
        		fold.append(dataset_copy.pop(index))
        	dataset_split.append(fold)
        return np.asarray(dataset_split)

# "/content/drive/MyDrive/Colab Notebooks/datamining project-3/project3_dataset1.txt"
file_name = input("enter file path :")
k = int(input("enter value of k :"))
df = pd.read_csv(file_name , sep = '\t' , header  =  None)
# print("dataframe is :\n",df)
# print(len(df))
# print(df,"\n\n")
dataset_split  =  cross_validation_split(df.to_numpy(), 10)
# print(type(dataset_split))
dataset_split = dataset_split.tolist()
# for i in range (len(dataset_split)) :
#     print("i value is :",i,"\n",dataset_split[i],"\n\nlength is :",len(dataset_split[i]),"\n\n\n\n\n\n\n")
res = []
acc_sum = 0 
prec_sum = 0
recall_sum = 0
f1_measure_sum = 0 
for i in range(len(dataset_split)):
    train = dataset_split[:i] + dataset_split[i+1:]
    test = dataset_split[i]
    test =  pd.DataFrame(test)
    df_train =  pd.DataFrame()
    for j in range(len(train)):
        t =  pd.DataFrame(train[j])
        df_train = df_train.append(t)
    # print("type of train",type(df_train))
    # print("type of test",type(test.iloc[:,:-1]))
    computed_labels = compute_lables_NaiveBayes(df_train,test.iloc[:,:-1])
    acc, prec, recall, f1_measure = performance_metric(computed_labels,test[test.columns[-1]])
    acc_sum = acc_sum+acc
    prec_sum = prec_sum+prec
    recall_sum =recall_sum+recall
    f1_measure_sum = f1_measure_sum+f1_measure
print("the mean values of metrics are :",acc_sum/(k)," ",prec_sum/(k)," ",recall_sum/(k)," ",f1_measure_sum/(k))
computed_labels = compute_lables_NaiveBayes(df,df.iloc[:,:-1])
acc, prec, recall, f1_measure = performance_metric(computed_labels,df[df.columns[-1]])
print(" accuracy :",acc," precision :",prec," recall :",recall, " f1_measure :",f1_measure)


