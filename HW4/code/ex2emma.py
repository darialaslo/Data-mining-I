#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''

import numpy as np
import math
import sklearn.datasets


# Splits X matrix and y vector into subsets based on split specified by attribute_index and theta
def split_data(X, y, attribute_index, theta):

    X1 = X[X[:, attribute_index] < theta]
    X2 = X[X[:, attribute_index] >= theta]

    y1 = y[X[:, attribute_index] < theta]
    y2 = y[X[:, attribute_index] >= theta]

    return [X1, y1, X2, y2]


# Computes info content of a SUBSET of X that has labels y
def compute_information_content(y):
   

    #setting the number of classes 
    classes=3 

    #initialising information content
    info_content=0

    #iterating through the different classes and computing the info_content
    for i in range(classes):
        prob = len(y[y==i]) / len(y)
        info = prob  *  np.log2(prob)
        info_content = info_content + info
    
    info_content = (-1) * info_content
    

    return info_content


# Computes info_A of X for matrix X with labels y that is split according to attribute_index and theta
def compute_information_a(X, y, attribute_index, theta):
    info=0

    X1 = split_data(X, y, attribute_index, theta)[0]
    y1 = split_data(X, y, attribute_index, theta)[1]
    X2 = split_data(X, y, attribute_index, theta)[2]
    y2 = split_data(X, y, attribute_index, theta)[3]

    info_1=compute_information_content(y1)
    info_2=compute_information_content(y2)

    InfoA_1 = (len(y1) / len(y)) * info_1
    InfoA_2 = (len(y2) / len(y)) * info_2
    info=InfoA_1 + InfoA_2
    

    return info


# X = Iris data matrix
# y = label vector
# theta = split value for attribute specified by attribute_index
def compute_information_gain(X, y, attribute_index, theta):
    gain = compute_information_content(y) - compute_information_a(X, y, attribute_index, theta)
    return gain



if __name__ == '__main__':

    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    target_names = iris.target_names
    num_features = len(set(feature_names))

    ####################################################################

    ####################################################################

    print('Exercise 2.b')
    print('------------')
# attribute_index = index of attribute column (0, 1, 2 or 3)
    sepl = compute_information_gain(X, y, 0, 5.0)
    sepw = compute_information_gain(X, y, 1, 3.0)
    petl = compute_information_gain(X, y, 2, 2.5)
    petw = compute_information_gain(X, y, 3, 1.5)

    print('Split ( sepal length (cm) < 5.0 ): information gain = {:.2f}'.format(sepl))
    print('Split ( sepal width (cm) < 3.0 ): information gain = {:.2f}'.format(sepw))
    print('Split ( petal length (cm) < 2.5 ): information gain = {:.2f}'.format(petl))
    print('Split ( petal width (cm) < 1.5 ): information gain = {:.2f}'.format(petw))
    print('')


    print('Exercise 2.c')
    print('------------')

    print('')

    ####################################################################
    # Exercise 2.d
    ####################################################################

    # Do _not_ remove this line because you will get different splits
    # which make your results different from the expected ones...
    np.random.seed(42)

    print('Accuracy score using cross-validation')
    print('-------------------------------------\n')



    print('')
    print('Feature importances for _original_ data set')
    print('-------------------------------------------\n')


    print('')
    print('Feature importances for _reduced_ data set')
    print('------------------------------------------\n')
