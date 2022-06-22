#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''

import numpy as np
import sklearn.datasets
import math
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def split_data(X, y, attribute_index, theta):
    """
    Splits the dataset (features and labels) into two subsets based on an attribute index 
    and on a split value for the specified attribute.

    :param X:the matrix containing the features
    :param y: the vector containing the labels
    :param attribute_index: the index of the attributed to be split by
    :param theta: the split value for the attribute 

    Returns:
    X1: all features with a value for the attribute smaller than the split value
    y1: all labels with a value for the attribute smaller than the split value
    X2: all features with a value for the attribute larger than or equal to the split value
    y2:all labels with a value for the attribute larger than or equal to the split value

    """

    X1=X[X[:,attribute_index]<theta]
    X2=X[X[:,attribute_index]>=theta]
    y1=y[X[:,attribute_index]<theta]
    y2=y[X[:,attribute_index]>=theta]

    #making sure the split is done correctly for both the features and the labels
    assert(X.shape[0]==(X1.shape[0]+X2.shape[0]))
    assert(y.shape[0]==(y1.shape[0]+y2.shape[0]))
    assert(X1.shape[0]==y1.shape[0])
    assert(X2.shape[0]==y2.shape[0])

    return X1, y1, X2, y2

def compute_information_content(y):

    """
    Calculates the information content of a subset X with labels y. 
    It is computed as - the sum over the m classes/labels of the 
    probability that an arbitrary tuple belongs to that class multiplied by
    logarithm in base 2 of the same probability. 

    :param y: the vector containing the labels

    Returns:
    info_content

    """
    #setting the number of classes 
    classes=3 

    #initialising information content
    info_content=0

    #iterating through the different classes and computing the info_content
    for i in range(classes):
        prob = len(y[y==i]) / len(y)
        if prob!=0:
            info = prob  *  np.log2(prob)
            info_content = info_content + info
    
    info_content = (-1) * info_content
    

    return info_content

def compute_information_a(X, y, attribute_index, theta):

    """
    Calculates the information content of A, on a subset X with labels y
    that is split according to the split defined by the pair(attribute_index, theta) 

    :param X:the matrix containing the features
    :param y: the vector containing the labels
    :param attribute_index: the index of the attributed to be split by
    :param theta: the split value for the attribute 

    Returns:
    info_content_a

    """
    #splitting the dataset according to the pair 
    X1, y1, X2, y2 = split_data(X, y, attribute_index, theta)

    #compute the information content for y1 and y2
    info_content_y1 = compute_information_content(y1)
    info_content_y2 = compute_information_content(y2)

    #initialising information content for a
    info_content_a=0

    #computing the information content for a
    info_content_a= (len(y1)/len(y)*info_content_y1) +(len(y2)/len(y)*info_content_y2)
    
    return info_content_a

def compute_information_gain(X, y, attribute_index, theta):

    """
    Calculates the information gain of the split 
    of an attribute for the given dataset.
    The split is given by the pair (attribute_index, theta).

    :param X:the matrix containing the features
    :param y: the vector containing the labels
    :param attribute_index: the index of the attributed to be split by
    :param theta: the split value for the attribute 

    Returns:
    info_gain

    """
    #computing the information gain

    info_gain=compute_information_content(y) - compute_information_a(X, y, attribute_index, theta)
    
    return info_gain




if __name__ == '__main__':

    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    target_names=iris.target_names
    num_features = len(set(feature_names))

   

    ####################################################################

    #calculating the information gain for the different attributes and theta values
    information_gain_sepl= compute_information_gain(X, y, 0, 5)
    information_gain_sepw= compute_information_gain(X, y, 1, 3)
    information_gain_petl= compute_information_gain(X, y, 2, 2.5)
    information_gain_petw= compute_information_gain(X, y, 3, 1.5)
    
    ####################################################################

    print('Exercise 2.b')
    print('------------')

    print("Split ( sepal length (cm) < 5.0) : information gain",'{0:.2f}'.format(information_gain_sepl))
    print("Split ( sepal width (cm) < 3.0) : information gain",'{0:.2f}'.format(information_gain_sepw))
    print("Split ( petal length (cm) < 2.5) : information gain",'{0:.2f}'.format(information_gain_petl))
    print("Split ( petal width (cm) < 1.5) : information gain",'{0:.2f}'.format(information_gain_petw))


    print('')

    print('Exercise 2.c')
    print('------------')

    print(" I would select ( petal length (cm) < 2.5) to be the first split because it has the highest information gain (0.92) \n and we are interested in maximising the information gain.")

    print('')

    print('Exercise 2.d')
    print('------------')

    ####################################################################
    # Exercise 2.d
    ####################################################################

    # Do _not_ remove this line because you will get different splits
    # which make your results different from the expected ones...
    np.random.seed(42)

    kf=KFold(n_splits=5, shuffle =True )
    accuracies_org=[]
    feature_importances_org=[[]]

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model=tree.DecisionTreeClassifier()
        model=model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        accuracies_org.append(accuracy_score(y_pred, y_test))
    
        feature_importances_org.append(model.feature_importances_)


    #getting the reduced dataset
    X=X[y!=2]
    y=y[y!=2]

    #performing classification on the reduced dataset
    kf_red=KFold(n_splits=5, shuffle =True )
    accuracies_red=[]
    feature_importances_red=[[]]

    for train_index, test_index in kf_red.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model=tree.DecisionTreeClassifier()
        model=model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        accuracy_red= accuracy_score(y_test, y_pred)
        accuracies_red.append(accuracy_red)
        
        feature_importances_red.append(model.feature_importances_)

    
    ##################### REPORTING RESULTS ########################

    print("The mean accuracy score is ",'{0:.2f}'.format(np.mean(accuracies_org)*100), '\n')

    print("For the original data the two most important features are: \n- petal length\n- petal width \n (see reported feature importances below). \n")
    print("For the reduced data set the most important feature is: \n- petal length \n This suggests that petal width is no longer required for classification and that the samples \n in the class that has been removed from the data set had values for \n this attribute (petal length) that are overlapping with the range of values of the \n classes left in the dataset. Now that there are two distinct ranges corresponding to \n the two classes, this attribute is sufficient for classification. This means that the \n petal length is not sufficient to classify the dataset when the thirs class is added. \n  ")






    print('Accuracy score using cross-validation')
    print('-------------------------------------\n')
    print('{0:.2f}'.format(np.mean(accuracies_org)*100))


    print('')
    print('Feature importances for _original_ data set')
    print('-------------------------------------------\n')

    print("Features in this order : sepal length, sepal width, petal length, petal width \n")
    for i in range(1,6):
        print("Fold ", i)
        print(feature_importances_org[i])


    print('')
    print('Feature importances for _reduced_ data set')
    print('------------------------------------------\n')
    print("Features in this order : sepal length, sepal width, petal length, petal width \n ")
    for i in range(1,6):
        print("Fold ", i)
        print(feature_importances_red[i])
