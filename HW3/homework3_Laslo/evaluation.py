"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the metrics that are invoked from the main program.

Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
Extended by: Bastian Rieck <bastian.rieck@bsse.ethz.ch>
"""

import numpy as np


def confusion_matrix(y_true, y_pred):
    '''
    Function for calculating TP, FP, TN, and FN.
    The input includes the vector of true labels
    and the vector of predicted labels
    '''
    tp=0
    fp=0
    tn=0
    fn=0

    for i in range(len(y_true)):
        if (y_true[i]==1):
            if(y_true[i]==y_pred[i]):
                tp+=1
            else:
                fn+=1
        
        else:
            if(y_true[i]==y_pred[i]):
                tn+=1
            else:
                fp+=1
        
    return np.array([[tp, fp], [fn, tn]])


def compute_precision(y_true, y_pred):
    """
    Function: compute_precision
    Invoke confusion_matrix() to obtain the counts
    """
    matrix=confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = (0,0), (0,1), (1,0), (1,1)
    
    return matrix[tp]/(matrix[tp]+matrix[fp])


def compute_recall(y_true, y_pred):
    """
    Function: compute_recall
    Invoke confusion_matrix() to obtain the counts
    """
    matrix=confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = (0,0), (0,1), (1,0), (1,1)

    return matrix[tp]/(matrix[tp]+matrix[fn])


def compute_accuracy(y_true, y_pred):
    """
    Function: compute_accuracy
    Invoke the confusion_matrix() to obtain the counts
    """
    matrix=confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = (0,0), (0,1), (1,0), (1,1)

    return (matrix[tp]+matrix[tn])/(matrix.sum())




