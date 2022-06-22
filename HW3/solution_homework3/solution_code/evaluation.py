"""
Homework : Evaluating classifiers
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
    The layout is the following:

                          yTrue
                   |  y = 1 | y = -1 |
            --------------------------
            y = 1  |   TP   |   FP   |
     yPred  --------------------------
            y = -1 |   FN   |   TN   |
            --------------------------
    '''
    
    # Create the confusion matrix
    mat = np.zeros((2, 2))
    # Get the unique elements of the array and iterate through them
    vec_elem = np.unique(y_true)
    for elem in vec_elem:
        idx = (y_true == elem)
        # Determine if it's TP or TN
        if elem > 0:
            # TP
            mat[0, 0] = sum(y_pred[idx] == elem)
            # FN
            mat[1, 0] = sum(y_pred[idx] != elem)
        else:
            # TN
            mat[1, 1] = sum(y_pred[idx] == elem)
            # FP
            mat[0, 1] = sum(y_pred[idx] != elem)

    return mat


def compute_precision(y_true, y_pred):
    """
    Function: compute_precision
    precision = TP / (TP + FP)
    Invoke confusion_matrix() to obtain the counts
    """
    mat = confusion_matrix(y_true, y_pred)
    # Divide by the sum of the first row
    return mat[0, 0] / mat.sum(axis = 1)[0]


def compute_recall(y_true, y_pred):   
    """
    Function: compute_recall
    recall = TP / (TP + FN)
    Invoke confusion_matrix() to obtain the counts
    """
    mat = confusion_matrix(y_true, y_pred)
    # Divide by the sum of the first column
    return mat[0, 0] / mat.sum(axis = 0)[0]


def compute_accuracy(y_true, y_pred):
    """
    Function: compute_accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    Invoke the confusion_matrix() to obtain the counts
    """
    mat = confusion_matrix(y_true, y_pred)
    # Divide by the total sum
    return (mat[0, 0] + mat[1, 1]) / mat.sum()

