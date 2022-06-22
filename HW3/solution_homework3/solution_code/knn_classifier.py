import os
import numpy as np
# Compute all-pair distances using cdist()
from scipy.spatial import distance
# Count number of elements with each label among k-nearest neighbors
import collections

class KNNClassifier:
    '''
    A class object that implements the methods of a k-Nearest Neighbor classifier
    The class assumes there are only two labels, namely POS and NEG

    Attributes of the class
    -----------------------
    k : Number of neighbors
    X : A matrix containing the data points (train set)
    y : A vector with the labels
    dist : Distance metric used. Possible values are: 'euclidean', 'hamming', 'minkowski', and others
           For a full list of possible metrics have a look at:
           http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    '''     
    def __init__(self, X, y, metric):
        '''
        Constructor when X and Y are given.
        
        Parameters
        ----------
        X : Matrix with data points
        Y : Vector with class labels
        metric : Name of the distance metric to use
        '''
        # Default values
        self.verbose = False
        self.k = 1

        # Parameters
        self.X = X
        self.y = y
        self.metric = metric


    def debug(self, switch):
        '''
        Method to set the debug mode.
        
        Parameters
        ----------
        switch : String with value 'on' or 'off'
        '''
        self.verbose = True if switch == "on" else False


    def set_k(self, k):
        '''
        Method to set the value of k.
        
        Parameters
        ----------
        k : Number of nearest neighbors
        '''
        self.k = k


    def _compute_distances(self, X, x):
        '''
        Private function to compute distances. Invokes distance function from SciPy.
        Each row of X is a data point in the training set. 
        x is one data point in the test set. 
        Compute the distance between x and all rows in X
    
        Parameters
        ----------
        x : a vector (data point)
        '''
        return distance.cdist(X, x, self.metric)


    def predict(self, x):
        '''
        Method to predict the label of one data point.
        
        Parameters
        ----------
        x : Vector from the test data.
        '''
        # Obtain the two labels. Get the unique elements of y
        classes = np.unique(self.y)
        # The positive class will be the largest label
        POS = max(classes)
        NEG = min(classes)

        # Compute the distance between x and all rows in X
        dist = self._compute_distances(self.X, x[np.newaxis, :])

        # Sort the distances in ascending order. Get sorted indices.
        # Note: The return of compute_distances requires indexing by the first column
        idx = np.argsort(dist[:, 0])

        # Determine the predicted label using the top k (shortest) distances
        # Apply 2-level indexing
        counter_class = collections.Counter(self.y[idx[:self.k]])

        if self.verbose:
            print("%d-nearest neighbors" % self.k)
            print("Row num.: ", idx[range(0, self.k)])
            print("Labels  : ", self.y[idx[range(0, self.k)]])
            print("Count of classes")
            print("%s : %d" % (POS, counter_class[POS]))
            print("%s : %d" % (NEG, counter_class[NEG]))

        # Determine if there's a tie
        if counter_class[POS] == counter_class[NEG]:
            if self.verbose:
                print("\tTie!")
            # Break the tie. Use k - 1. Note: this is valid only for a binary classification problem
            counter_class = collections.Counter(self.y[idx[range(0, self.k - 1)]])
        
        # Return the label that has the largest count
        return POS if counter_class[POS] > counter_class[NEG] else NEG

