import os
import numpy as np


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

    HINT: for using the attributes of the class in the class' methods, you can use: self.attribute_name
          (e.g. self.X for accessing the value of the attribute named X)
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
        Private function to compute distances. 
        Compute the distance between x and all points in X
    
        Parameters
        ----------
        x : a vector (data point)
        '''
        distances=[]

        for i in range(X.shape[0]):
            distance=np.sqrt(sum((x-X[i,:])**2))
            distances.append(distance)

        return distances


    def predict(self, x):
        '''
        Method to predict the label of one data point.
        Here you actually code the KNN algorithm.
       
        Hint: for calling the method _compute_distance 
              (which is private), you can use: 
              self._compute_distances(self.X, x) 
        
        Parameters
        ----------
        x : Vector from the test data.
        '''
        y=self.y
        y=np.reshape(y, (len(y),1))

        distances=self._compute_distances(self.X, x)
        distances=np.reshape(distances, (len(distances),1))
        neighbours=np.concatenate((distances, y), axis=1)
  
        neighbours_sorted = neighbours[neighbours[:,0].argsort()]

        k=self.k
        neg=0
        pos=0
        for i in neighbours_sorted[:k,1]:
            if(i==0):
                neg+=1
            else:
                pos+=1
        if(neg>pos):
            y_pred=0
        elif(pos>neg):
            y_pred=1
        elif(pos==neg):
            if(neighbours_sorted[k-1,1]==0):
                y_pred=1
            elif(neighbours_sorted[k-1,1]==1):
                y_pred=0
           
    
        return y_pred

