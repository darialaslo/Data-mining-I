"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Main program for k-NN.
Predicts the labels of the test data using the training data.
The k-NN algorithm is executed for different values of k (user-entered parameter)


Original author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
Extended by: Bastian Rieck <bastian.rieck@bsse.ethz.ch>
"""

import argparse
import os
import sys
import numpy as np

# Import the file with the performance metrics 
import evaluation

# Class imports
from knn_classifier import KNNClassifier


# Constants
# 1. Files with the datapoints and class labels
DATA_FILE  = "matrix_mirna_input.txt"
PHENO_FILE = "phenotype.txt"

# 2. Classification performance metrics to compute
PERF_METRICS = ["accuracy", "precision", "recall"]


def load_data(dir_path): # (TO DO)
    """
    Function for loading the data.
    Receives the path to a directory that will contain the DATA_FILE and PHENO_FILE.
    Loads both files into memory as numpy arrays. Matches the patientId to make
    sure the class labels are correctly assigned.

    Returns
     X : a matrix with the data points
     y : a vector with the class labels
    """
    file_name_data= "{}/matrix_mirna_input.txt".format(dir_path)
    file_name_labels="{}/phenotype.txt".format(dir_path)

    with open(file_name_data, 'r') as f_in:
        # Create a dictionary of lists. Key to the dictionary is the group name
        dict_doc = {}
        for line in f_in:
            # Remove the trailing newline and separate the fields
            parts = line.rstrip().split("\t")

            # If the group does not exist in the dictionary, create it
            if not parts[0] in dict_doc and parts[0]!='patientId':
                # Use the group name as key. Initialize list
                dict_doc[parts[0]] = []
                dict_doc[parts[0]].append(parts[1:])
    
    with open(file_name_labels, 'r') as f_in:
        # Create a dictionary of lists. Key to the dictionary is the group name
        dict_doc_labels = {}
        for line in f_in:
            # Remove the trailing newline and separate the fields
            parts = line.rstrip().split("\t")

            # If the group does not exist in the dictionary, create it
            if not parts[0] in dict_doc_labels and parts[0]!='patientId':
                # Use the group name as key. Initialize list
                dict_doc_labels[parts[0]] = []
                #add values
                dict_doc_labels[parts[0]].append(parts[1])

    #extract values for X
    X= np.array([val for val in dict_doc.values()],dtype=float)
    X=np.reshape(X, (X.shape[0], X.shape[2]))
    
    #initialise vector for the labels 
    y=[]
    for key, val in dict_doc.items():
        
        if dict_doc_labels[key]==['+']:
            y.append(1)
        else:
            y.append(0)
    


    return X, y


def obtain_performance_metrics(y_true, y_pred): # (TO DO)
    """
    Function obtain_performance_metrics
    Receives two numpy arrays with the true and predicted labels.
    Computes all classification performance metrics.
    
    In this function you might call the functions:
    compute_accuracy(), compute_precision(), compute_recall()
    from the evaluation.py file. You can call them by writing:
    evaluation.compute_accuracy, and similarly.

    Returns a vector with one value per metric. The positions in the
    vector match the metric names in PERF_METRICS.
    """

    #calculate metrics using evaluation.py
    accuracy=evaluation.compute_accuracy(y_true, y_pred)
    precision= evaluation.compute_precision(y_true, y_pred)
    recall = evaluation.compute_recall (y_true, y_pred)

    return np.array([accuracy, precision, recall])



#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(description="Compute distance functions on vectors")
    parser.add_argument("--traindir", required=True, 
                        help="Path to the directory containing the input training data")
    parser.add_argument("--testdir", required=True,
                        default="file_info.txt",
                        help="Path to the directory containing the test data")
    parser.add_argument("--outdir", required=True, 
                        help="Path to the output directory, where the output file will be created")
    parser.add_argument("--mink", required=True, 
                        help="The minimum value of k on which k-NN algorithm will be invoked")
    parser.add_argument("--maxk", required=True, 
                        help="The maximum value of k on which to run k-NN. This parameter,\
                            is used to run the algorithm for multiple values of k")
    args = parser.parse_args()


    # If the output directory does not exist, then create it
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    
    # Read the training and test data. For each dataset, get also the true labels.
    # Use the function load_data().
    # Important: Match the patientId between data points and class labels
    
    X_train, y_true_train=load_data(args.traindir)
    X_test, y_true_test=load_data(args.testdir)



    # Create the output file & write the header as specified in the homework sheet
    try:
        file_name = "{}/output_knn.txt".format(args.outdir)
        f_out=open(file_name, 'w') 
    #raise exception if the file can't be created
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

    #write header 
    f_out.write("{}\t{}\t{}\t{}\n".format('Value of k',PERF_METRICS[0],\
            PERF_METRICS[1], PERF_METRICS[2]))
        

    ############################## KNN algorithm ####################################

    # Create the k-NN object. (Hint about how to do it in the homework sheet)

    knn = KNNClassifier(X_train, y_true_train, metric = 'euclidean')

    # Iterate through all possible values of k:
    # HINT: remember to set the number of neighbors for the KNN object through: knn_obj.set_k(k)

    # 1. Perform KNN training and classify all the test points. In this step, you will
    # obtain a prediction for each test point. 

    k_min=args.mink
    k_max=args.maxk

    for k in range(int(k_min), int(k_max)+1):

        knn.set_k(k)
        y_pred=[]
        
        for i in range(X_test.shape[0]):
            x=X_test[i,:]
            y=knn.predict(x)
            y_pred.append(y)
        
            # 2. Compute performance metrics given the true-labels vector and the predicted-
            # labels vector (you might consider to use obtain_performance_metrics() function)

        METRICS= obtain_performance_metrics(y_true_test,y_pred)

        # Transform the vector of metrics to a string
        str_metrics = "\t".join("{0:.2f}".format(m) for m in METRICS)


            # 3. Write performance results in the output file, as indicated the in homework
            # sheet. Close the file.

        f_out.write(
            '{}\t{}\n'.format(k, str_metrics)
        )
    f_out.close()

