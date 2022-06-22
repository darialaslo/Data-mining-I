"""
Homework3 : Naive Bayes

Main program for Naive Bayes.
Computes the probabilities and predicts the labels for a certain entry.

Original author: Ana Daria Laslo
"""

import argparse
import os
import sys
import numpy as np



def get_prob_class(X, value, class_label): 
    """
    Function get_prob_benign  
    Receives the input matrix containing values for the different features in the first 4 columns \
        and the label in the fifth column (2 - benign, 4 - malign), one of the possible values for the features,
        in this case ranging from 1 to 10 and the calss label (2 - benign, 4 - malign)
    Computes the different probabilities of being in the class benign given a feature is equal to the input value. 
    The probabilities are calculated separately. 
    Returns a vector containing the probabilities for the 4 different classes. 
    """
  
    #calculate metrics using evaluation.py
    class_size=len(X[X[:,4]==class_label])

    clump_counts=np.count_nonzero(X[X[:,4]==class_label][:,0]==value)
    if clump_counts==0:
        clump=0
    else:
        #excluding NANs 
        clump_size=class_size-np.count_nonzero(np.isnan(X[X[:,4]==class_label][:,0]))
        clump=clump_counts/clump_size
    
    uniformity_counts=np.count_nonzero(X[X[:,4]==class_label][:,1]==value)
    if uniformity_counts==0:
        uniformity=0
    else:
        uniformity_size=class_size-np.count_nonzero(np.isnan(X[X[:,4]==class_label][:,1]))
        uniformity= uniformity_counts/uniformity_size

    marginal_counts=np.count_nonzero(X[X[:,4]==class_label][:,2]==value)
    if marginal_counts==0:
        marginal=0
    else:
        marginal_size=class_size-np.count_nonzero(np.isnan(X[X[:,4]==class_label][:,2]))
        marginal = marginal_counts/marginal_size

    mitoses_counts=np.count_nonzero(X[X[:,4]==class_label][:,3]==value)
    if mitoses_counts==0:
        mitoses=0
    else:
        mitoses_size=class_size-np.count_nonzero(np.isnan(X[X[:,4]==class_label][:,3]))
        mitoses = np.count_nonzero(X[X[:,4]==class_label][:,3]==value)/mitoses_size

    return np.array([clump, uniformity, marginal, mitoses])



#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(description="Compute distance functions on vectors")
    parser.add_argument("--traindir", required=True, 
                        help="Path to the directory containing the input training data")
    parser.add_argument("--outdir", required=True, 
                        help="Path to the output directory, where the output file will be created")
    args = parser.parse_args()

    # Load and read the training data.
    file_name_input="{}/tumor_info.txt".format(args.traindir)
    X_train=np.genfromtxt(file_name_input, delimiter='\t', dtype=float)


    # If the output directory does not exist, then create it
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    #Set class labels 
    labels=['2', '4']

    # Create the output file & write the header for all classes
    
    try:
        file_name = "{}/output_summary_class_{}.txt".format(args.outdir, labels[0])
        f_out_class1=open(file_name, 'w') 
    #raise exception if the file can't be created
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

    #write header 
    f_out_class1.write("{}\t{}\t{}\t{}\t{}\n".format('Value','Clump', 'Uniformity', 'Marginal', 'Mitoses'))

    try:
        file_name = "{}/output_summary_class_{}.txt".format(args.outdir, labels[1])
        f_out_class2=open(file_name, 'w') 
    #raise exception if the file can't be created
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

    #write header 
    f_out_class2.write("{}\t{}\t{}\t{}\t{}\n".format('Value', 'Clump', 'Uniformity', 'Marginal', 'Mitoses'))
        

    ############################## Computing probabilities ####################################

    #Iterate through all classes
    for class_ in np.unique(X_train[:,4]):
        #iterate through all possible values for the features(1 to 10):
        for value in range(1, 11):
            probabilities=get_prob_class(X_train, value, class_ )
            #transfrom vector of probabilities to a string
            str_probabilities = "\t".join("{0:.3f}".format(p) for p in probabilities)

            if class_==2:
                f_out_class1.write('{}\t{}\n'.format(str(value), str_probabilities))
            elif class_==4:
                f_out_class2.write('{}\t{}\n'.format(str(value), str_probabilities))

    f_out_class1.close()
    f_out_class2.close()

    #extract probabilities for the different classes 
    output1="{}/output_summary_class_{}.txt".format(args.outdir, labels[0])
    output2="{}/output_summary_class_{}.txt".format(args.outdir, labels[1])
    prob_class1=np.genfromtxt(output1, delimiter='\t', dtype=float)
    prob_class2=np.genfromtxt(output2, delimiter='\t', dtype=float)

    #### Predicting the class for the given sample #########

    #storing the features of the given sample 
    clump=6
    uniformity=2
    marginal=2
    mitoses=1

    p_y_class1=len(X_train[X_train[:,4]==2])/X_train.shape[0]
    p_y_class2=len(X_train[X_train[:,4]==4])/X_train.shape[0]


    #calculate probabilities for the diffent classes 
    sample_class1= prob_class1[prob_class1[:,0]==clump][:,1]*prob_class1[prob_class1[:,0]==uniformity][:,2]*prob_class1[prob_class1[:,0]==marginal][:,3]*prob_class1[prob_class1[:,0]==mitoses][:,4]*p_y_class1
    sample_class2= prob_class2[prob_class1[:,0]==clump][:,1]*prob_class2[prob_class1[:,0]==uniformity][:,2]*prob_class2[prob_class1[:,0]==marginal][:,3]*prob_class2[prob_class1[:,0]==mitoses][:,4]*p_y_class2

    if sample_class1 > sample_class2:
        print("The predicted label for the sample is ", str(labels[0]))
    else:
        print("The predicted label for the sample is ", str(labels[1]))



