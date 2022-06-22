"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Main program for Naive Bayes data summarization.
Compute the probabilities of each attribute/value and class, i.e. P(y = y_i | X = x)

"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# Extended by: Bastian Rieck <bastian.rieck@bsse.ethz.ch>

import argparse
import os
import sys
import numpy as np
# Use pandas to read/process the data
import pandas as pd

# Constants
# Names of columns in the file
COL_NAMES  = ["clump", "uniformity", "marginal", "mitoses", "class"]
# Possible values for each attribute
NUM_VALUES = 10

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------
if __name__ == '__main__':
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(description="Compute distance functions on vectors")
    parser.add_argument("--traindir", required=True, 
                        help="Path to the location of the training data. It will contain the file: tumor_info.txt")
    parser.add_argument("--outdir", required=True, 
                        help="Path to the output directory, where the output file will be created")
    args = parser.parse_args()

    # If the output directory does not exist, then create it
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Read the training data into a data frame
    df = pd.read_csv("%s/tumor_info.txt" % args.traindir, sep='\t', header=None, names=COL_NAMES)

    # Get the unique class values
    vec_labels = np.unique(df["class"])
    # Iterate through the different class labels
    for label in vec_labels:
        # Initialize the matrix that will contain all the summary probabilities
        mat = np.zeros((NUM_VALUES, len(COL_NAMES) - 1))

        # Iterate through all attributes, exept the class
        for col in range(0, len(COL_NAMES) - 1):
            # Get the attribute name
            name = COL_NAMES[col]

            # Slice the data frame to obtain the current attribute+class
            attrib_class = df[[name, "class"]]
            
            # Get a summary of the counts for each value in this attribute/class
            # Example: data frame counts for attribute "clump" and class "2"
            #          class
            #   clump       
            #   1        137
            #   2         45
            #   3         91
            #   :          : 
            idx = attrib_class["class"] == label
            counts = attrib_class[idx].groupby(name).count()
            # Get the sum of all counts for this attribute/class
            sum_attrib = float(counts["class"].sum())

            # Get the list of indeces (rownames) and convert them to integers
            rownames = [int(x) for x in list(counts.index)]
            # Iterate through the counts and fill up the matrix
            # Note: The matrix uses 0-based indeces
            for row in rownames:
                mat[row - 1, col] = counts["class"][row] / sum_attrib

        # Create the output file
        try:
            file_name = "%s/output_summary_class_%d.txt" % (args.outdir, label)
            f_out = open(file_name, 'w')
        except IOError:
            print("Output file {} cannot be created".format(file_name))
            sys.exit(1)

        # Save the contents of the matrix into the file
        # Create the header and save it
        header = "\t".join("%s" % x for x in COL_NAMES[0:len(COL_NAMES) - 1])
        header = "Value\t" + header
        f_out.write("%s\n" % header)
        # Iterate through all rows in the matrix
        for row in range(0, NUM_VALUES):
            line = "\t".join("%.3f" % x for x in mat[row, :])
            f_out.write("%d\t%s\n" % (row + 1, line))

    f_out.close()

