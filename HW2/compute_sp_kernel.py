from shortest_path_kernel import floyd_warshall 
from shortest_path_kernel import sp_kernel
import os
import sys
import argparse
import numpy as np


import scipy.io






if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute distance functions on time-series"
    )
    parser.add_argument(
        "--datadir",
        required=True,
        help="Path to input directory containing file EGC200_TRAIN.txt"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where timeseries_output.txt will be created"
    )

    args = parser.parse_args()

    # Set the paths
    data_dir = args.datadir
    out_dir = args.outdir

    os.makedirs(args.outdir,exist_ok=True)

    # Read the file
    mat = scipy.io.loadmat("{}/{}".format(args.datadir, 'MUTAG.mat'))
    label = np.reshape(mat['lmutag'], (len(mat['lmutag'], ))) 
    data = np.reshape(mat['MUTAG']['am'], (len(label), ))


    # Create the output file
    try:
        file_name = "{}/graphs_output.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

    mut = data[label== 1]
    non_mut = data[label== -1]

    cdict = {}
    cdict['mutagenic'] = mut
    cdict['non-mutagenic'] = non_mut

    label_group = ['mutagenic', 'non-mutagenic']


    # Write header for output file
    f_out.write('{}\t{}\n'.format(
        'Pair of classes',
        'SP'))
    
    

    # Iterate through all combinations of pairs
    for idx_g1 in range(len(label_group)):
        for idx_g2 in range(idx_g1, len(label_group)):
            # Get the group data
            group1 = cdict[label_group[idx_g1]]
            group2 = cdict[label_group[idx_g2]]

            count = 0
            kernel_value=0
            # Get average similarity
            for x in group1:
                for y in group2:

                    # Compute shortest path kernel
                    S1=floyd_warshall(x)
                    S2=floyd_warshall(y)
                    kernel_value+=sp_kernel(S1,S2)
                    count += 1
            kernel_value /=count


            # Transform the vector of distances to a string
            str_sim = '{0:.2f}'.format(kernel_value)

            # Save the output
            f_out.write(
                '{}:{}\t{}\n'.format(
                    label_group[idx_g1], label_group[idx_g2], str_sim)
            )
    f_out.close()
