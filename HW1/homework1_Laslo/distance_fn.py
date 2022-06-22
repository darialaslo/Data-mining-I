"""Homework 1: Distance functions on vectors.

Homework 1: Distance functions on vectors
Course    : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the distance functions that are invoked from the main
program.
"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# Author: Bastian Rieck <bastian.rieck@bsse.ethz.ch>

import numpy as np
import math


def manhattan_dist(v1, v2):

    absolut_diff=np.abs(v1-v2)
    dist=np.sum(absolut_diff)

    return dist



def hamming_dist(v1, v2):
    dist=0
    for x in range(v1.shape[0]):
        x1i=v1[x]
        x2i=v2[x]
        if (x1i>0):
            x1i=1
        if (x2i>0):
            x2i=1
        absolut_value=abs(x1i-x2i)
        dist=dist+absolut_value

    return dist


def euclidean_dist(v1, v2):

    sum_diff=0

    for i in range(v1.shape[0]):
        diff_squared= (v1[i]-v2[i])**2
        sum_diff= sum_diff+diff_squared
        
    dist=math.sqrt(sum_diff)

    return dist


def chebyshev_dist(v1, v2):

   
    dist=0
    for i in range(v1.shape[0]):
        if (np.abs(v1[i]-v2[i])>dist):
            dist=np.abs(v1[i]-v2[i])

    return dist


def minkowski_dist(v1, v2, d):
    
    diff=np.abs(v1-v2)
    diff_power=np.power(diff,d)
    sum_diffs=np.sum(diff_power)
    power_coeff=1/float(d)
    dist_min=sum_diffs**power_coeff
    
    return dist_min

