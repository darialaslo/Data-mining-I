"""Skeleton file for your solution to the shortest-path kernel."""


def floyd_warshall(A):
    """Implement the Floyd--Warshall on an adjacency matrix A.

    Parameters
    ----------
    A : `np.array` of shape (n, n)
        Adjacency matrix of an input graph. If A[i, j] is `1`, an edge
        connects nodes `i` and `j`.

    Returns
    -------
    An `np.array` of shape (n, n), corresponding to the shortest-path
    matrix obtained from A.
    """
    import numpy as np
    #setting all 0 entries to infinity
    SP=np.where(A==0, float("inf"),A)

    #determining the size of A
    n=A.shape[0]

    #iterating and calculating the shortest path matrix
    for k in range(0,n):
        for i in range(0,n):
            for j in range (0,n):
                if SP[i,j]>SP[i,k]+SP[k,j]:
                    #updating the shortest path when finding a better one
                    SP[i,j]=SP[i,k]+SP[k,j]

    return SP


def sp_kernel(S1, S2):
    """Calculate shortest-path kernel from two shortest-path matrices.

    Parameters
    ----------
    S1: `np.array` of shape (n, n)
        Shortest-path matrix of the first input graph.

    S2: `np.array` of shape (m, m)
        Shortest-path matrix of the second input graph.

    Returns
    -------
    A single `float`, corresponding to the kernel value of the two
    shortest-path matrices
    """
    kernel_value=0
    n=S1.shape[0]
    m=S2.shape[0]

    for i in range(0,n):
        for j in range(i+1,n):
            s1=S1[i,j]
            for k in range(0,m):
                for l in range(k+1, m):
                    s2=S2[k,l]
                    if s1==s2:
                        kernel_value+=1

    return kernel_value
