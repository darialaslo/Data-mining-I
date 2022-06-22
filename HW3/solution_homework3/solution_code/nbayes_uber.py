'''
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)


Program for calculating the maximum value of the posterior distribution
as well as its expected value, both approximatd and analytically.
'''

#!/usr/bin/env python3

import numpy as np
from functools import partial
from IPython import embed

def prior(N, N_max):
    return 1 / N_max


def evidence(D, N_max, prior):
    return sum([1/N * prior(N) for N in range(D, N_max + 1)])


def likelihood(D, N):
    if D <= N:
        return 1 / N
    else:
        return 0


def posterior(D, N, N_max):
    return likelihood(D, N) * prior(N, N_max) / evidence(D, N_max, partial(prior, N_max=1000))


if __name__ == '__main__':
    N_max = 1000
    D = 60

    posterior_distribution = [posterior(D, N, N_max) for N in range(D, N_max + 1)]

    print('Posterior has its maximum value of', np.max(posterior_distribution), 'at', np.argmax(posterior_distribution) + D)

