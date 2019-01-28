# coding: utf-8

# # Sample code for exercise 1-1
# 2017 Aug. Tsuyoshi Okubo  
# 2018 Dec. modified
# 
# In this code, you can simulate SVD(Schmidt decomposition) of random vector with m^N dimension.  
# You can change   
# - N: # of sites   
# - m: size of spin  
# 
# (In this case, these variable just mean the size of Hilbert space and no relation to spin system.)  
import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot
import argparse

#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='SVD of random vector with m^N dimension')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=10,
                        help='set system size N (default = 10)')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=3,
                        help='size of spin (size of local Hilbert space) (default = 3)')
    return parser.parse_args()

## read params from command line
args = parse_args()
N = args.N      ## Chain length 
m = args.m      ## m = 2S + 1, e.g. m=3 for S=1 


## make random complex vector
vec = (np.random.rand(m**N)-0.5) + 1.0j * (np.random.rand(m**N)-0.5)

## Make matrix from wave function
Mat = vec[:].reshape(m**(int(N/2)),m**(N-int(N/2)))

## SVD
U,s,VT = linalg.svd(Mat,full_matrices=False)


## Entanglement entropy
EE = -np.sum(s**2*np.log(s**2))
print("normalization = "+ repr(np.sum(s**2)))

s /=np.sqrt(np.sum(s**2))
EE = -np.sum(s**2*np.log(s**2))
print("Entanglement entropy = " + repr(EE))


## plot singular values
pyplot.title(repr(N)+" sites random vector")
pyplot.plot(np.arange(m**(N/2)),s,"o")
pyplot.xlabel("index")
pyplot.ylabel("sigular value")
pyplot.yscale("log")
pyplot.show()

