
# coding: utf-8

# # Sample code for exercise 1-2
# 2017 Aug. Tsuyoshi Okubo  
# 2018 Dec. modified
# 
# In this code, you can simulate SVD(Schmidt decomposition) of the ground state of spin model on 1d chain.  
# $$\mathcal{H} = \sum_{i} J_z S_{i,z}S_{i+1,z} + J_{xy} (S_{i,x}S_{i+1,x} + S_{i,y}S_{i+1,y}) - h_x \sum_i S_{i,x} + D\sum_i S_{i,z}^2$$
# 
# You can change   
# 
# - N: # of sites
# - m: size of spin  (2S = 2m + 1)  
# - Jz: amplitude of SzSz interaction  
# - Jxy: amplitude of SxSx + SySy interaction  
# - hx : amplitude of external field alogn x direction  
# - D : Single ion anisotropy  
# - periodic: Flag for periodic boundary condition  

import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot
import ED 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SVD siumulator for the GS of one dimensional spin model')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=10,
                        help='set system size N (default = 10)')
    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=1.0,
                        help='amplitude for SzSz interaction  (default = 1.0)')
    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=1.0,
                        help='amplitude for SxSx + SySy interaction  (default = 1.0)')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=3,
                        help='Spin size m=2S +1  (default = 3)' )
    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=0.0,
                        help='extarnal magnetix field  (default = 0.0)')
    parser.add_argument('-D', metavar='D',dest='D', type=float, default=0.0,
                        help='single ion anisotropy Sz^2  (default = 0.0)')
    parser.add_argument('--periodic',dest='periodic', action="store_true",
                        help='set periodic boundary condision (default = open)')
    return parser.parse_args()

## read params from command line
args = parse_args()
N = args.N       ## Chain length 
m = args.m       ## m = 2S + 1, e.g. m=3 for S=1 
Jz = args.Jz      ## Jz for SzSz interaction
Jxy = args.Jxy    ## Jxy for SxSx + SySy interaction
hx = args.hx     ## external field along x direction
D = args.D       ## single ion anisotropy
periodic = args.periodic ## periodic boundasry condition


print("2S = m - 1, N-site spin chain")
print("N = "+repr(N))
print("m = "+repr(m))
print("Hamiltonian parameters:")
print("Jz = "+repr(Jz))
print("Jxy = "+repr(Jxy))
print("hx = "+repr(hx))
print("D = "+repr(D))
print("periodic = "+repr(periodic))


## Obtain the smallest eigenvalue
eig_val,eig_vec = ED.Calc_GS(m,Jz, Jxy,hx,D,N,k=1,periodic=periodic)
if periodic :
    print("Ground state energy per bond= " +repr(eig_val[0]/N))
else:
    print("Ground state energy per bond= " +repr(eig_val[0]/(N-1)))


## Make matrix from wave function
Mat = eig_vec[:,0].reshape(m**int(N/2),m**(N-int(N/2)))

## SVD
U,s,VT = linalg.svd(Mat,full_matrices=False)


## Entanglement entropy
EE = -np.sum(s**2*np.log(s**2))
print("normalization = "+ repr(np.sum(s**2)))

s /=np.sqrt(np.sum(s**2))
EE = -np.sum(s**2*np.log(s**2))
print("Entanglement entropy = " + repr(EE))


## plot singular values
pyplot.title(repr(N)+" sites spin chain")
pyplot.plot(np.arange(m**(N/2)),s,"o")
pyplot.xlabel("index")
pyplot.ylabel("sigular value")
pyplot.yscale("log")
pyplot.show()

