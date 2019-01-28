# coding: utf-8

# # Sample code for exercise 2-1
# 2017 Aug. Tsuyoshi Okubo  
# 2018 Dec. modified
# 
# In this code, you can simulate MPS decompositionof the ground state of spin model on 1d chain.  
# $$\mathcal{H} = \sum_{i} J_z S_{i,z}S_{i+1,z} + J_{xy} (S_{i,x}S_{i+1,x} + S_{i,y}S_{i+1,y}) - h_x \sum_i S_{i,x} + D\sum_i S_{i,z}^2$$
# 
# Note that, in this exercise, the MPS is exact (no approximation). Thus, the energy calculated from MPS should be same with the ED.  
# You can change   
# 
# - N: # of sites
# - m: size of spin  (2S = 2m + 1)  
# - Jz: amplitude of SzSz interaction  
# - Jxy: amplitude of SxSx + SySy interaction  
# - hx : amplitude of external field alogn x direction  
# - D : Single ion anisotropy  
# - (periodic: In this exercize, we only consider open boundary)


import numpy as np
import scipy.linalg as linalg
import ED
import TEBD
from matplotlib import pyplot
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='siumulator for MPS decomposition of the GS of one dimensional spin model (without approximation)')
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
    return parser.parse_args()


## read params from command line
args = parse_args()
N = args.N       ## Chain length 
m = args.m       ## m = 2S + 1, e.g. m=3 for S=1 
Jz = args.Jz      ## Jz for SzSz interaction
Jxy = args.Jxy    ## Jxy for SxSx + SySy interaction
hx = args.hx     ## external field along x direction
D = args.D       ## single ion anisotropy

#periodic = False ## in this exersize , we only consider open boundary


print("2S = m - 1, N-site spin chain")
print("N = "+repr(N))
print("m = "+repr(m))
print("Hamiltonian parameters:")
print("Jz = "+repr(Jz))
print("Jxy = "+repr(Jxy))
print("hx = "+repr(hx))
print("D = "+repr(D))


## Obtain the smallest eigenvalue
eig_val,eig_vec = ED.Calc_GS(m,Jz, Jxy,hx,D,N,k=1)
print("Ground state energy per bond= " +repr(eig_val[0]/(N-1)))


## Make exact MPS (from "left")
Tn = []
lam = [np.ones((1,))]
lam_inv = 1.0/lam[0]
R_mat = eig_vec[:,0].reshape(m,m**(N-1))

chi_l=1
for i in range(N-1):
    U,s,VT = linalg.svd(R_mat,full_matrices=False)
    chi_r = s.size

    Tn.append(np.tensordot(np.diag(lam_inv),U.reshape(chi_l,m,chi_r),(1,0)).transpose(1,0,2))
    lam.append(s)
    lam_inv = 1.0/s
    R_mat = np.dot(np.diag(s),VT).reshape(chi_r*m,m**(N-i-2))
    chi_l = chi_r
Tn.append(VT.reshape(m,m,1).transpose(1,0,2))
lam.append(np.ones((1,)))


## Calculate Energy
Env_left=[]
Env_right=[]
for i in range(N):
    Env_left.append(np.identity((lam[i].shape[0])))
    Env_right.append(np.dot(np.dot(np.diag(lam[i+1]),np.identity((lam[i+1].shape[0]))),np.diag(lam[i+1])))

print("Energy of MPS = "+repr(TEBD.Calc_Energy(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,D)))

