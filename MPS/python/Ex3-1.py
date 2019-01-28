# coding: utf-8

# # Sample code for exercise 3-1
# 2017 Aug. Tsuyoshi Okubo  
# 2018 Dec. modified
# 
# In this code, you can perform TEBD simulation of the ground state of spin model on 1d chain.  
# $$\mathcal{H} = \sum_{i} J_z S_{i,z}S_{i+1,z} + J_{xy} (S_{i,x}S_{i+1,x} + S_{i,y}S_{i+1,y}) - h_x \sum_i S_{i,x} + D\sum_i S_{i,z}^2$$
# 
# Note that, the accuracy of the calculation depends on chi_max, tau, and iteration steps.
# tau is gradually decreases from tau_max to tau_min
# 
# 
# You can change   
# 
# - N: # of sites
# - m: size of spin  (2S = 2m + 1)  
# - Jz: amplitude of SzSz interaction  
# - Jxy: amplitude of SxSx + SySy interaction  
# - hx : amplitude of external field alogn x direction  
# - D : Single ion anisotropy  
# - (periodic: In this exercize, we only consider open boundary)
# - chi_max : maximum bond dimension of MPS
# - tau_max : maximum value of tau
# - tau_min : minimum value of tau
# - T_step : Total ITE steps
# - output_dyn_num : output data step


import numpy as np
import scipy.linalg as linalg
import ED
import TEBD
from matplotlib import pyplot
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='TEBD siumulator for one dimensional spin model')
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
    parser.add_argument('-chi', metavar='chi_max',dest='chi_max', type=int, default=20,
                        help='maximum bond dimension at truncation  (default = 20)')
    parser.add_argument('-tau_max', metavar='tau_max',dest='tau_max', type=float, default=0.1,
                        help='start imaginary time step  (default = 0.1)')
    parser.add_argument('-tau_min', metavar='tau_min',dest='tau_min', type=float, default=0.001,
                        help='final imaginary time step  (default = 0.001)')
    parser.add_argument('-tau_step', metavar='tau_step',dest='tau_step', type=int, default=2000,
                        help='ITE steps. tau decreses from tau_max to tau_min gradually  (default = 2000)')
    parser.add_argument('-output_dyn_num', metavar='output_dyn_num',dest='output_dyn_num', type=int, default=100,
                        help='number of data points at dynamics output  (default = 100)')

    return parser.parse_args()


## read params from command line
args = parse_args()

N = args.N       ## Chain length 
m = args.m       ## m = 2S + 1, e.g. m=3 for S=1 
Jz = args.Jz      ## Jz for SzSz interaction
Jxy = args.Jxy    ## Jxy for SxSx + SySy interaction
hx = args.hx     ## external field along x direction
D = args.D       ## single ion anisotropy
chi_max = args.chi_max      ## maxmum bond dimension at truncation
tau_max = args.tau_max     ## start imaginary time tau
tau_min = args.tau_min   ## final imaginary time tau
T_step = args.tau_step       ## ITE steps
output_dyn_num = args.output_dyn_num ## output steps


print("2S = m - 1, N-site spin chain")
print("N = "+repr(N))
print("m = "+repr(m))
print("Hamiltonian parameters:")
print("Jz = "+repr(Jz))
print("Jxy = "+repr(Jxy))
print("hx = "+repr(hx))
print("D = "+repr(D))

print("chi_max = "+repr(chi_max))

print("tau_max = "+repr(tau_max))
print("tau_min = "+repr(tau_min))
print("T_step = "+repr(T_step))
print("output_dyn_num = "+repr(output_dyn_num))



## Obtain the smallest eigenvalue
eig_val,eig_vec = ED.Calc_GS(m,Jz, Jxy,hx,D,N,k=1)
Eg = eig_val[0]/(N-1)



##TEBD simulation
Tn, lam,T_list,E_list,mz_list = TEBD.TEBD_Simulation(m,Jz,Jxy,hx,D,N,chi_max,tau_max,tau_min,T_step,output_dyn=True,output_dyn_num=output_dyn_num)



## Calculate Energy
Env_left,Env_right = TEBD.Calc_Environment(Tn,lam,canonical=False)
E_mps = TEBD.Calc_Energy(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,D)

print("Ground state energy per bond = " +repr(Eg))
print("TEBD energy per bond = " + repr(E_mps))



## plot energy dynamics
pyplot.title("TEBD Energy dynamics")
pyplot.plot(T_list[1:],E_list[1:],"o")
pyplot.xlabel("T")
pyplot.ylabel("E(T)")
pyplot.axhline(y=Eg, color='red')
pyplot.show()

