# Tensor Renormalization Groupe
# based on M. Levin and C. P. Nave, PRL 99 120601 (2007)
# and X.-C. Gu, M. Levin, and X.-G. Wen, PRB 78, 205116(2008).
# 2014 Dec. (bug fixed 2015 Jan.) Tsuyoshi Okubo

# coding:utf-8
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import argparse
import TRG 

#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='Tensor Network Renormalization for Square lattice Ising model')
    parser.add_argument('-D', metavar='D',dest='D', type=int, default=4,
                        help='set bond dimension D for TRG')
    parser.add_argument('-n', metavar='n',dest='n', type=int, default=1,
                        help='set size n representing L=2^n')
    parser.add_argument('-T', metavar='T',dest='T', type=float, default=2.0,
                        help='set Temperature')
    parser.add_argument('--step', action='store_const', const=True,
                        default=False, help='Perform multi temperature calculation')
    parser.add_argument('--energy', action='store_const', const=True,
                        default=False, help='Calculate energy density by using impurity tensor')

    parser.add_argument('-Tmin', metavar='Tmin',dest='Tmin', type=float, default=1.5,
                        help='set minimum temperature for step calculation')

    parser.add_argument('-Tmax', metavar='Tmax',dest='Tmax', type=float, default=3.0,
                        help='set maximum temperature for step calculation')
    parser.add_argument('-Tstep', metavar='Tstep',dest='Tstep', type=float, default=0.1,
                        help='set temperature increments for step calculation')

    
    return parser.parse_args()

def Calculate_TRG(D=4,n=1,T=2.0,Tmin=1.5,Tmax=3.0,Tstep=0.1,Energy_flag=False,Step_flag=False):

    TRG_step = 2*n - 1
    L = 2**n
    
    tag = "_L"+str(L)+"_D"+str(D)
    
    file_free_energy = open("free_energy"+tag+".dat","w")
    file_free_energy.write("# L = "+repr(L) + ", D = "+repr(D) + "\n")
    if Energy_flag:
        file_energy = open("energy"+tag+".dat","w")
        file_energy.write("# L = "+repr(L) + ", D = "+repr(D) + "\n")

    if Step_flag:
        ## step calculation
        for T in np.arange(Tmin,Tmax,Tstep):
            if Energy_flag:
                free_energy_density,energy_density = TRG.TRG_Square_Ising(T,D,TRG_step,Energy_flag=True)
                file_free_energy.write(repr(T)+" "+repr(free_energy_density) + "\n")
                file_energy.write(repr(T)+" "+repr(energy_density) + "\n")
            else:
                free_energy_density = TRG.TRG_Square_Ising(T,D,TRG_step,Energy_flag=False)
                file_free_energy.write(repr(T)+" "+repr(free_energy_density) + "\n")                    
    else:
        ## single calculation
        if Energy_flag:
            free_energy_density,energy_density = TRG.TRG_Square_Ising(T,D,TRG_step,Energy_flag=True)
            file_free_energy.write(repr(T)+" "+repr(free_energy_density) + "\n")
            file_energy.write(repr(T)+" "+repr(energy_density) + "\n")
        else:
            free_energy_density = TRG.TRG_Square_Ising(T,D,TRG_step,Energy_flag=False)
            file_free_energy.write(repr(T)+" "+repr(free_energy_density) + "\n")


def read_free_energy(file_name):
    T = []
    f = []
    for line in open(file_name, "r"):
        data = line.split()
        if data[0] =="#":
            header = line
            continue
        T.append(float(data[0]))
        f.append(float(data[1]))
    return T,f

def main():
    ## read params from command line
    args = parse_args()
    Calculate_TRG(args.D,args.n,args.T,args.Tmin,args.Tmax,args.Tstep,args.energy,args.step)    
    
if __name__ == "__main__":
    main()

