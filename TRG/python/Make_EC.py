# coding:utf-8
import numpy as np
import sys


def Calculate_EC(T,f):
    T_num = len(T)
    E = np.empty(T_num-2)
    C = np.empty(T_num-2)
    T_cut = np.empty(T_num-2)
    for i in range(0,T_num-2):
        E[i] = f[i+1]-T[i+1] * (f[i+2]-f[i])/(T[i+2]-T[i])
        C[i] = -T[i+1] * (f[i+2]+f[i]-2.0*f[i+1])/(T[i+2]-T[i+1])**2
        T_cut[i] = T[i+1]
    return T_cut,E,C

def main():
    argvs = sys.argv  
    argc = len(argvs)

    if (argc < 2):
        file_name = "free_energy.dat"
    else:
        file_name = argvs[1]

    T = []
    f = []
    for line in open(file_name, "r"):
        data = line.split()
        if data[0] =="#":
            header = line
            continue
        T.append(float(data[0]))
        f.append(float(data[1]))

    T_cut,E,C = Calculate_EC(T,f)

    file_e = open("energy_from_"+file_name,"w")
    file_c = open("specific_heat_from_"+file_name,"w")
    file_e.writelines(header)
    file_c.writelines(header)

    for i in range(0,T_cut.size):
        file_e.write(repr(T_cut[i])+" "+repr(E[i])+"\n")
        file_c.write(repr(T_cut[i])+" "+repr(C[i])+"\n")
if __name__ == "__main__":
    main()

