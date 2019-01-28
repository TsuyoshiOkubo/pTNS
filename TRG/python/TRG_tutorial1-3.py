# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from TensorNetwork_TRG import *

## calculate free energies for L=2,4,8,16,32 with fixed D_cut
Dcut = 4
Tmin = 1.0
Tmax = 4.0
Tstep = 0.1

Calculate_TRG(D = Dcut, n=1, T=1.0, Tmin=Tmin, Tmax=Tmax, Tstep=Tstep, Energy_flag=False, Step_flag=True)
Calculate_TRG(D = Dcut, n=2, T=1.0, Tmin=Tmin, Tmax=Tmax, Tstep=Tstep, Energy_flag=False, Step_flag=True)
Calculate_TRG(D = Dcut, n=3, T=1.0, Tmin=Tmin, Tmax=Tmax, Tstep=Tstep, Energy_flag=False, Step_flag=True)
Calculate_TRG(D = Dcut, n=4, T=1.0, Tmin=Tmin, Tmax=Tmax, Tstep=Tstep, Energy_flag=False, Step_flag=True)
Calculate_TRG(D = Dcut, n=5, T=1.0, Tmin=Tmin, Tmax=Tmax, Tstep=Tstep, Energy_flag=False, Step_flag=True)

## read data
T_L2,f_L2 = read_free_energy("free_energy_L2_D4.dat")
T_L4,f_L4 = read_free_energy("free_energy_L4_D4.dat")
T_L8,f_L8 = read_free_energy("free_energy_L8_D4.dat")
T_L16,f_L16 = read_free_energy("free_energy_L16_D4.dat")
T_L32,f_L32 = read_free_energy("free_energy_L32_D4.dat")

## plot data
fig1,ax1= plt.subplots()
ax1.set_xlabel("T")
ax1.set_ylabel("f")
ax1.set_title("Free energy density of square lattice Ising model")
ax1.plot(T_L2, f_L2, label = "L=2")
ax1.plot(T_L4,f_L4, label = "L=4")
ax1.plot(T_L8, f_L8, label = "L=8")
ax1.plot(T_L16,f_L16, label = "L=16")
ax1.plot(T_L32, f_L32, label = "L=32")

plt.legend()

plt.show()
