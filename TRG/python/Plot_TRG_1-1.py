# coding:utf-8
import matplotlib.pyplot as plt
import TensorNetwork_TRG as TN

## read data
T_L2,f_L2 = TN.read_free_energy("free_energy_L2_D4.dat")
T_L4,f_L4 = TN.read_free_energy("free_energy_L4_D4.dat")
T_L8,f_L8 = TN.read_free_energy("free_energy_L8_D4.dat")
T_L16,f_L16 = TN.read_free_energy("free_energy_L16_D4.dat")
T_L32,f_L32 = TN.read_free_energy("free_energy_L32_D4.dat")

# read exact data
T_L2e,f_L2e = TN.read_free_energy("exact/outputs/free_energy_exact_L2.dat")
T_L4e,f_L4e = TN.read_free_energy("exact/outputs/free_energy_exact_L4.dat")
T_L8e,f_L8e = TN.read_free_energy("exact/outputs/free_energy_exact_L8.dat")
T_L16e,f_L16e = TN.read_free_energy("exact/outputs/free_energy_exact_L16.dat")
T_L32e,f_L32e = TN.read_free_energy("exact/outputs/free_energy_exact_L32.dat")

## plot data
fig1,ax1= plt.subplots()
ax1.set_xlabel("T")
ax1.set_ylabel("f")
ax1.set_title("Free energy density of square lattice Ising model")
ax1.plot(T_L2e, f_L2e, "r",label = "L=2: Exact")
ax1.plot(T_L4e,f_L4e, "g",label = "L=4: Exact")
ax1.plot(T_L8e, f_L8e, "b",label = "L=8: Exact")
ax1.plot(T_L16e,f_L16e, "c",label = "L=16: Exact")
ax1.plot(T_L32e, f_L32e, "m",label = "L=32: Exact")
ax1.plot(T_L2, f_L2, "ro",label = "L=2")
ax1.plot(T_L4,f_L4, "go",label = "L=4")
ax1.plot(T_L8, f_L8, "bo",label = "L=8")
ax1.plot(T_L16,f_L16, "co",label = "L=16")
ax1.plot(T_L32, f_L32, "mo",label = "L=32")

ax1.legend(loc="lower left")

plt.show()
