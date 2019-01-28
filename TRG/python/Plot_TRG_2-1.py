# coding:utf-8
import matplotlib.pyplot as plt
import TensorNetwork_TRG as TN

## read data
T_L2,e_L2 = TN.read_free_energy("energy_from_free_energy_L2_D4.dat")
T_L2,c_L2 = TN.read_free_energy("specific_heat_from_free_energy_L2_D4.dat")

## plot Energy
fig1,ax1= plt.subplots()
ax1.set_xlabel("T")
ax1.set_ylabel("E")
ax1.set_title("Energy density of square lattice Ising model")
ax1.plot(T_L2, e_L2, "o",label = "L=2")

## plot Specific Heat
fig2,ax2= plt.subplots()
ax2.set_xlabel("T")
ax2.set_ylabel("C")
ax2.set_title("Specific heat of square lattice Ising model")
ax2.plot(T_L2, c_L2, "o",label = "L=2")
#ax1.legend(loc="lower left")

plt.show()
