# coding:utf-8
import matplotlib.pyplot as plt
import TensorNetwork_TRG as TN
import Make_EC

## read data
T_L2,f_L2 = TN.read_free_energy("free_energy_L2_D4.dat")
T_L4,f_L4 = TN.read_free_energy("free_energy_L4_D4.dat")
T_L8,f_L8 = TN.read_free_energy("free_energy_L8_D4.dat")
T_L16,f_L16 = TN.read_free_energy("free_energy_L16_D4.dat")
T_L32,f_L32 = TN.read_free_energy("free_energy_L32_D4.dat")

# Calclate Energy and Specific Heat from free_energy differences
T_L2_cut,E_L2,C_L2 = Make_EC.Calculate_EC(T_L2,f_L2)
T_L4_cut,E_L4,C_L4 = Make_EC.Calculate_EC(T_L4,f_L4)
T_L8_cut,E_L8,C_L8 = Make_EC.Calculate_EC(T_L8,f_L8)
T_L16_cut,E_L16,C_L16 = Make_EC.Calculate_EC(T_L16,f_L16)
T_L32_cut,E_L32,C_L32 = Make_EC.Calculate_EC(T_L32,f_L32)


## plot data
## energy
fig1,ax1= plt.subplots()
ax1.set_xlabel("T")
ax1.set_ylabel("E")
ax1.set_title("Energy density of square lattice Ising model")
ax1.plot(T_L2_cut, E_L2, "o", label = "L=2")
ax1.plot(T_L4_cut,E_L4, "o", label = "L=4")
ax1.plot(T_L8_cut, E_L8, "o", label = "L=8")
ax1.plot(T_L16_cut,E_L16, "o", label = "L=16")
ax1.plot(T_L32_cut, E_L32, "o", label = "L=32")

ax1.legend(loc="upper left")


## Specific Heat
fig2,ax2= plt.subplots()
ax2.set_xlabel("T")
ax2.set_ylabel("C")
ax2.set_title("Specific Heat of square lattice Ising model")
ax2.plot(T_L2_cut, C_L2, "o", label = "L=2")
ax2.plot(T_L4_cut,C_L4, "o", label = "L=4")
ax2.plot(T_L8_cut, C_L8, "o", label = "L=8")
ax2.plot(T_L16_cut,C_L16, "o", label = "L=16")
ax2.plot(T_L32_cut, C_L32, "o",label = "L=32")

ax2.legend(loc="upper left")

plt.show()
