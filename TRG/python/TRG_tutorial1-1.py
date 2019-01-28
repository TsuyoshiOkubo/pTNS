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
