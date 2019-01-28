# coding:utf-8
import numpy as np
from PEPS_Parameters import *

## Lattice setting

Tensor_list = np.zeros((LX,LY),dtype=int)
NN_Tensor = np.zeros((N_UNIT,4),dtype=int)
A_sub_list = np.zeros(N_UNIT/4,dtype=int)
B_sub_list = np.zeros(N_UNIT/4,dtype=int)
C_sub_list = np.zeros(N_UNIT/4,dtype=int)
D_sub_list = np.zeros(N_UNIT/4,dtype=int)

a_num=0
b_num=0
c_num=0
d_num=0
for ix in range(LX):
    for iy in range(LY):
        num = iy * LX + ix
        Tensor_list[ix,iy] = num
        if iy % 2 ==0:
            if ix % 2 ==0:
                A_sub_list[a_num] = num
                a_num += 1                
            else:
                B_sub_list[b_num] = num
                b_num += 1
        else:
            if ix % 2 ==0:
                C_sub_list[c_num] = num
                c_num += 1                
            else:
                D_sub_list[d_num] = num
                d_num += 1
    
for i in range(N_UNIT):
    ix = i%LX
    iy = i/LX
    
    NN_Tensor[i,0] = Tensor_list[(ix -1 + LX)%LX, iy]
    NN_Tensor[i,1] = Tensor_list[ix,(iy+1)%LY]
    NN_Tensor[i,2] = Tensor_list[(ix+1)%LX,iy]
    NN_Tensor[i,3] = Tensor_list[ix,(iy-1+LY)%LY]



    
