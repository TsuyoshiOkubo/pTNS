# coding:utf-8
import numpy as np
from PEPS_Parameters import *

## Lattice setting

Tensor_list = np.zeros((LX,LY),dtype=int)
Tensor_position = np.zeros((N_UNIT,2),dtype=int)

NN_Tensor = np.zeros((N_UNIT,4),dtype=int)
A_sub_list = np.zeros(N_UNIT/2,dtype=int)
B_sub_list = np.zeros(N_UNIT/2,dtype=int)

a_num=0
b_num=0
# Tensors are assinged based on LX_ori and LY_ori
for ix in range(LX_ori):
    for iy in range(LY_ori):
        num = iy * LX_ori + ix
        Tensor_list[ix,iy] = num
        Tensor_position[num,0] = ix
        Tensor_position[num,1] = iy
        if (ix + iy)%2 ==0:
            A_sub_list[a_num] = num
            a_num += 1
        else:
            B_sub_list[b_num] = num
            b_num += 1

## extend for larger "periodic" unit cell
if (LX > LX_ori):
    ## assuming LY_ori = LY
    for ix in range(LX_ori,LX):
        slide = LX_diff * (ix / LX_ori)
        ix_ori = ix % LX_ori
        for iy in range(LY_ori):
            iy_ori = (iy - slide + LY_ori) % LY_ori
            num = Tensor_list[ix_ori,iy_ori]
            Tensor_list[ix,iy] = num
elif (LY > LY_ori):
    ## assuming LX_ori = LX
    for iy in range(LY_ori,LY):
        slide = LY_diff * (iy / LY_ori)
        iy_ori = iy % LY_ori
        for ix in range(LX_ori):
            ix_ori = (ix - slide + LX_ori) % LX_ori
            num = Tensor_list[ix_ori,iy_ori]
            Tensor_list[ix,iy] = num
#else LX = LX_ori, LY = LY_ori
    
for i in range(N_UNIT):
    ix = i%LX_ori
    iy = i/LX_ori
    
    NN_Tensor[i,0] = Tensor_list[(ix -1 + LX)%LX, iy]
    NN_Tensor[i,1] = Tensor_list[ix,(iy+1)%LY]
    NN_Tensor[i,2] = Tensor_list[(ix+1)%LX,iy]
    NN_Tensor[i,3] = Tensor_list[ix,(iy-1+LY)%LY]



    
