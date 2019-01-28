# coding:utf-8
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import time
import sys

## import basic routines
from PEPS_Basics import *
from PEPS_Parameters import *
from Square_lattice_CTM import *

## timers
time_simple_update=0.0
time_full_update=0.0
time_env=0.0
time_obs=0.0

## Parameters
theta = 0.01
Jx = np.cos(theta * np.pi)
Jy = np.sin(theta * np.pi)

Initial_AKLT = True
Random_amp = 0.02
##

second_ST = True
tau = 0.01
tau_step = 1000

tau_full = 0.01
tau_full_step = 0

Initialize_every=False

## Tensors
Tn=[np.zeros((D,D,D,D,3),dtype=TENSOR_DTYPE)]
eTt=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
eTr=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
eTb=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
eTl=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]

C1=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
C2=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
C3=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
C4=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)

## lambda values from A
lambda_tensor = np.zeros((N_UNIT,4,D),dtype=float)

for i in range(1,N_UNIT):
    Tn.append(np.zeros((D,D,D,D,3),dtype=TENSOR_DTYPE))
    eTt.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTr.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTb.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTl.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))

def Set_Hamiltonian():

    Ham = np.zeros((9,9))
    ## projector onto S=2 subspace
    Ham[0,0] = 1.0
    Ham[1,1] = 0.5
    Ham[2,2] = 1.0/6.0
    Ham[3,3] = 0.5
    Ham[4,4] = 2.0/3.0 
    Ham[5,5] = 0.5
    Ham[6,6] = 1.0/6.0
    Ham[7,7] = 0.5
    Ham[8,8] = 1.0

    Ham[1,3] = 0.5
    Ham[3,1] = 0.5
    Ham[5,7] = 0.5
    Ham[7,5] = 0.5

    Ham[2,6] = 1.0/6.0
    Ham[6,2] = 1.0/6.0
    Ham[2,4] = 1.0/3.0
    Ham[4,2] = 1.0/3.0
    Ham[4,6] = 1.0/3.0
    Ham[6,4] = 1.0/3.0

    return Ham

def Initialize_Tensors(Tn,initial_AKLT=False,random_amp=0.02):

    np.random.seed(11)
    if initial_AKLT:
        #horizontal AKLT
        for i in range(N_UNIT):
            if isinstance(Tn[i][0,0,0,0,0],complex):
                Tn[i][:] = random_amp* ((np.random.rand(D,D,D,D,3)-0.5) + 1.0j * (np.random.rand(D,D,D,D,3)-0.5))
            else:
                Tn[i][:] = random_amp* (np.random.rand(D,D,D,D,3)-0.5)

            Tn[i][0,0,1,0,0] = np.sqrt(2.0/3.0)

            Tn[i][0,0,0,0,1] = -np.sqrt(1.0/3.0)
            Tn[i][1,0,1,0,1] = np.sqrt(1.0/3.0)

            Tn[i][1,0,0,0,2] = -np.sqrt(2.0/3.0)

    else:
        for i in range(N_UNIT/2):        
            num = A_sub_list[i]
            if isinstance(Tn[num][0,0,0,0,0],complex):
                Tn[num][:] = random_amp* ((np.random.rand(D,D,D,D,3)-0.5) + 1.0j * (np.random.rand(D,D,D,D,3)-0.5))
            else:
                Tn[num][:] = random_amp* (np.random.rand(D,D,D,D,3)-0.5)
            Tn[num][0,0,0,0,0] = 1.0

            num = B_sub_list[i] 
            if isinstance(Tn[num][0,0,0,0,0],complex):
                Tn[num][:] = random_amp* ((np.random.rand(D,D,D,D,3)-0.5) + 1.0j * (np.random.rand(D,D,D,D,3)-0.5))
            else:
                Tn[num][:] = random_amp* (np.random.rand(D,D,D,D,3)-0.5)
            Tn[num][0,0,0,0,2] = 1.0

Initialize_Tensors(Tn,initial_AKLT=Initial_AKLT, random_amp=Random_amp)

lambda_tensor = np.ones((N_UNIT,4,D),dtype=float)
Ham = Set_Hamiltonian()

s,U = linalg.eigh(Ham)

op12_Jx =  np.dot(np.dot(U,np.diag(np.exp(-tau * s * Jx))),U.conj().T).reshape(3,3,3,3).transpose(2,3,0,1)
op12_Jx_2 =  np.dot(np.dot(U,np.diag(np.exp(-tau*0.5 * s * Jx))),U.conj().T).reshape(3,3,3,3).transpose(2,3,0,1)

op12_Jy =  np.dot(np.dot(U,np.diag(np.exp(-tau * s * Jy))),U.conj().T).reshape(3,3,3,3).transpose(2,3,0,1)
op12_Jy_2 =  np.dot(np.dot(U,np.diag(np.exp(-tau*0.5 * s * Jy))),U.conj().T).reshape(3,3,3,3).transpose(2,3,0,1)


start_simple=time.time()


for int_tau in range(0,tau_step):
    if second_ST:
        ## simple update
        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jx_2,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jx_2,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c


        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jy_2,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## y-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jy,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jy_2,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jx_2,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c

        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jx_2,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c

    else:
        ## simple update
        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jx,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jx,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c


        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jy,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## y-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_Jy,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c


    ## done simple update
time_simple_update += time.time() - start_simple


## Start full update
if tau_full_step > 0:

    op12_Jx =  np.dot(np.dot(U,np.diag(np.exp(-tau_full * s * Jx))),U.conj().T).reshape(3,3,3,3).transpose(2,3,0,1)
    op12_Jx_2 =  np.dot(np.dot(U,np.diag(np.exp(-tau_full*0.5 * s * Jx))),U.conj().T).reshape(3,3,3,3).transpose(2,3,0,1)

    op12_Jy =  np.dot(np.dot(U,np.diag(np.exp(-tau_full * s * Jy))),U.conj().T).reshape(3,3,3,3).transpose(2,3,0,1)
    op12_Jy_2 =  np.dot(np.dot(U,np.diag(np.exp(-tau_full*0.5 * s * Jy))),U.conj().T).reshape(3,3,3,3).transpose(2,3,0,1)


    ## Environment 
    start_env = time.time()
    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
    time_env += time.time() - start_env

start_full = time.time()
for int_tau in range(0,tau_full_step):
    if second_ST:
        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_Jx_2,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                

            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_Jx_2,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_Jy_2,1)

            if Full_Use_FFU:
                iy = num/LX
                iy_j = num_j/LX
                Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)


        ## y-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,1]


            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_Jy,1)

            if Full_Use_FFU:
                iy = num/LX
                iy_j = num_j/LX
                Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_Jy_2,1)

            if Full_Use_FFU:
                iy = num/LX
                iy_j = num_j/LX
                Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_Jx_2,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_Jx_2,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                

            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

    else:
        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_Jx,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                

            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_Jx,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_Jy,1)

            if Full_Use_FFU:
                iy = num/LX
                iy_j = num_j/LX
                Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)


        ## y-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,1]


            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_Jy,1)

            if Full_Use_FFU:
                iy = num/LX
                iy_j = num_j/LX
                Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)


    ## done full update
time_full_update += time.time()-start_full

## Calc physical quantities
start_env = time.time()
Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
time_env += time.time() - start_env

op_identity = np.identity(3)
op_identity_12 = np.identity(3**2).reshape(3,3,3,3)
## note for general cases, we should use the transepose of Ham due to the definition of operator
## In this model, Ham is symmetric. Thus, no problem occurs
op_ene_x = Jx * Ham.reshape(3,3,3,3)
op_ene_y = Jy * Ham.reshape(3,3,3,3)

op_sx = np.zeros((3,3))
op_sx[0,1] = 1.0/np.sqrt(2.0)
op_sx[1,0] = 1.0/np.sqrt(2.0)
op_sx[1,2] = 1.0/np.sqrt(2.0)
op_sx[2,1] = 1.0/np.sqrt(2.0)

#!!note operator should be transposed due to the definition
op_sy = np.zeros((3,3),dtype=complex)
op_sy[0,1] = 1.0j/np.sqrt(2.0)
op_sy[1,0] = -1.0j/np.sqrt(2.0)
op_sy[1,2] = 1.0j/np.sqrt(2.0)
op_sy[2,1] = -1.0j/np.sqrt(2.0)

op_sz = np.zeros((3,3))
op_sz[0,0] = 1.0
op_sz[2,2] = -1.0

ene = np.zeros((N_UNIT,2))

mz = np.zeros(N_UNIT)
mx = np.zeros(N_UNIT)
my = np.zeros(N_UNIT)

start_obs = time.time()
for i in range(N_UNIT):
    ## site
    norm = Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_identity)
    mz[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_sz)/norm)
    mx[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_sx)/norm)
    my[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_sy)/norm)
    

    print "## Mag ",theta,i,norm,mx[i],my[i],mz[i],np.sqrt(mx[i]**2+my[i]**2+mz[i]**2)


for num in range(0,N_UNIT):        
    
    ## x direction
    num_j = NN_Tensor[num,2]
    norm_x = Contract_two_sites_holizontal_op12(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_identity_12)

    ene[num,0] = np.real(Contract_two_sites_holizontal_op12(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_ene_x)/norm_x)


    ## y direction
    num_j = NN_Tensor[num,3]        

    norm_y = Contract_two_sites_vertical_op12(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_identity_12)
    ene[num,1] = np.real(Contract_two_sites_vertical_op12(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_ene_y)/norm_y)


    print "## Ene", theta, num,norm_x,norm_y,ene[num,0],ene[num,1]

Energy = np.sum(ene)/N_UNIT
print "Enegy per site=",theta,Energy

sublatice_mag = np.zeros((3))
for i in range(N_UNIT/2):
    num = A_sub_list[i]
    sublatice_mag[0] += mx[num]
    sublatice_mag[1] += my[num]
    sublatice_mag[2] += mz[num]

    num = B_sub_list[i]
    sublatice_mag[0] -= mx[num]
    sublatice_mag[1] -= my[num]
    sublatice_mag[2] -= mz[num]
    

sublatice_mag /= N_UNIT
print "sublatice magnetization=",theta,sublatice_mag[0],sublatice_mag[1],sublatice_mag[2],np.sqrt(sublatice_mag[0]**2 + sublatice_mag[1]**2 + sublatice_mag[2]**2)

time_obs += time.time() - start_obs

    
print "## time simple update=",time_simple_update
print "## time full update=",time_full_update
print "## time environment=",time_env
print "## time observable=",time_obs
