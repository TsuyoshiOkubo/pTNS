# coding:utf-8
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import time

## import basic routines
from PEPS_Basics import *
from PEPS_Parameters import *
from Square_lattice_CTM import *

## timers
time_simple_update=0.0
time_full_update=0.0
time_env=0.0
time_obs=0.0

## Square lattice transeverse Ising
## Parameters
hx_min =1.4
d_hx = 0.02
hx_step = 1

tau = 0.01
tau_step = 1000

tau_full = 0.01
tau_full_step = 0

Initialize_every=False

## Tensors
Tn=[np.zeros((D,D,D,D,2),dtype=TENSOR_DTYPE)]
eTt=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
eTr=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
eTb=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
eTl=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]

C1=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
C2=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
C3=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
C4=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)

## lambda values from A
lambda_tensor = np.zeros((N_UNIT,4,D),dtype=TENSOR_DTYPE)

for i in range(1,N_UNIT):
    Tn.append(np.zeros((D,D,D,D,2),dtype=TENSOR_DTYPE))
    eTt.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTr.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTb.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTl.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    

def Set_Hamiltonian(hx):
    Ham = np.zeros((4,4))
    Ham[0,0] = -0.25 
    Ham[0,1] = -0.125 * hx 
    Ham[0,2] = -0.125 * hx 

    Ham[1,0] = -0.125 * hx
    Ham[1,1] = 0.25
    Ham[1,3] = -0.125 * hx

    Ham[2,0] = -0.125 * hx
    Ham[2,2] = 0.25
    Ham[2,3] = -0.125 * hx

    Ham[3,1] = -0.125 * hx
    Ham[3,2] = -0.125 * hx
    Ham[3,3] = -0.25

    return Ham

def Initialize_Tensors(Tn):
    ## ferro state with weak randomness

    np.random.seed(11)
    Tn_temp = 0.02* (np.random.rand(D,D,D,D,2)-0.5)

    for i in range(0,N_UNIT):
        Tn[i][:] = Tn_temp.copy()
        Tn[i][0,0,0,0,0] = 1.0
    
    

for int_hx in range(0,hx_step):
    ## Initialize tensor every step
    if (int_hx==0 or Initialize_every):
        Initialize_Tensors(Tn)
        lambda_tensor = np.ones((N_UNIT,4,D),dtype=float)
    
    hx = hx_min + int_hx * d_hx

    Ham = Set_Hamiltonian(hx)
    s,U = linalg.eigh(Ham)
    
    op12 =  np.dot(np.dot(U,np.diag(np.exp(-tau * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    start_simple=time.time()
    for int_tau in range(0,tau_step):

        ## simple update
        
        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c
            
        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c
            
        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## y-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,1]


            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c


        ## done simple update
    time_simple_update += time.time() - start_simple


    ## Start full update
    if tau_full_step > 0:
        Ham = Set_Hamiltonian(hx)
        s,U = linalg.eigh(Ham)

        op12 =  np.dot(np.dot(U,np.diag(np.exp(-tau_full * s))),U.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)


        ## Environment 
        start_env = time.time()
        Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
        time_env += time.time() - start_env

    start_full = time.time()
    for int_tau in range(0,tau_full_step):

        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12,2)

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

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12,2)

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

            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12,1)

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


            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12,1)

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

    op_identity = np.identity(2)

    op_mz = np.zeros((2,2))
    op_mz[0,0] = 0.5
    op_mz[1,1] = -0.5

    op_mx = np.zeros((2,2))
    op_mx[1,0] = 0.5
    op_mx[0,1] = 0.5

    mz = np.zeros(N_UNIT)
    mx = np.zeros(N_UNIT)

    zz = np.zeros((N_UNIT,2))
    start_obs = time.time()
    for i in range(0,N_UNIT):        
        norm = Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_identity)
        mz[i] = Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_mz)/norm
        mx[i] = Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_mx)/norm

        if Debug_flag:
            print hx,i,norm,mz[i],mx[i]
    for num in range(0,N_UNIT):
        num_j = NN_Tensor[num,2]

        ## x direction
        norm_x = Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_identity,op_identity)
        zz[num,0] = Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mz,op_mz)/norm_x


        ## y direction
        num_j = NN_Tensor[num,3]        

        norm_y = Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_identity,op_identity)
        zz[num,1] = Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mz,op_mz)/norm_y

        if Debug_flag:
            print hx,num,norm_x,norm_y,zz[num,0],zz[num,1]
        
    time_obs += time.time() - start_obs
        
    print hx, -np.sum(zz)/N_UNIT - hx * np.sum(mx)/N_UNIT,np.sum(mz)/N_UNIT,np.sum(mx)/N_UNIT,np.sum(zz)/N_UNIT


    
print "## time simple update=",time_simple_update
print "## time full update=",time_full_update
print "## time environment=",time_env
print "## time observable=",time_obs
