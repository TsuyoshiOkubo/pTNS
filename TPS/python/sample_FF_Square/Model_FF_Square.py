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

## Fully Frastrated Square lattice Heisenberg
## for Experiment 
## J1 = -17.7K   = -1.0
## J2 = 13.2K    = 0.7458
## J3 = 11.0K    = 0.6215
## J4 = 11.0K    = 0.6215
## J5 = -7.3K    = -0.4124
## J6 = -5.6K    = -0.3164

## Parameters
J1 = -1.0 
J2 = 0.7458
J3 = 0.6215
J4 = 0.6215
J5 = -0.4124
J6 = -0.3164


hz_min =0.0
d_hz = 0.02
hz_step = 1

tau = 0.01
tau_step = 1000

tau_full = 0.01
tau_full_step = 10

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
lambda_tensor = np.zeros((N_UNIT,4,D),dtype=float)

for i in range(1,N_UNIT):
    Tn.append(np.zeros((D,D,D,D,2),dtype=TENSOR_DTYPE))
    eTt.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTr.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTb.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTl.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    

def Set_Hamiltonian(hz,J):
    Ham = np.zeros((4,4))
    Ham[0,0] = 0.25 * J - hz * 0.25

    Ham[1,1] = -0.25 * J
    Ham[1,2] = 0.5 * J

    Ham[2,1] = 0.5 * J
    Ham[2,2] = -0.25 * J

    Ham[3,3] = 0.25 * J + hz * 0.25
    
    return Ham

def Initialize_Tensors(Tn):
    ## Random tensors

    np.random.seed(11)
    
    for i in range(0,N_UNIT):
        if isinstance(Tn[i][0,0,0,0,0],complex):
            Tn_temp_real = (np.random.rand(D,D,D,D,2)-0.5)
            Tn_temp_imag = (np.random.rand(D,D,D,D,2)-0.5)
            Tn[i][:] = Tn_temp_real + 1.0j * Tn_temp_imag
        else:
            Tn[i][:] = (np.random.rand(D,D,D,D,2)-0.5)
    ## anti ferro state with weak randomness

    #np.random.seed(11)
    #if isinstance(Tn[0][0,0,0,0,0],complex):
    #    Tn_temp = 0.02* ((np.random.rand(D,D,D,D,2)-0.5) + 1.0j * (np.random.rand(D,D,D,D,2)-0.5))
    #else:
    #    Tn_temp = 0.0 2* (np.random.rand(D,D,D,D,2)-0.5)
    #for i in range(0,N_UNIT):        
    #    Tn[i][:] = Tn_temp.copy()
    #    ix = i % LX
    #    iy = i / LY
    #    if ((ix + iy)%2 == 0):
    #        Tn[i][0,0,0,0,0] = 1.0
    #    else:
    #        Tn[i][0,0,0,0,1] = 1.0

for int_hz in range(0,hz_step):
    ## Initialize tensor every step
    if (int_hz==0 or Initialize_every):
        Initialize_Tensors(Tn)
        lambda_tensor = np.ones((N_UNIT,4,D),dtype=float)
    
    hz = hz_min + int_hz * d_hz

    Ham_J1 = Set_Hamiltonian(hz,J1)
    Ham_J2 = Set_Hamiltonian(hz,J2)
    Ham_J3 = Set_Hamiltonian(hz,J3)
    Ham_J4 = Set_Hamiltonian(hz,J4)
    Ham_J5 = Set_Hamiltonian(hz,J5)
    Ham_J6 = Set_Hamiltonian(hz,J6)


    s_J1,U_J1 = linalg.eigh(Ham_J1)
    s_J2,U_J2 = linalg.eigh(Ham_J2)
    s_J3,U_J3 = linalg.eigh(Ham_J3)
    s_J4,U_J4 = linalg.eigh(Ham_J4)
    s_J5,U_J5 = linalg.eigh(Ham_J5)
    s_J6,U_J6 = linalg.eigh(Ham_J6)
    

    op12_J1 =  np.dot(np.dot(U_J1,np.diag(np.exp(-tau * s_J1))),U_J1.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12_J2 =  np.dot(np.dot(U_J2,np.diag(np.exp(-tau * s_J2))),U_J2.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12_J3 =  np.dot(np.dot(U_J3,np.diag(np.exp(-tau * s_J3))),U_J3.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12_J4 =  np.dot(np.dot(U_J4,np.diag(np.exp(-tau * s_J4))),U_J4.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12_J5 =  np.dot(np.dot(U_J5,np.diag(np.exp(-tau * s_J5))),U_J5.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12_J6 =  np.dot(np.dot(U_J6,np.diag(np.exp(-tau * s_J6))),U_J6.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)

    start_simple=time.time()


    for int_tau in range(0,tau_step):

        ## simple update
        
        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/4):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J2,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c
            
        ## x-bond D sub-lattice
        for i in range(0,N_UNIT/4):
            num = D_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J1,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c
            
        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/4):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J5,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c

        ## x-bond C sub-lattice
        for i in range(0,N_UNIT/4):
            num = C_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J6,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c
            

        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/4):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J4,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## y-bond D sub-lattice
        for i in range(0,N_UNIT/4):
            num = D_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J4,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## y-bond B sub-lattice
        for i in range(0,N_UNIT/4):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J3,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## y-bond C sub-lattice
        for i in range(0,N_UNIT/4):
            num = C_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J3,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c
            

        ## done simple update
    time_simple_update += time.time() - start_simple


    ## Start full update
    if tau_full_step > 0:
        op12_J1 =  np.dot(np.dot(U_J1,np.diag(np.exp(-tau_full * s_J1))),U_J1.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
        op12_J2 =  np.dot(np.dot(U_J2,np.diag(np.exp(-tau_full * s_J2))),U_J2.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
        op12_J3 =  np.dot(np.dot(U_J3,np.diag(np.exp(-tau_full * s_J3))),U_J3.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
        op12_J4 =  np.dot(np.dot(U_J4,np.diag(np.exp(-tau_full * s_J4))),U_J4.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
        op12_J5 =  np.dot(np.dot(U_J5,np.diag(np.exp(-tau_full * s_J5))),U_J5.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
        op12_J6 =  np.dot(np.dot(U_J6,np.diag(np.exp(-tau_full * s_J6))),U_J6.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)


        ## Environment 
        start_env = time.time()
        Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
        time_env += time.time() - start_env

    start_full = time.time()
    for int_tau in range(0,tau_full_step):

        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/4):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_J2,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
            
        ## x-bond D sub-lattice
        for i in range(0,N_UNIT/4):
            num = D_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_J1,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/4):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_J5,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## x-bond C sub-lattice
        for i in range(0,N_UNIT/4):
            num = C_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_J6,2)

            if Full_Use_FFU:
                ix = num%LX
                ix_j = num_j%LX
                Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
                
        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/4):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_J4,1)

            if Full_Use_FFU:
                iy = num/LX
                iy_j = num_j/LX
                Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)


        ## y-bond D sub-lattice
        for i in range(0,N_UNIT/4):
            num = D_sub_list[i]
            num_j = NN_Tensor[num,1]


            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_J4,1)

            if Full_Use_FFU:
                iy = num/LX
                iy_j = num_j/LX
                Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## y-bond B sub-lattice
        for i in range(0,N_UNIT/4):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,1]


            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_J3,1)

            if Full_Use_FFU:
                iy = num/LX
                iy_j = num_j/LX
                Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
            else:
                Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        ## y-bond C sub-lattice
        for i in range(0,N_UNIT/4):
            num = C_sub_list[i]
            num_j = NN_Tensor[num,1]


            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_J3,1)

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

    op_my = np.zeros((2,2),dtype=complex)
    op_my[1,0] = 0.5j
    op_my[0,1] = -0.5j
    
    mz = np.zeros(N_UNIT)
    mx = np.zeros(N_UNIT)
    my = np.zeros(N_UNIT)

    zz = np.zeros((N_UNIT,2))
    xx = np.zeros((N_UNIT,2))
    yy = np.zeros((N_UNIT,2))

    start_obs = time.time()
    for i in range(0,N_UNIT):        
        norm = Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_identity)
        mz[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_mz)/norm)
        mx[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_mx)/norm)
        my[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_my)/norm)


        print "## Mag ",hz,i,norm,mx[i],my[i],mz[i],np.sqrt(mx[i]**2+my[i]**2+mz[i]**2)
    for num in range(0,N_UNIT):
        num_j = NN_Tensor[num,2]

        ## x direction
        norm_x = Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_identity,op_identity)
        zz[num,0] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mz,op_mz)/norm_x)
        xx[num,0] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mx,op_mx)/norm_x)
        yy[num,0] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_my,op_my)/norm_x)


        ## y direction
        num_j = NN_Tensor[num,3]        

        norm_y = Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_identity,op_identity)
        zz[num,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mz,op_mz)/norm_y)
        xx[num,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mx,op_mx)/norm_y)
        yy[num,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_my,op_my)/norm_y)


        print "## Dot", hz,num,norm_x,norm_y,xx[num,0],yy[num,0],zz[num,0],xx[num,1],yy[num,1],zz[num,1]
        
    time_obs += time.time() - start_obs


    Energy_J1 = 0.0
    Energy_J2 = 0.0
    Energy_J3 = 0.0
    Energy_J4 = 0.0
    Energy_J5 = 0.0
    Energy_J6 = 0.0

    for num in range(0,N_UNIT/4):
        num_a = A_sub_list[num]
        num_b = B_sub_list[num]
        num_c = C_sub_list[num]
        num_d = D_sub_list[num]
             
        ## J1
        Energy_J1 += (xx[num_d,0] + yy[num_d,0] + zz[num_d,0])


        ## J2
        Energy_J2 += (xx[num_a,0] + yy[num_a,0] + zz[num_a,0])

        ## memo 2018/7/6 J3とJ4が入れ替わっている気がする。要修正
        ## J3
        Energy_J3 += (xx[num_b,1] + yy[num_b,1] + zz[num_b,1])
        Energy_J3 += (xx[num_c,1] + yy[num_c,1] + zz[num_c,1])

        ## J4
        Energy_J4 += (xx[num_a,1] + yy[num_a,1] + zz[num_a,1])
        Energy_J4 += (xx[num_d,1] + yy[num_d,1] + zz[num_d,1])
        
        ## J5
        Energy_J5 += (xx[num_b,0] + yy[num_b,0] + zz[num_b,0])

        ## J6
        Energy_J6 += (xx[num_c,0] + yy[num_c,0] + zz[num_c,0])


    print hz, (J1 * Energy_J1 + J2 * Energy_J2 + J3 * Energy_J3 + J4 * Energy_J4 + J5 * Energy_J5 + J6 * Energy_J6 - hz * np.sum(mz))/N_UNIT, np.sum(mz)/N_UNIT,Energy_J1,Energy_J2,Energy_J3,Energy_J4,Energy_J5,Energy_J6

    
print "## time simple update=",time_simple_update
print "## time full update=",time_full_update
print "## time environment=",time_env
print "## time observable=",time_obs
