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
N_ = 2
m_ = 1
J = -1.0/N_
##
D_phys = scipy.special.comb(N_ + m_ - 1,m_,exact=True)

Initial_VBS = False#columner VBS only work for m=1
Random_amp = 0.02


second_ST = False#True
tau = 0.01
tau_step = 1000

tau_full = 0.01
tau_full_step = 0

Initialize_every=False

## Tensors
Tn=[np.zeros((D,D,D,D,D_phys),dtype=TENSOR_DTYPE)]
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
    Tn.append(np.zeros((D,D,D,D,D_phys),dtype=TENSOR_DTYPE))
    eTt.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTr.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTb.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    eTl.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))

def Create_table_P(N_in,m_in):
    import math as math
    table_P = np.zeros((N_in+1,m_in+1),dtype=int)
    for N in range(1,N_in+1):
        table_P[N,1] = N

    if m_in >= 2:
        for m in range(2,m_in+1):
            for N in range(1,N_in+1):
                table_P[N,m] = table_P[N-1,m] + table_P[N,m-1]

    #print "table_P=",table_P
    Hilbert_dim = table_P[N_in,m_in]
    extended_dim = N_in**m_in
    N_box = np.zeros((Hilbert_dim,m_in),dtype=int) 
    Extend_table = np.zeros((extended_dim),dtype=int)
    Extend_factor = np.zeros((extended_dim),dtype=float)

    N_box_local = np.zeros((m_in+1),dtype=int)

    #make permutation table
    permutation_num = math.factorial(m_in)
    permutation_table = np.zeros((permutation_num,m_in),dtype=int)
    for j in range(permutation_num):
        buff = np.arange(m_in)            
        present_j = j
        for m in range(m_in-1):      
            denom = math.factorial(m_in-1-m)
            step = present_j / denom
            temp = buff[m + step]
            for k in range(m + step,m,-1):
                buff[k] = buff[k-1]
            buff[m] = temp
            present_j %= denom
        #print "j=",j,buff
        permutation_table[j,:] = buff[:]

    for i in range(Hilbert_dim):
        N_max = N_in
        base_N = 0
        for m in range(m_in,0,-1):
            for N in range(1,N_max+1):
                if i < base_N + table_P[N,m]:
                    N_box_local[m] = N-1
                    base_N += table_P[N-1,m]
                    N_max = N
                    break
        N_box[i,0:m_in] = N_box_local[1:m_in+1]

        ## permutation
        N_factor = np.zeros((N_in),dtype=int)
        for m in range(m_in):
            N_factor[N_box[i,m]] += 1
        factor = 1
        for N in range(N_in):
            if N_factor[N] >0:
                factor *= math.factorial(N_factor[N])
        factor = math.factorial(m_in)/factor

        for perm in range(permutation_num):
            num_extend = 0
            for m in range(m_in):
                num_extend += N_box[i,permutation_table[perm,m]] * N_in**m
            
            Extend_table[num_extend] = i
            Extend_factor[num_extend] = 1.0/np.sqrt(factor)
            #print "extend",N_box[i,:],num_extend
                
        #print "factor",i,N_factor,factor
        #print "i=",N_box[i,0:m_in],N_box_local[1:m_in+1]
            

                   
    #print table_P[N_in,m_in],scipy.special.comb(N_in + m_in - 1,m_in,exact=True)
    return N_box,Extend_table,Extend_factor

def Set_Hamiltonian(J,N_in,m_in):
    import math as math
    #print "N_in,m_in",N_in,m_in
    D_phys_single = N_in
    ## permutaion information
    N_box_list,Extend_table,Extend_factor = Create_table_P(N_in,m_in)
    Ham_single = np.zeros((D_phys_single**2,D_phys_single**2))
    for n in range(N_in):
        i = n * D_phys_single + n
        for m in range(N_in):
            j = m * D_phys_single + m
            Ham_single[i,j] = J        
    if m_in ==1:
        Ham = Ham_single
    elif m_in > 1:
        #Ham_one = np.ones((D_phys_single**2,D_phys_single**2),dtype=float)
        Ham_one = np.identity((D_phys_single**2),dtype=float)

        Ham = np.zeros((N_in**(2*m_in),N_in**(2*m_in)),dtype=float)
        Ham_extended = Ham_single
        for i in range(m_in-1):
            Ham_extended = np.kron(Ham_extended,Ham_one)
        #print "Ham_extended",Ham_extended
        reshape_list = [N_in]
        for i in range(4*m_in-1):
            reshape_list.append(N_in)
        Ham_extended = Ham_extended.reshape(reshape_list)
        #print "Ham_extended_reshape",Ham_extended


        for m1 in range(m_in):
            for m2 in range(m_in):                
                transpose_list = [2*m1]
                for m in range(m_in):
                    if m != m1:
                        transpose_list.append(2*m)
                transpose_list.append(2*m2 + 1)
                for m in range(m_in):
                    if m != m2:
                        transpose_list.append(2*m+1)

                transpose_list.append(2*m1 +2*m_in)
                for m in range(m_in):
                    if m != m1:
                        transpose_list.append(2*m+2*m_in)
                transpose_list.append(2*m2 + 1+2*m_in)
                for m in range(m_in):
                    if m != m2:
                        transpose_list.append(2*m+1+2*m_in)

                #print "transpose_list",transpose_list
                Ham += Ham_extended.transpose(transpose_list).reshape(N_in**(2*m_in),N_in**(2*m_in)) 
            
        #print "Ham",Ham


        #project onto Bose particle space (size  (N+m-1)_C_m
        Hilbert_dim = N_box_list.shape[0]
        extended_dim = N_in**m_in
        Projector_single = np.zeros((Hilbert_dim,extended_dim))
        for i in range(extended_dim):
            Hilbert_num = Extend_table[i]
            Projector_single[Hilbert_num,i] = Extend_factor[i]
        #print "Projector_single",Projector_single
        Projector = np.kron(Projector_single,Projector_single)
        
        #print "Projector",Projector

        Ham = np.dot(np.dot(Projector,Ham),Projector.T)
    else:
        " m >= 1 is required!"
        sys.exit()

    #print "Ham_small",Ham

    #e = linalg.eigvalsh(Ham)
    #print "eigenvalues",e[Ham.shape[0]-1]
    return Ham,N_box_list


def Initialize_Tensors(Tn,initial_VBS=False, random_amp=0.02):
    ## Random tensors

    D_phys= Tn[0].shape[4]
    D = Tn[0].shape[0]
    np.random.seed(11)
    if initial_VBS:
        for ix in range(LX):
            for iy in range(LY):
                i = Tensor_list[ix,iy]
                if isinstance(Tn[i][0,0,0,0,0],complex):
                    Tn[i][:] = Random_amp* ((np.random.rand(D,D,D,D,D_phys)-0.5) + 1.0j * (np.random.rand(D,D,D,D,D_phys)-0.5))
                else:
                    Tn[i][:] = Random_amp* (np.random.rand(D,D,D,D,D_phys)-0.5)

                if (D >= N_):
                    for n in range(N_):
                        if ix%2 ==0:
                            Tn[i][0,0,n,0,n] = 1.0
                        else:
                            Tn[i][n,0,0,0,n] = 1.0
                else:
                    for n in range(D):
                        if ix%2 ==0:
                            Tn[i][0,0,n,0,n] = 1.0
                        else:
                            Tn[i][n,0,0,0,n] = 1.0
        
    else:
        for i in range(N_UNIT):
            if isinstance(Tn[i][0,0,0,0,0],complex):
                Tn[i][:] = Random_amp* ((np.random.rand(D,D,D,D,D_phys)-0.5) + 1.0j * (np.random.rand(D,D,D,D,D_phys)-0.5))
            else:
                Tn[i][:] = Random_amp* (np.random.rand(D,D,D,D,D_phys)-0.5)
        
            Tn[i][0,0,0,0,0] = 1.0



Initialize_Tensors(Tn,initial_VBS=Initial_VBS, random_amp=Random_amp)
lambda_tensor = np.ones((N_UNIT,4,D),dtype=float)

Ham_J,N_box_list = Set_Hamiltonian(J,N_,m_)

s,U = linalg.eigh(Ham_J)

op12_J =  np.dot(np.dot(U,np.diag(np.exp(-tau * s))),U.conj().T).reshape(D_phys,D_phys,D_phys,D_phys).transpose(2,3,0,1)
op12_J_2 =  np.dot(np.dot(U,np.diag(np.exp(-tau*0.5 * s))),U.conj().T).reshape(D_phys,D_phys,D_phys,D_phys).transpose(2,3,0,1)

start_simple=time.time()


for int_tau in range(0,tau_step):
    if second_ST:
        ## simple update
        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J_2,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J_2,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c


        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J_2,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## y-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J_2,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J_2,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c

        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J_2,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c

    else:
        ## simple update
        ## x-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c

        ## x-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,2]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J,2)
            lambda_tensor[num,2] = lambda_c
            lambda_tensor[num_j,0] = lambda_c


        ## y-bond A sub-lattice
        for i in range(0,N_UNIT/2):
            num = A_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c

        ## y-bond B sub-lattice
        for i in range(0,N_UNIT/2):
            num = B_sub_list[i]
            num_j = NN_Tensor[num,1]

            Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_J,1)
            lambda_tensor[num,1] = lambda_c
            lambda_tensor[num_j,3] = lambda_c


    ## done simple update
time_simple_update += time.time() - start_simple


## Start full update
if tau_full_step > 0:
    op12_J =  np.dot(np.dot(U,np.diag(np.exp(-tau_full * s))),U.conj().T).reshape(D_phys,D_phys,D_phys,D_phys).transpose(2,3,0,1)
    op12_J_2 =  np.dot(np.dot(U,np.diag(np.exp(-tau_full*0.5 * s))),U.conj().T).reshape(D_phys,D_phys,D_phys,D_phys).transpose(2,3,0,1)


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

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_J_2,2)

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

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_J_2,2)

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

            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_J_2,1)

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


            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_J,1)

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

            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_J_2,1)

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

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_J_2,2)

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

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_J_2,2)

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

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_J,2)

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

            Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_J,2)

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

            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_J,1)

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


            Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_J,1)

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

op_identity = np.identity(D_phys)
op_identity_12 = np.identity(D_phys**2).reshape(D_phys,D_phys,D_phys,D_phys)
op_ene = Ham_J.reshape(D_phys,D_phys,D_phys,D_phys)

ene = np.zeros((N_UNIT,2))

start_obs = time.time()
for num in range(0,N_UNIT):        
    num_j = NN_Tensor[num,2]

    ## x direction
    norm_x = Contract_two_sites_holizontal_op12(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_identity_12)

    ene[num,0] = np.real(Contract_two_sites_holizontal_op12(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_ene)/norm_x)


    ## y direction
    num_j = NN_Tensor[num,3]        

    norm_y = Contract_two_sites_vertical_op12(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_identity_12)
    ene[num,1] = np.real(Contract_two_sites_vertical_op12(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_ene)/norm_y)


    print "## ene", num,norm_x,norm_y,ene[num,0],ene[num,1]

Energy = np.sum(ene)/N_UNIT
print "Enegy per site=",Energy
time_obs += time.time() - start_obs

    
print "## time simple update=",time_simple_update
print "## time full update=",time_full_update
print "## time environment=",time_env
print "## time observable=",time_obs
