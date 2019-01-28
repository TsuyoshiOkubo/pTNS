# coding:utf-8
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg

## import basic routines
from PEPS_Basics import *
from PEPS_Parameters import *
from Lattice import *

def Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix):
    ## Do one step left move absoving X=ix column
    ## part of C1, C4, eTl will be modified
    
    for iy in range(0,LY):
        i = Tensor_list[ix,iy]
        j = NN_Tensor[i,2]
        k = NN_Tensor[j,3]
        l = NN_Tensor[k,0]
        if CTM_Projector_corner:
            PU, PL = Calc_projector_left_block(C1[i],C4[l],eTt[i],eTb[l],eTl[l],eTl[i],Tn[i],Tn[l])
        else:
            PU, PL = Calc_projector_updown_blocks(C1[i],C2[j],C3[k],C4[l],eTt[i],eTt[j],eTr[j],eTr[k],eTb[k],eTb[l],eTl[l],eTl[i],Tn[i],Tn[j],Tn[k],Tn[l])
            #print "PU left",PU[0,0,0,0],PU[0,0,0,1],PU[0,0,0,2],PU[0,0,0,3]
            #print "PL left",PL[0,0,0,0],PL[0,0,0,1],PL[0,0,0,2],PL[0,0,0,3]
        if iy == 0:
            PUs = [PU]
            PLs = [PL]
        else:
            PUs.append(PU)
            PLs.append(PL)

    ## update 
    C1_bak = C1.copy()
    C4_bak = C4.copy()
    eTl_bak =[]
    for num in range(N_UNIT):
        eTl_bak.append(eTl[num].copy())
        
    for iy in range(0,LY):
        i = Tensor_list[ix,iy]
        j = NN_Tensor[i,2]
        k = NN_Tensor[j,3]
        l = NN_Tensor[k,0]

        iy_up = (iy+1)%LY
        iy_down = (iy-1+LY)%LY

        C1[j],C4[k] = Calc_Next_CTM(C1_bak[i],C4_bak[l],eTt[i],eTb[l],PUs[iy_up],PLs[iy_down])

        
        eTl[j] = Calc_Next_eT(eTl_bak[i],Tn[i],PUs[iy],PLs[iy_up])
        eTl[k] = Calc_Next_eT(eTl_bak[l],Tn[l],PUs[iy_down],PLs[iy])

def Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix):
    ## Do one step right move absobing X=ix column
    ## part of C2, C3, eTr will be modified
    
    for iy in range(0,LY):
        k = Tensor_list[ix,iy]
        l = NN_Tensor[k,0]
        i = NN_Tensor[l,1]
        j = NN_Tensor[i,2] 

        if CTM_Projector_corner:
            PU, PL = Calc_projector_left_block(C3[k],C2[j],
                        eTb[k],eTt[j],eTr[j],eTr[k],
                            Tn[k].transpose(2,3,0,1,4),Tn[j].transpose(2,3,0,1,4))
        else:
            PU, PL = Calc_projector_updown_blocks(C3[k],C4[l],C1[i],C2[j],
                        eTb[k],eTb[l],eTl[l],eTl[i],eTt[i],eTt[j],eTr[j],eTr[k],
                            Tn[k].transpose(2,3,0,1,4),Tn[l].transpose(2,3,0,1,4),Tn[i].transpose(2,3,0,1,4),Tn[j].transpose(2,3,0,1,4))
        if iy == 0:
            PUs = [PU]
            PLs = [PL]
        else:
            PUs.append(PU)
            PLs.append(PL)

    ## update 
    C2_bak = C2.copy()
    C3_bak = C3.copy()
    eTr_bak =[]
    for num in range(N_UNIT):
        eTr_bak.append(eTr[num].copy())
    for iy in range(0,LY):
        k = Tensor_list[ix,iy]
        l = NN_Tensor[k,0]
        i = NN_Tensor[l,1]
        j = NN_Tensor[i,2] 

        iy_up = (iy + 1)%LY
        iy_down = (iy -1 + LY)%LY

        C3[l],C2[i] = Calc_Next_CTM(C3_bak[k],C2_bak[j],eTb[k],eTt[j],PUs[iy_down],PLs[iy_up])

        eTr[l] = Calc_Next_eT(eTr_bak[k],Tn[k].transpose(2,3,0,1,4),PUs[iy],PLs[iy_down])
        eTr[i] = Calc_Next_eT(eTr_bak[j],Tn[j].transpose(2,3,0,1,4),PUs[iy_up],PLs[iy])

def Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy):
    ## Do one step top move absobing Y=iy row
    ## part of C1, C2, eTt will be modified
    
    for ix in range(0,LX):
        j = Tensor_list[ix,iy]
        k = NN_Tensor[j,3]
        l = NN_Tensor[k,0]
        i = NN_Tensor[l,1] 

        if CTM_Projector_corner:
            PU, PL = Calc_projector_left_block(C2[j],C1[i],
                        eTr[j],eTl[i],eTt[i],eTt[j],
                                Tn[j].transpose(1,2,3,0,4),Tn[i].transpose(1,2,3,0,4))
        else:
            PU, PL = Calc_projector_updown_blocks(C2[j],C3[k],C4[l],C1[i],
                        eTr[j],eTr[k],eTb[k],eTb[l],eTl[l],eTl[i],eTt[i],eTt[j],
                                Tn[j].transpose(1,2,3,0,4),Tn[k].transpose(1,2,3,0,4),Tn[l].transpose(1,2,3,0,4),Tn[i].transpose(1,2,3,0,4))
            #print "PU top",PU[0,0,0,0],PU[0,0,0,1],PU[0,0,0,2],PU[0,0,0,3]
            #print "PL top",PL[0,0,0,0],PL[0,0,0,1],PL[0,0,0,2],PL[0,0,0,3]
        if ix == 0:
            PUs = [PU]
            PLs = [PL]
        else:
            PUs.append(PU)
            PLs.append(PL)

    ## update 
    C1_bak = C1.copy()
    C2_bak = C2.copy()
    eTt_bak =[]
    for num in range(N_UNIT):
        eTt_bak.append(eTt[num].copy())
    for ix in range(0,LX):
        j = Tensor_list[ix,iy]
        k = NN_Tensor[j,3]
        l = NN_Tensor[k,0]
        i = NN_Tensor[l,1] 

        ix_right = (ix + 1)%LX
        ix_left = (ix -1 + LX)%LX

        C2[k],C1[l] = Calc_Next_CTM(C2_bak[j],C1_bak[i],eTr[j],eTl[i],PUs[ix_right],PLs[ix_left])

        eTt[k] = Calc_Next_eT(eTt_bak[j],Tn[j].transpose(1,2,3,0,4),PUs[ix],PLs[ix_right])
        eTt[l] = Calc_Next_eT(eTt_bak[i],Tn[i].transpose(1,2,3,0,4),PUs[ix_left],PLs[ix])

def Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy):
    ## Do one step bottom move absobing Y=iy row
    ## part of C3, C4, eTb will be modified
    
    for ix in range(0,LX):
        l = Tensor_list[ix,iy]
        i = NN_Tensor[l,1] 
        j = NN_Tensor[i,2]
        k = NN_Tensor[j,3]

        if CTM_Projector_corner:
            PU, PL = Calc_projector_left_block(C4[l],C3[k],
                        eTl[l],eTr[k],eTb[k],eTb[l],
                                Tn[l].transpose(3,0,1,2,4),Tn[k].transpose(3,0,1,2,4))
        else:
            PU, PL = Calc_projector_updown_blocks(C4[l],C1[i],C2[j],C3[k],
                        eTl[l],eTl[i],eTt[i],eTt[j],eTr[j],eTr[k],eTb[k],eTb[l],
                                Tn[l].transpose(3,0,1,2,4),Tn[i].transpose(3,0,1,2,4),Tn[j].transpose(3,0,1,2,4),Tn[k].transpose(3,0,1,2,4))
        if ix == 0:
            PUs = [PU]
            PLs = [PL]
        else:
            PUs.append(PU)
            PLs.append(PL)

    ## update 
    C3_bak = C3.copy()
    C4_bak = C4.copy()
    eTb_bak =[]
    for num in range(N_UNIT):
        eTb_bak.append(eTb[num].copy())
    for ix in range(0,LX):
        l = Tensor_list[ix,iy]
        i = NN_Tensor[l,1] 
        j = NN_Tensor[i,2]
        k = NN_Tensor[j,3]

        ix_right = (ix + 1)%LX
        ix_left = (ix -1 + LX)%LX

        C4[i],C3[j] = Calc_Next_CTM(C4_bak[l],C3_bak[k],eTl[l],eTr[k],PUs[ix_left],PLs[ix_right])

        eTb[i] = Calc_Next_eT(eTb_bak[l],Tn[l].transpose(3,0,1,2,4),PUs[ix],PLs[ix_left])
        eTb[j] = Calc_Next_eT(eTb_bak[k],Tn[k].transpose(3,0,1,2,4),PUs[ix_right],PLs[ix])


def Check_Convergence_CTM(C1,C2,C3,C4,C1_old,C2_old,C3_old,C4_old):
    sig_max = 0.0
    convergence = True
    for i in range(0,N_UNIT):
        ## C1
        lam_new = linalg.svd(C1[i],compute_uv=False)
        lam_new /= np.sqrt(np.sum(lam_new**2))
        lam_old = linalg.svd(C1_old[i],compute_uv=False)
        lam_old /= np.sqrt(np.sum(lam_old**2))

        sig = np.sqrt(np.sum((lam_new - lam_old)**2))

        if sig > CTM_Convergence_Epsilon:
            sig_max = sig
            convergence = False
            break
        elif sig > sig_max:
            sig_max = sig

        ## C2
        lam_new = linalg.svd(C2[i],compute_uv=False)
        lam_new /= np.sqrt(np.sum(lam_new**2))
        lam_old = linalg.svd(C2_old[i],compute_uv=False)
        lam_old /= np.sqrt(np.sum(lam_old**2))

        sig = np.sqrt(np.sum((lam_new - lam_old)**2))
        
        if sig > CTM_Convergence_Epsilon:
            sig_max = sig
            convergence = False
            break
        elif sig > sig_max:
            sig_max = sig

        ## C3
        lam_new = linalg.svd(C3[i],compute_uv=False)
        lam_new /= np.sqrt(np.sum(lam_new**2))
        lam_old = linalg.svd(C3_old[i],compute_uv=False)
        lam_old /= np.sqrt(np.sum(lam_old**2))

        sig = np.sqrt(np.sum((lam_new - lam_old)**2))
        
        if sig > CTM_Convergence_Epsilon:
            sig_max = sig
            convergence = False
            break
        elif sig > sig_max:
            sig_max = sig

        ## C4
        lam_new = linalg.svd(C1[i],compute_uv=False)
        lam_new /= np.sqrt(np.sum(lam_new**2))
        lam_old = linalg.svd(C1_old[i],compute_uv=False)
        lam_old /= np.sqrt(np.sum(lam_old**2))

        sig = np.sqrt(np.sum((lam_new - lam_old)**2))
        
        if sig > CTM_Convergence_Epsilon:
            sig_max = sig
            convergence = False
            break
        elif sig > sig_max:
            sig_max = sig

    return convergence, sig_max
def Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,initialize=True):
    ## Calc environment tensors
    ## C1,C2,C3,C4 and eTt,eTl,eTr,eTb will be modified

    ## Initialize
    if initialize:
        for i in range(0,N_UNIT):
            C1[i] = np.zeros(C1[i].shape,dtype=TENSOR_DTYPE)
            num = NN_Tensor[NN_Tensor[i,0],1]
            d1 = Tn[num].shape[3]**2
            d2 = Tn[num].shape[2]**2
            C1[i][0:np.minimum(CHI,d1),0:np.minimum(CHI,d2)] = np.tensordot(
                Tn[num],Tn[num].conj(),([0,1,4],[0,1,4])).transpose(1,3,0,2)\
                .reshape(d1,d2)[0:np.minimum(CHI,d1),0:np.minimum(CHI,d2)] 
            #print "C1 init", C1[i][0,0], C1[i][1,0], C1[i][2,0], C1[i][3,0]

            C2[i] = np.zeros(C2[i].shape,dtype=TENSOR_DTYPE)
            num = NN_Tensor[NN_Tensor[i,2],1]
            d1 = Tn[num].shape[0]**2
            d2 = Tn[num].shape[3]**2
            C2[i][0:np.minimum(CHI,d1),0:np.minimum(CHI,d2)] = np.tensordot(
                Tn[num],Tn[num].conj(),([1,2,4],[1,2,4])).transpose(0,2,1,3)\
                .reshape(d1,d2)[0:np.minimum(CHI,d1),0:np.minimum(CHI,d2)] 

            C3[i] = np.zeros(C3[i].shape,dtype=TENSOR_DTYPE)
            num = NN_Tensor[NN_Tensor[i,2],3]
            d1 = Tn[num].shape[1]**2
            d2 = Tn[num].shape[0]**2
            C3[i][0:np.minimum(CHI,d1),0:np.minimum(CHI,d2)] = np.tensordot(
                Tn[num],Tn[num].conj(),([2,3,4],[2,3,4])).transpose(1,3,0,2)\
                .reshape(d1,d2)[0:np.minimum(CHI,d1),0:np.minimum(CHI,d2)] 

            C4[i] = np.zeros(C4[i].shape,dtype=TENSOR_DTYPE)
            num = NN_Tensor[NN_Tensor[i,0],3]
            d1 = Tn[num].shape[2]**2
            d2 = Tn[num].shape[1]**2
            C4[i][0:np.minimum(CHI,d1),0:np.minimum(CHI,d2)] = np.tensordot(
                Tn[num],Tn[num].conj(),([0,3,4],[0,3,4])).transpose(1,3,0,2)\
                .reshape(d1,d2)[0:np.minimum(CHI,d1),0:np.minimum(CHI,d2)] 

            eTt[i] = np.zeros(eTt[i].shape,dtype=TENSOR_DTYPE)
            num = NN_Tensor[i,1]
            d1 = Tn[num].shape[0]**2
            d2 = Tn[num].shape[2]**2
            d34 = Tn[num].shape[3]
            eTt[i][0:np.minimum(CHI,d1),0:np.minimum(CHI,d2),0:d34,0:d34] = np.tensordot(
                Tn[num],Tn[num].conj(),([1,4],[1,4])).transpose(0,3,1,4,2,5)\
                .reshape(d1,d2,d34,d34)[0:np.minimum(CHI,d1),0:np.minimum(CHI,d2),0:d34,0:d34]

            eTr[i] = np.zeros(eTr[i].shape,dtype=TENSOR_DTYPE)
            num = NN_Tensor[i,2]
            d1 = Tn[num].shape[1]**2
            d2 = Tn[num].shape[3]**2
            d34 = Tn[num].shape[0]
            eTr[i][0:np.minimum(CHI,d1),0:np.minimum(CHI,d2),0:d34,0:d34] = np.tensordot(
                Tn[num],Tn[num].conj(),([2,4],[2,4])).transpose(1,4,2,5,0,3)\
                .reshape(d1,d2,d34,d34)[0:np.minimum(CHI,d1),0:np.minimum(CHI,d2),0:d34,0:d34]

            eTb[i] = np.zeros(eTb[i].shape,dtype=TENSOR_DTYPE)
            num = NN_Tensor[i,3]
            d1 = Tn[num].shape[2]**2
            d2 = Tn[num].shape[0]**2
            d34 = Tn[num].shape[1]
            eTb[i][0:np.minimum(CHI,d1),0:np.minimum(CHI,d2),0:d34,0:d34] = np.tensordot(
                Tn[num],Tn[num].conj(),([3,4],[3,4])).transpose(2,5,0,3,1,4)\
                .reshape(d1,d2,d34,d34)[0:np.minimum(CHI,d1),0:np.minimum(CHI,d2),0:d34,0:d34]

            eTl[i] = np.zeros(eTl[i].shape,dtype=TENSOR_DTYPE)
            num = NN_Tensor[i,0]
            d1 = Tn[num].shape[3]**2
            d2 = Tn[num].shape[1]**2
            d34 = Tn[num].shape[2]
            eTl[i][0:np.minimum(CHI,d1),0:np.minimum(CHI,d2),0:d34,0:d34] = np.tensordot(
                Tn[num],Tn[num].conj(),([0,4],[0,4])).transpose(2,5,0,3,1,4)\
                .reshape(d1,d2,d34,d34)[0:np.minimum(CHI,d1),0:np.minimum(CHI,d2),0:d34,0:d34]

    ### Initialize done

    convergence = False
    count = 0
    sig_max = 0.0
    C1_old = C1.copy()
    C2_old = C2.copy()
    C3_old = C3.copy()
    C4_old = C4.copy()
    ##eTt_old = eTt
    ##eTr_old = eTr
    ##eTb_old = eTb
    ##eTl_old = eTl
    
    while (not convergence) and (count < Max_CTM_Iteration) :
        ## left move
        for ix in range(0,LX):
            Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
        ## right move
        for ix in range(0,-LX,-1):
            Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,(ix+1 + LX)%LX)
        ## top move
        for iy in range(0,-LY,-1):
            Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,(iy+1 + LY)%LY)
        ## bottom move
        for iy in range(0,LY):
            Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)

        convergence, sig_max = Check_Convergence_CTM(C1,C2,C3,C4,C1_old,C2_old,C3_old,C4_old)
        count += 1

        C1_old = C1.copy()
        C2_old = C2.copy()
        C3_old = C3.copy()
        C4_old = C4.copy()
        if Debug_flag:
            print "## CTM: count, sig_max",count, sig_max
        
    if not convergence:
        print "## Warning CTM: CMTs did not converge! count, sig_max = ", count,sig_max
    if Debug_flag:
        print "## CTM: count to convergence=",count

    #return C1,C2,C3,C4,eTt,eTr,eTb,eTl
