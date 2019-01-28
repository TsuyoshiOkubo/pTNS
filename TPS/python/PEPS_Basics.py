
# coding:utf-8
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import scipy.linalg.interpolative 

from PEPS_Parameters import *
## Basic routines independent on unit cell structures

## Contractions

def Contract_one_site(C1,C2,C3,C4,eT1,eT2,eT3,eT4,Tn1,op1):
##############################
# (((((C2*(C3*eT2))*(C1*eT1))*(((C4*eT4)*eT3)*Tn1c))*Tn1)*op1)
# cpu_cost= 6.04e+10  memory= 3.0207e+08
# final_bond_order  ()
##############################
    return np.tensordot(
                np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                C2, np.tensordot(
                                    C3, eT2, ([0], [1])
                                ), ([1], [1])
                            ), np.tensordot(
                                C1, eT1, ([1], [0])
                            ), ([0], [1])
                        ), np.tensordot(
                            np.tensordot(
                                np.tensordot(
                                    C4, eT4, ([1], [0])
                                ), eT3, ([0], [1])
                            ), np.conjugate(Tn1), ([2, 5], [0, 3])
                        ), ([0, 2, 3, 5], [2, 5, 0, 4])
                    ), Tn1, ([0, 1, 2, 3], [2, 1, 0, 3])
                ), op1, ([0, 1], [1, 0])
            )

def Contract_two_sites_holizontal(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,Tn1,Tn2,op1,op2):
##   |  |
##  -T1-T2-
##   |  |
    ##############################
    # ((Tn1*(((C1*eT1)*(C4*eT6))*(Tn1c*(((C2*eT2)*(Tn2*(((C3*eT4)*eT3)*(Tn2c*op2))))*eT5))))*op1)
    # cpu_cost= 1.204e+11  memory= 3.041e+08
    # final_bond_order  ()
    ##############################
    return np.tensordot(
                np.tensordot(
                    Tn1, np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                C1, eT1, ([1], [0])
                            ), np.tensordot(
                                C4, eT6, ([1], [0])
                            ), ([0], [1])
                        ), np.tensordot(
                            np.conjugate(Tn1), np.tensordot(
                                np.tensordot(
                                    np.tensordot(
                                        C2, eT2, ([0], [1])
                                    ), np.tensordot(
                                        Tn2, np.tensordot(
                                            np.tensordot(
                                                np.tensordot(
                                                    C3, eT4, ([1], [0])
                                                ), eT3, ([0], [1])
                                            ), np.tensordot(
                                                np.conjugate(Tn2), op2, ([4], [1])
                                            ), ([2, 5], [3, 2])
                                        ), ([2, 3, 4], [3, 1, 6])
                                    ), ([0, 2, 3], [3, 1, 5])
                                ), eT5, ([2], [0])
                            ), ([2, 3], [2, 5])
                        ), ([0, 2, 3, 5], [3, 1, 5, 0])
                    ), ([0, 1, 2, 3], [1, 0, 3, 4])
                ), op1, ([0, 1], [0, 1])
            )


def Contract_two_sites_vertical(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,Tn1,Tn2,op1,op2):
##   |
##  -T1-
##   |
##  -T2-
##   |
##############################
# ((Tn1*(((C2*eT2)*(C1*eT1))*(Tn1c*(((C3*eT3)*(Tn2*(((C4*eT5)*eT4)*(Tn2c*op2))))*eT6))))*op1)
# cpu_cost= 1.204e+11  memory= 3.0411e+08
# final_bond_order  ()
##############################
    return np.tensordot(
                np.tensordot(
                    Tn1, np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                C2, eT2, ([1], [0])
                            ), np.tensordot(
                                C1, eT1, ([1], [0])
                            ), ([0], [1])
                        ), np.tensordot(
                            np.conjugate(Tn1), np.tensordot(
                                np.tensordot(
                                    np.tensordot(
                                        C3, eT3, ([0], [1])
                                    ), np.tensordot(
                                        Tn2, np.tensordot(
                                            np.tensordot(
                                                np.tensordot(
                                                    C4, eT5, ([1], [0])
                                                ), eT4, ([0], [1])
                                            ), np.tensordot(
                                                np.conjugate(Tn2), op2, ([4], [1])
                                            ), ([2, 5], [0, 3])
                                        ), ([0, 3, 4], [1, 3, 6])
                                    ), ([0, 2, 3], [3, 1, 5])
                                ), eT6, ([2], [0])
                            ), ([0, 3], [5, 2])
                        ), ([0, 2, 3, 5], [3, 1, 5, 0])
                    ), ([0, 1, 2, 3], [4, 1, 0, 3])
                ), op1, ([0, 1], [0, 1])
            )

def Contract_two_sites_holizontal_op12(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,Tn1,Tn2,op12):
    ##   |  |
    ##  -T1-T2-
    ##   |  |
    ##############################
    # two_sites_holizontal_op12.dat
    ##############################
    # (op12*((eT1*(Tn1*(Tn1c*(eT5*(C1*(C4*eT6))))))*(eT2*(Tn2*(Tn2c*(eT4*(C2*(C3*eT3))))))))
    # cpu_cost= 2.20416e+11  memory= 6.0502e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        op12, np.tensordot(
            np.tensordot(
                eT1, np.tensordot(
                    Tn1, np.tensordot(
                        np.conjugate(Tn1), np.tensordot(
                            eT5, np.tensordot(
                                C1, np.tensordot(
                                    C4, eT6, ([1], [0])
                                ), ([0], [1])
                            ), ([1], [1])
                        ), ([0, 3], [5, 2])
                    ), ([0, 3], [6, 4])
                ), ([0, 2, 3], [7, 0, 3])
            ), np.tensordot(
                eT2, np.tensordot(
                    Tn2, np.tensordot(
                        np.conjugate(Tn2), np.tensordot(
                            eT4, np.tensordot(
                                C2, np.tensordot(
                                    C3, eT3, ([0], [1])
                                ), ([1], [1])
                            ), ([0], [1])
                        ), ([2, 3], [5, 2])
                    ), ([2, 3], [6, 4])
                ), ([1, 2, 3], [7, 1, 4])
            ), ([0, 1, 3, 5], [0, 1, 3, 5])
        ), ([0, 1, 2, 3], [0, 2, 1, 3])
    )

def Contract_two_sites_vertical_op12(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,Tn1,Tn2,op12):
    ##   |
    ##  -T1-
    ##   |
    ##  -T2-
    ##   |
    ##############################
    # two_sites_vertical_op12.dat
    ##############################
    # (op12*((eT2*(Tn1*(Tn1c*(eT6*(C1*(C2*eT1))))))*(eT3*(Tn2*(Tn2c*(eT5*(C3*(C4*eT4))))))))
    # cpu_cost= 2.20416e+11  memory= 6.0502e+08
    # final_bond_order ()
    ##############################
    return np.tensordot(
        op12, np.tensordot(
            np.tensordot(
                eT2, np.tensordot(
                    Tn1, np.tensordot(
                        np.conjugate(Tn1), np.tensordot(
                            eT6, np.tensordot(
                                C1, np.tensordot(
                                    C2, eT1, ([0], [1])
                                ), ([1], [1])
                            ), ([1], [0])
                        ), ([0, 1], [2, 5])
                    ), ([0, 1], [4, 6])
                ), ([0, 2, 3], [7, 0, 3])
            ), np.tensordot(
                eT3, np.tensordot(
                    Tn2, np.tensordot(
                        np.conjugate(Tn2), np.tensordot(
                            eT5, np.tensordot(
                                C3, np.tensordot(
                                    C4, eT4, ([0], [1])
                                ), ([1], [1])
                            ), ([0], [1])
                        ), ([0, 3], [2, 5])
                    ), ([0, 3], [4, 6])
                ), ([1, 2, 3], [7, 1, 4])
            ), ([0, 1, 3, 5], [0, 1, 3, 5])
        ), ([0, 1, 2, 3], [0, 2, 1, 3])
    )

def Contract_four_sites(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,eT7,eT8,Tn1,Tn2,Tn3,Tn4,op1,op2,op3,op4):
##############################
# ((((C1*eT1)*eT8)*Tn1c)*(op1*Tn1))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (e12, e78, tc12, tc41, t12, t41)
##############################
    LT = np.tensordot(
        np.tensordot(
            np.tensordot(
                np.tensordot(
                    C1, eT1, ([1], [0])
                ), eT8, ([0], [1])
            ), Tn1.conj(), ([2, 5], [1, 0])
        ), np.tensordot(
            op1, Tn1, ([0], [4])
        ), ([1, 3, 6], [2, 1, 0])
    )

##############################
# ((((C2*eT3)*eT2)*Tn2c)*(op2*Tn2))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (e34, e12, tc12, tc23, t12, t23)
##############################
    RT = np.tensordot(
        np.tensordot(
            np.tensordot(
                np.tensordot(
                    C2, eT3, ([1], [0])
                ), eT2, ([0], [1])
            ), Tn2.conj(), ([2, 5], [2, 1])
        ), np.tensordot(
            op2, Tn2, ([0], [4])
        ), ([1, 3, 6], [3, 2, 0])
    )

##INFO:584 (2,1) Finish 368/584 script=[1, 3, -1, 2, -1, 5, -1, 0, 4, -1, -1]
##############################
# ((((C3*eT5)*eT4)*Tn3c)*(op3*Tn3))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (e56, e34, tc34, tc23, t34, t23)
##############################
    RB = np.tensordot(
        np.tensordot(
            np.tensordot(
                np.tensordot(
                    C3, eT5, ([1], [0])
                ), eT4, ([0], [1])
            ), Tn3.conj(), ([2, 5], [3, 2])
        ), np.tensordot(
            op3, Tn3, ([0], [4])
        ), ([1, 3, 6], [4, 3, 0])
    )

##############################
# ((((C4*eT7)*eT6)*Tn4c)*(op4*Tn4))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (e78, e56, tc41, tc34, t41, t34)
##############################
    LB = np.tensordot(
        np.tensordot(
            np.tensordot(
                np.tensordot(
                    C4, eT7, ([1], [0])
                ), eT6, ([0], [1])
            ), Tn4.conj(), ([2, 5], [0, 3])
        ), np.tensordot(
            op4, Tn4, ([0], [4])
        ), ([1, 3, 6], [1, 4, 0])
    )


##INFO:9999 (2,11) Finish 0/9999 script=[0, 1, -1, 2, -1, 3, -1]
##############################
# (((LT*RT)*RB)*LB)
# cpu_cost= 2.0001e+12  memory= 5e+08
# final_bond_order  ()
##############################
    return np.tensordot(
        np.tensordot(
            np.tensordot(
                LT, RT, ([0, 2, 4], [1, 2, 4])
            ), RB, ([3, 4, 5], [1, 3, 5])
        ), LB, ([0, 1, 2, 3, 4, 5], [0, 2, 4, 1, 3, 5])
    )

def Contract_1x3(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,eT7,eT8,Tn1,Tn2,Tn3,op1,op2,op3):
##############################
# contract_1x3.dat
##############################
# (op1*(Tn1*((eT1*(C1*eT8))*(Tn1c*((C2*eT2)*(eT3*(Tn2*((Tn2c*op2)*(eT7*(eT4*(Tn3*((Tn3c*op3)*(eT6*(C3*(C4*eT5)))))))))))))))
# cpu_cost= 1.804e+11  memory= 4.0413e+08
# final_bond_order ()
##############################
    return np.tensordot(
        op1, np.tensordot(
            Tn1, np.tensordot(
                np.tensordot(
                    eT1, np.tensordot(
                        C1, eT8, ([0], [1])
                    ), ([0], [0])
                ), np.tensordot(
                    Tn1.conj(), np.tensordot(
                        np.tensordot(
                            C2, eT2, ([1], [0])
                        ), np.tensordot(
                            eT3, np.tensordot(
                                Tn2, np.tensordot(
                                    np.tensordot(
                                        Tn2.conj(), op2, ([4], [1])
                                    ), np.tensordot(
                                        eT7, np.tensordot(
                                            eT4, np.tensordot(
                                                Tn3, np.tensordot(
                                                    np.tensordot(
                                                        Tn3.conj(), op3, ([4], [1])
                                                    ), np.tensordot(
                                                        eT6, np.tensordot(
                                                            C3, np.tensordot(
                                                                C4, eT5, ([0], [1])
                                                            ), ([1], [1])
                                                        ), ([0], [1])
                                                    ), ([0, 3], [2, 5])
                                                ), ([0, 3, 4], [4, 6, 2])
                                            ), ([1, 2, 3], [5, 1, 3])
                                        ), ([0], [3])
                                    ), ([0, 3], [2, 5])
                                ), ([0, 3, 4], [4, 6, 2])
                            ), ([1, 2, 3], [5, 1, 3])
                        ), ([1], [0])
                    ), ([2, 3], [2, 4])
                ), ([0, 2, 3, 5], [3, 1, 6, 0])
            ), ([0, 1, 2, 3], [1, 0, 3, 4])
        ), ([0, 1], [0, 1])
    )


def Contract_2x3(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,eT7,eT8,eT9,eT10,Tn1,Tn2,Tn3,Tn4,Tn5,Tn6,op1,op2,op3,op4,op5,op6):
## following contraction is optimized by assuming honeycomb lattice structure
##############################
# contract_2x3.dat
##############################
# (op1*(Tn1*(Tn2*((eT1*eT2)*((Tn1c*(Tn2c*op2))*(((Tn3*Tn4)*(((Tn3*op3)*(Tn4c*op4))*(eT6*((C2*eT3)*(eT4*(C3*eT5))))))*((Tn5*Tn6)*(((Tn5c*op5)*(Tn6c*op6))*(eT7*((C1*eT10)*(eT9*(C4*eT8))))))))))))
# cpu_cost= 1.27041e+12  memory= 7.00124e+08
# final_bond_order ()
##############################
    return np.tensordot(
        op1, np.tensordot(
            Tn1, np.tensordot(
                Tn2, np.tensordot(
                    np.tensordot(
                        eT1, eT2, ([1], [0])
                    ), np.tensordot(
                        np.tensordot(
                            Tn1.conj(), np.tensordot(
                                Tn2.conj(), op2, ([4], [1])
                            ), ([2], [0])
                        ), np.tensordot(
                            np.tensordot(
                                np.tensordot(
                                    Tn3, Tn4, ([3], [1])
                                ), np.tensordot(
                                    np.tensordot(
                                        np.tensordot(
                                            Tn3.conj(), op3, ([4], [1])
                                        ), np.tensordot(
                                            Tn4.conj(), op4, ([4], [1])
                                        ), ([3], [1])
                                    ), np.tensordot(
                                        eT6, np.tensordot(
                                            np.tensordot(
                                                C2, eT3, ([1], [0])
                                            ), np.tensordot(
                                                eT4, np.tensordot(
                                                    C3, eT5, ([0], [1])
                                                ), ([1], [1])
                                            ), ([1], [0])
                                        ), ([0], [5])
                                    ), ([2, 5, 6], [7, 9, 2])
                                ), ([2, 3, 5, 6, 7], [10, 2, 11, 6, 4])
                            ), np.tensordot(
                                np.tensordot(
                                    Tn5, Tn6, ([1], [3])
                                ), np.tensordot(
                                    np.tensordot(
                                        np.tensordot(
                                            Tn5.conj(), op5, ([4], [1])
                                        ), np.tensordot(
                                            Tn6.conj(), op6, ([4], [1])
                                        ), ([1], [3])
                                    ), np.tensordot(
                                        eT7, np.tensordot(
                                            np.tensordot(
                                                C1, eT10, ([0], [1])
                                            ), np.tensordot(
                                                eT9, np.tensordot(
                                                    C4, eT8, ([1], [0])
                                                ), ([0], [1])
                                            ), ([1], [0])
                                        ), ([1], [5])
                                    ), ([0, 2, 4], [9, 2, 7])
                                ), ([0, 2, 3, 4, 7], [11, 6, 1, 10, 4])
                            ), ([0, 2, 3, 5, 6], [2, 0, 5, 3, 6])
                        ), ([0, 2, 5, 6], [9, 6, 4, 1])
                    ), ([0, 2, 3, 5], [8, 0, 5, 2])
                ), ([1, 2, 3, 4], [1, 5, 4, 3])
            ), ([0, 1, 2, 3], [4, 1, 0, 3])
        ), ([0, 1], [0, 1])
    )
    
    
    
## environment
def Calc_projector_corner(C1,C4,eT1,eT6,eT7,eT8,Tn1,Tn4):
## based on R. Orus and G. Vidal Phys. Rev. B 80, 094403 (2009)
    e12 = eT1.shape[1]
    e56 = eT6.shape[0]
    e78 = eT8.shape[0]
    t12 = Tn1.shape[2]
    t41 = Tn1.shape[3]
    t34 = Tn4.shape[2]

    if t41 != 1:
##INFO:104 (2,3) Finish 72/104 script=[3, 0, 1, -1, 2, -1, 4, -1, -1]
##############################
# (Tn1*(((C1*eT1)*eT8)*Tn1c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t12, t41, e12, e78, tc12, tc41)
##############################
        LT = np.tensordot(
            Tn1, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C1, eT1, ([1], [0])
                    ), eT8, ([0], [1])
                ), Tn1.conj(), ([2, 5], [1, 0])
            ), ([0, 1, 4], [3, 1, 6])
        ).transpose(3,1,5,2,0,4).reshape(e78*t41**2,e12*t12**2)
    

##INFO:104 (2,3) Finish 74/104 script=[3, 0, 2, -1, 1, -1, 4, -1, -1]
##############################
# (Tn4*(((C4*eT7)*eT6)*Tn4c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t41, t34, e78, e56, tc41, tc34)
##############################
        LB = np.tensordot(
            Tn4, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C4, eT7, ([1], [0])
                    ), eT6, ([0], [1])
                ), Tn4.conj(), ([2, 5], [0, 3])
            ), ([0, 3, 4], [1, 3, 6])
        ).transpose(2,0,4,3,1,5).reshape(e78*t41**2,e56*t34**2)

        def Mult_vec(vec):
            return np.tensordot(np.tensordot(vec,LT,(0,0)),LT.conj(),(0,1)) + np.tensordot(np.tensordot(vec,LB.conj(),(0,0)),LB,(0,1))

        def Mult_mat(mat):
            return np.tensordot(LT.conj(),np.tensordot(LT,mat,(0,0)),(1,0)) +np.tensordot(LB,np.tensordot(LB.conj(),mat,(0,0)),(1,0)) 
        
        MatVec = spr_linalg.LinearOperator((e78*t41**2,e78*t41**2),matvec=Mult_vec, matmat=Mult_mat,dtype=LT.dtype)
        eig,Vec = spr_linalg.eigsh(MatVec,k=e78) #cost O (chi^10)

        PL = Vec.reshape(e78,t41,t41,e78)
        PU = PL.conj().copy()

    else:
        PU = np.identity(e78).reshape(e78,t41,t41,e78)
        PL = PU.copy()
        
    return PU,PL

def Calc_projector_left_block(C1,C4,eT1,eT6,eT7,eT8,Tn1,Tn4):
## Original (cheaper version of P. Corboz, T.M.Rice and M. Troyer, PRL 113, 046402(2014))

    e12 = eT1.shape[1]
    e56 = eT6.shape[0]
    e78 = eT8.shape[0]
    t12 = Tn1.shape[2]
    t41 = Tn1.shape[3]
    t34 = Tn4.shape[2]

    if t41 != 1:
##INFO:104 (2,3) Finish 72/104 script=[3, 0, 1, -1, 2, -1, 4, -1, -1]
##############################
# (Tn1*(((C1*eT1)*eT8)*Tn1c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t12, t41, e12, e78, tc12, tc41)
##############################
        LT = np.tensordot(
            Tn1, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C1, eT1, ([1], [0])
                    ), eT8, ([0], [1])
                ), Tn1.conj(), ([2, 5], [1, 0])
            ), ([0, 1, 4], [3, 1, 6])
        ).transpose(3,1,5,2,0,4).reshape(e78*t41**2,e12*t12**2)
    

##INFO:104 (2,3) Finish 74/104 script=[3, 0, 2, -1, 1, -1, 4, -1, -1]
##############################
# (Tn4*(((C4*eT7)*eT6)*Tn4c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t41, t34, e78, e56, tc41, tc34)
##############################
        LB = np.tensordot(
            Tn4, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C4, eT7, ([1], [0])
                    ), eT6, ([0], [1])
                ), Tn4.conj(), ([2, 5], [0, 3])
            ), ([0, 3, 4], [1, 3, 6])
        ).transpose(2,0,4,3,1,5).reshape(e78*t41**2,e56*t34**2)

        ## if e78*t41**2 < e12*t12**2:
        ##     ## QR decomposition
        ##     R1 = linalg.qr(LT.T,mode="r")[0:e78*t41**2,:]
        ## else:
        ##     R1 = LT.T

        ## if e78*t41**2 < e56*t34**2:
        ##     ## QR decomposition
        ##     R2 = linalg.qr(LB.T,mode="r")[0:e78*t41**2,:]
        ## else:
        ##     R2 = LB.T

        R1 = LT.T
        R2 = LB.T

        if (t12 != 1 and t34 != 1) and (Use_Partial_SVD or Use_Interporative_SVD):

            def Mult_vec(vec):
                return np.tensordot(R1,np.tensordot(R2,vec,(0,0)),(1,0))

            def R_Mult_vec(vec):
                return np.tensordot(R2.conj(),np.tensordot(R1.conj(),vec,(0,0)),(1,0))

            def Mult_mat(mat):
                return np.tensordot(R1,np.tensordot(R2,mat,(0,0)),(1,0))


            MatVec = spr_linalg.LinearOperator((R1.shape[0],R2.shape[0]),matvec=Mult_vec,rmatvec=R_Mult_vec,matmat=Mult_mat,dtype=R1.dtype)

            if Use_Interporative_SVD:
                U,s,V = linalg.interpolative.svd(MatVec,eps_or_k=2*e78) #cost O (chi^10) 
                s = s[:e78]
                U = U[:,:e78]
                VT= V[:,:e78].conj().T
            else:
                U,s,VT = spr_linalg.svds(MatVec,k=e78) #cost O (chi^10)
            
                
        else:
            ## full svd
            U,s,VT = linalg.svd(np.tensordot(R1,R2,(1,1)),full_matrices=False)

            ## truncation
            U = U[:,0:e78]
            VT = VT[0:e78,:]
            s = s[0:e78]

        ## Orthogonalization
        if Use_Orthogonalization:
            U,R = linalg.qr(U,mode="economic")
            VT = np.dot(R,VT)
        
        denom = s[0]
        for x in np.nditer(s,flags = [ 'buffered'],op_flags=['readwrite']):
            if (x/denom > Inverse_projector_cut):
                x[...] = 1.0/np.sqrt(x)
            else:
                x[...] = 0.0
        ## O(D^{10})
        PU = np.tensordot(R2,np.tensordot(VT.conj(),np.diag(s),(0,0)),(0,0)).reshape(e78,t41,t41,e78)
        PL = np.tensordot(R1,np.tensordot(U.conj(),np.diag(s),(1,0)),(0,0)).reshape(e78,t41,t41,e78)

    else:
        PU = np.identity(e78).reshape(e78,t41,t41,e78)
        PL = PU.copy()
        
    return PU,PL

def Calc_projector_updown_blocks(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,eT7,eT8,Tn1,Tn2,Tn3,Tn4):
## based on P. Corboz, T.M.Rice and M. Troyer, PRL 113, 046402(2014)

    e12 = eT1.shape[1]
    e34 = eT3.shape[1]
    e56 = eT5.shape[1]
    e78 = eT7.shape[1]
    t12 = Tn1.shape[2]
    t23 = Tn2.shape[3]
    t34 = Tn3.shape[0]
    t41 = Tn4.shape[1]

    if t41 != 1:
##INFO:104 (2,3) Finish 72/104 script=[3, 0, 1, -1, 2, -1, 4, -1, -1]
##############################
# (Tn1*(((C1*eT1)*eT8)*Tn1c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t12, t41, e12, e78, tc12, tc41)
##############################
        LT = np.tensordot(
            Tn1, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C1, eT1, ([1], [0])
                    ), eT8, ([0], [1])
                ), Tn1.conj(), ([2, 5], [1, 0])
            ), ([0, 1, 4], [3, 1, 6])
        ).transpose(3,1,5,2,0,4).reshape(e78*t41**2,e12*t12**2)
    

##INFO:104 (2,3) Finish 74/104 script=[3, 0, 2, -1, 1, -1, 4, -1, -1]
##############################
# (Tn2*(((C2*eT3)*eT2)*Tn2c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t12, t23, e34, e12, tc12, tc23)
##############################
        RT = np.tensordot(
            Tn2, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C2, eT3, ([1], [0])
                    ), eT2, ([0], [1])
                ), Tn2.conj(), ([2, 5], [2, 1])
            ), ([1, 2, 4], [3, 1, 6])
        ).transpose(3,0,4,2,1,5).reshape(e12*t12**2,e34*t23**2)

##INFO:104 (2,3) Finish 74/104 script=[3, 0, 2, -1, 1, -1, 4, -1, -1]
##############################
# (Tn3*(((C3*eT5)*eT4)*Tn3c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t34, t23, e56, e34, tc34, tc23)
##############################
        RB = np.tensordot(
            Tn3, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C3, eT5, ([1], [0])
                    ), eT4, ([0], [1])
                ), Tn3.conj(), ([2, 5], [3, 2])
            ), ([2, 3, 4], [3, 1, 6])
        ).transpose(2,0,4,3,1,5).reshape(e56*t34**2,e34*t23**2)


##INFO:104 (2,3) Finish 74/104 script=[3, 0, 2, -1, 1, -1, 4, -1, -1]
##############################
# (Tn4*(((C4*eT7)*eT6)*Tn4c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t41, t34, e78, e56, tc41, tc34)
##############################
        LB = np.tensordot(
            Tn4, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C4, eT7, ([1], [0])
                    ), eT6, ([0], [1])
                ), Tn4.conj(), ([2, 5], [0, 3])
            ), ([0, 3, 4], [1, 3, 6])
        ).transpose(2,0,4,3,1,5).reshape(e78*t41**2,e56*t34**2)

        
        if t23 != 1 and (Use_Partial_SVD or Use_Interporative_SVD):    
            def Mult_vec(vec):
                return np.tensordot(RT,np.tensordot(LT,np.tensordot(LB,np.tensordot(RB,vec,(1,0)),(1,0)),(0,0)),(0,0))

            def R_Mult_vec(vec):
                return np.tensordot(RB.conj(),np.tensordot(LB.conj(),np.tensordot(LT.conj(),np.tensordot(RT.conj(),vec,(1,0)),(1,0)),(0,0)),(0,0))

            def Mult_mat(mat):
                return np.tensordot(RT,np.tensordot(LT,np.tensordot(LB,np.tensordot(RB,mat,(1,0)),(1,0)),(0,0)),(0,0))


            MatVec = spr_linalg.LinearOperator((e34*t23**2,e34*t23**2),matvec=Mult_vec,rmatvec=R_Mult_vec,matmat=Mult_mat,dtype=RT.dtype)

            if Use_Interporative_SVD:
                U,s,V = linalg.interpolative.svd(MatVec,eps_or_k=2*e78) #cost O (chi^10) 
                s = s[:e78]
                U = U[:,:e78]
                VT= V[:,:e78].conj().T
            else:
                U,s,VT = spr_linalg.svds(MatVec,k=e78) #cost O (chi^10)


            ## Orthogonalization
            if Use_Orthogonalization:
                U,R = linalg.qr(U,mode="economic")
                VT = np.dot(R,VT)
                
            denom = s[0]
            for x in np.nditer(s,flags = [ 'buffered'],op_flags=['readwrite']):
                if (x/denom > Inverse_projector_cut):
                    x[...] = 1.0/np.sqrt(x)
                else:
                    x[...] = 0.0
            ## O(D^{10})
            PU = np.tensordot(LB,np.tensordot(RB,np.tensordot(VT.conj(),np.diag(s),(0,0)),(1,0)),(1,0)).reshape(e78,t41,t41,e78)
            PL = np.tensordot(LT,np.tensordot(RT,np.tensordot(U.conj(),np.diag(s),(1,0)),(1,0)),(1,0)).reshape(e78,t41,t41,e78)
        else:
            ## full svd 
            R1 = np.tensordot(RT,LT,(0,1))
            R2 = np.tensordot(RB,LB,(0,1))

            U,s,VT = linalg.svd(np.tensordot(R1,R2,(1,1)),full_matrices=False) #cost O (chi^12)

            #truncation 
            U = U[:,0:e78]
            VT = VT[0:e78,:]
            s = s[0:e78]

            ## Orthogonalization
            if Use_Orthogonalization:
                U,R = linalg.qr(U,mode="economic")
                VT = np.dot(R,VT)

            denom = s[0]
            for x in np.nditer(s,flags = [ 'buffered'],op_flags=['readwrite']):
                if (x/denom > Inverse_projector_cut):
                    x[...] = 1.0/np.sqrt(x)
                else:
                    x[...] = 0.0
            ## O(D^{10})
            PU = np.tensordot(R2,np.tensordot(VT.conj(),np.diag(s),(0,0)),(0,0)).reshape(e78,t41,t41,e78)
            PL = np.tensordot(R1,np.tensordot(U.conj(),np.diag(s),(1,0)),(0,0)).reshape(e78,t41,t41,e78)

    else:
        PU = np.identity(e78).reshape(e78,t41,t41,e78)
        PL = PU.copy()
        
    return PU,PL
    
def Calc_projector_SVD_Corboz(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,eT7,eT8,Tn1,Tn2,Tn3,Tn4):
## based on P. Corboz, T.M.Rice and M. Troyer, PRL 113, 046402(2014)

    e12 = eT1.shape[1]
    e34 = eT3.shape[1]
    e56 = eT5.shape[1]
    e78 = eT7.shape[1]
    t12 = Tn1.shape[2]
    t23 = Tn2.shape[3]
    t34 = Tn3.shape[0]
    t41 = Tn4.shape[1]

    if t41 != 1:
##INFO:104 (2,3) Finish 72/104 script=[3, 0, 1, -1, 2, -1, 4, -1, -1]
##############################
# (Tn1*(((C1*eT1)*eT8)*Tn1c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t12, t41, e12, e78, tc12, tc41)
##############################
        LT = np.tensordot(
            Tn1, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C1, eT1, ([1], [0])
                    ), eT8, ([0], [1])
                ), Tn1.conj(), ([2, 5], [1, 0])
            ), ([0, 1, 4], [3, 1, 6])
        ).transpose(3,1,5,2,0,4).reshape(e78*t41**2,e12*t12**2)

##INFO:104 (2,3) Finish 74/104 script=[3, 0, 2, -1, 1, -1, 4, -1, -1]
##############################
# (Tn2*(((C2*eT3)*eT2)*Tn2c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t12, t23, e34, e12, tc12, tc23)
##############################
        RT = np.tensordot(
            Tn2, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C2, eT3, ([1], [0])
                    ), eT2, ([0], [1])
                ), Tn2.conj(), ([2, 5], [2, 1])
            ), ([1, 2, 4], [3, 1, 6])
        ).transpose(3,0,4,2,1,5).reshape(e12*t12**2,e34*t23**2)

##INFO:104 (2,3) Finish 74/104 script=[3, 0, 2, -1, 1, -1, 4, -1, -1]
##############################
# (Tn3*(((C3*eT5)*eT4)*Tn3c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t34, t23, e56, e34, tc34, tc23)
##############################
        RB = np.tensordot(
            Tn3, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C3, eT5, ([1], [0])
                    ), eT4, ([0], [1])
                ), Tn3.conj(), ([2, 5], [3, 2])
            ), ([2, 3, 4], [3, 1, 6])
        ).transpose(2,0,4,3,1,5).reshape(e56*t34**2,e34*t23**2)


##INFO:104 (2,3) Finish 74/104 script=[3, 0, 2, -1, 1, -1, 4, -1, -1]
##############################
# (Tn4*(((C4*eT7)*eT6)*Tn4c))
# cpu_cost= 5.01e+10  memory= 3.0004e+08
# final_bond_order  (t41, t34, e78, e56, tc41, tc34)
##############################
        LB = np.tensordot(
            Tn4, np.tensordot(
                np.tensordot(
                    np.tensordot(
                        C4, eT7, ([1], [0])
                    ), eT6, ([0], [1])
                ), Tn4.conj(), ([2, 5], [0, 3])
            ), ([0, 3, 4], [1, 3, 6])
        ).transpose(2,0,4,3,1,5).reshape(e78*t41**2,e56*t34**2)


        if Use_Partial_SVD or Use_Interporative_SVD:
            ## partial svd
            def Mult_vec(vec):
                return np.tensordot(LT,np.tensordot(RT,np.tensordot(RB,np.tensordot(LB,vec,(0,0)),(0,0)),(1,0)),(1,0))

            def R_Mult_vec(vec):
                return np.tensordot(LB,np.tensordot(RB,np.tensordot(RT,np.tensordot(LT,vec,(0,0)),(0,0)),(1,0)),(1,0))

            def Mult_mat(mat):
                return np.tensordot(LT,np.tensordot(RT,np.tensordot(RB,np.tensordot(LB,mat,(0,0)),(0,0)),(1,0)),(1,0))


            MatVec = spr_linalg.LinearOperator((e78*t41**2,e78*t41**2),matvec=Mult_vec,rmatvec=R_Mult_vec,matmat=Mult_mat,dtype=RT.dtype)

            if Use_Interporative_SVD:
                U,s,V = linalg.interpolative.svd(MatVec,eps_or_k=2*e78) #cost O (chi^10) 
                s = s[:e78]
                U = U[:,:e78]
                VT= V[:,:e78].conj().T
            else:
                U,s,VT = spr_linalg.svds(MatVec,k=e78) #cost O (chi^10)

        else:
            ## full svd
            R1 = np.tensordot(RT,LT,(0,1))
            R2 = np.tensordot(RB,LB,(0,1))

            U,s,VT = linalg.svd(np.tensordot(R1,R2,(0,0)),full_matrices=False) #cost O (chi^10)

            #truncation 
            U = U[:,0:e78]


        ## Orthogonalization
        if Use_Orthogonalization:
            U,R = linalg.qr(U,mode="economic")
            VT = np.dot(R,VT)
            
        PL = U.reshape(e78,t41,t41,e78)
        PU = PL.conj().copy()

    else:
        PU = np.identity(e78).reshape(e78,t41,t41,e78)
        PL = PU.copy()
        
    return PU,PL


def Calc_Next_CTM(C1,C4,eT1,eT6,PU,PL):
    C1_out = np.tensordot(PU,np.tensordot(C1,eT1,(1,0)),([0,1,2],[0,2,3]))
    C4_out = np.tensordot(np.tensordot(eT6,C4,(1,0)),PL,([3,1,2],[0,1,2]))

    C1_out /= np.nanmax(np.absolute(C1_out))
    C4_out /= np.nanmax(np.absolute(C4_out))

    #w = linalg.svd(C1_out,compute_uv=False)
    #C1_out /= np.sum(w)
    #w = linalg.svd(C4_out,compute_uv=False)
    #C4_out /= np.sum(w)

    return C1_out,C4_out

def Calc_Next_eT(eT8,Tn1,PU,PL):
##############################
# ((Tn1*(Tn1c*(eT8*PL)))*PU)
# cpu_cost= 6e+10  memory= 3.0104e+08
# final_bond_order  (k2, l2, n1, n0)
##############################
    eT_out = np.tensordot(
            np.tensordot(
                Tn1, np.tensordot(
                    Tn1.conj(), np.tensordot(
                        eT8, PL, ([1], [0])
                    ), ([0, 1], [2, 4])
                ), ([0, 1, 4], [4, 5, 2])
            ), PU, ([1, 3, 4], [1, 2, 0])
        ).transpose(3,2,0,1)

    eT_out /= np.nanmax(np.absolute(eT_out))

    #eT_temp = np.trace(eT_out,axis1=2,axis2=3)
    #w = linalg.svd(eT_temp,compute_uv=False)
    #eT_out /= np.sum(w)
    
    return eT_out

## for simple update
def Simple_update_bond(Tn1,Tn2,lambda1,lambda2,op12,connect1,D_cut=None):
    ## --前処理--
    if connect1==0:
        connect2 = 2
    elif connect1 ==1:
        connect2 = 3
    elif connect1 ==2:
        connect2 = 0
    else:
        connect2 = 1
    
    
    lambda1_inv = [np.zeros(lambda1[0].shape)]
    lambda2_inv = [np.zeros(lambda2[0].shape)]
    for i in range(1,4):
        lambda1_inv.append(np.zeros(lambda1[i].shape))
        lambda2_inv.append(np.zeros(lambda2[i].shape))

    for i in range(4):
        it = np.nditer([lambda1[i],lambda1_inv[i]],flags = [ 'buffered'],op_flags = [['readonly'],['writeonly', 'no_broadcast']])
        for x,y in it:
            if x > Inverse_lambda_cut:
                y[...] = 1.0/x

        it = np.nditer([lambda2[i],lambda2_inv[i]],flags = [ 'buffered'],op_flags = [['readonly'],['writeonly', 'no_broadcast']])
        for x,y in it:
            if x > Inverse_lambda_cut:
                y[...] = 1.0/x
    

    m1 = Tn1.shape[4]
    m2 = Tn2.shape[4]
    dc = Tn1.shape[connect1]
    if D_cut is None:
        D_cut = dc
    if dc != Tn2.shape[connect2]:
        print "## error:bad connection at simple update",connect1,connect2,Tn1.shape[connect1],Tn2.shape[connect2]
    ## add lambda
    if connect1==0:
        Tn1_lambda = np.tensordot(np.tensordot(np.tensordot(Tn1,np.diag(lambda1[1]),(1,0)),np.diag(lambda1[2]),(1,0)),np.diag(lambda1[3]),(1,0)).transpose(2,3,4,0,1)
    elif connect1==1:
        Tn1_lambda = np.tensordot(np.tensordot(np.tensordot(Tn1,np.diag(lambda1[0]),(0,0)),np.diag(lambda1[2]),(1,0)),np.diag(lambda1[3]),(1,0)).transpose(2,3,4,0,1)
    elif connect1==2:
        Tn1_lambda = np.tensordot(np.tensordot(np.tensordot(Tn1,np.diag(lambda1[0]),(0,0)),np.diag(lambda1[1]),(0,0)),np.diag(lambda1[3]),(1,0)).transpose(2,3,4,0,1)
    else :
        Tn1_lambda = np.tensordot(np.tensordot(np.tensordot(Tn1,np.diag(lambda1[0]),(0,0)),np.diag(lambda1[1]),(0,0)),np.diag(lambda1[2]),(0,0)).transpose(2,3,4,0,1)

    if connect2==0:
        Tn2_lambda = np.tensordot(np.tensordot(np.tensordot(Tn2,np.diag(lambda2[1]),(1,0)),np.diag(lambda2[2]),(1,0)),np.diag(lambda2[3]),(1,0)).transpose(2,3,4,0,1)
    elif connect2==1:
        Tn2_lambda = np.tensordot(np.tensordot(np.tensordot(Tn2,np.diag(lambda2[0]),(0,0)),np.diag(lambda2[2]),(1,0)),np.diag(lambda2[3]),(1,0)).transpose(2,3,4,0,1)
    elif connect2==2:
        Tn2_lambda = np.tensordot(np.tensordot(np.tensordot(Tn2,np.diag(lambda2[0]),(0,0)),np.diag(lambda2[1]),(0,0)),np.diag(lambda2[3]),(1,0)).transpose(2,3,4,0,1)
    else :
        Tn2_lambda = np.tensordot(np.tensordot(np.tensordot(Tn2,np.diag(lambda2[0]),(0,0)),np.diag(lambda2[1]),(0,0)),np.diag(lambda2[2]),(0,0)).transpose(2,3,4,0,1)

        
    ## QR
    if (Tn1_lambda.shape[0]*Tn1_lambda.shape[1]*Tn1_lambda.shape[2] >= Tn1_lambda.shape[3]*Tn1_lambda.shape[4]):
        Q1,R1 = linalg.qr(Tn1_lambda.reshape(Tn1_lambda.shape[0]*Tn1_lambda.shape[1]*Tn1_lambda.shape[2],Tn1_lambda.shape[3]*Tn1_lambda.shape[4]),mode="economic")
    else:
        Q1 = np.identity(Tn1_lambda.shape[0]*Tn1_lambda.shape[1]*Tn1_lambda.shape[2])
        R1 = Tn1_lambda.reshape(Tn1_lambda.shape[0]*Tn1_lambda.shape[1]*Tn1_lambda.shape[2],Tn1_lambda.shape[3]*Tn1_lambda.shape[4])

    if (Tn2_lambda.shape[0]*Tn2_lambda.shape[1]*Tn2_lambda.shape[2] >= Tn2_lambda.shape[3]*Tn2_lambda.shape[4]):
        Q2,R2 = linalg.qr(Tn2_lambda.reshape(Tn2_lambda.shape[0]*Tn2_lambda.shape[1]*Tn2_lambda.shape[2],Tn2_lambda.shape[3]*Tn2_lambda.shape[4]),mode="economic")
    else:
        Q2 = np.identity(Tn2_lambda.shape[0]*Tn2_lambda.shape[1]*Tn2_lambda.shape[2])
        R2 = Tn2_lambda.reshape(Tn2_lambda.shape[0]*Tn2_lambda.shape[1]*Tn2_lambda.shape[2],Tn2_lambda.shape[3]*Tn2_lambda.shape[4])
    
    ## connetc R1, R2, op
##############################
# ((R1*R2)*op12)
# cpu_cost= 22400  memory= 3216
# final_bond_order  (c1, c2, m1o, m2o)
##############################
    Theta = np.tensordot(
            np.tensordot(
                R1.reshape(R1.shape[0],dc,m1), R2.reshape(R2.shape[0],dc,m2), ([1], [1]) 
            ), op12, ([1, 3], [0, 1])
        )

    ## svd
    
    try:
        U,s,VT = linalg.svd(Theta.transpose(0,2,1,3).reshape(R1.shape[0]*m1,R2.shape[0]*m2),full_matrices=False) 
    except:
        #--using eigen value problem routine--
        if (R1.shape[0]*m1 >= R2.shape[0]*m2):
            Mat_temp = Theta.transpose(0,2,1,3).reshape(R1.shape[0]*m1,R2.shape[0]*m2)
            e, V = linalg.eigh(np.dot(Mat_temp.T.conj(),Mat_temp))
            e = e[::-1]
            V = V[:,::-1]
            s = np.zeros(e.size)
            max_e = np.nanmax(e)
            for i in range(0,e.size):
                if (e[i]/max_e > 1e-20):
                    s[i] = np.sqrt(e[i])
                    e[i] = 1.0/np.sqrt(e[i])
                else:
                    s[i] = 0.0
                    e[i] = 0.0

            U = np.dot(Mat_temp,V)
            U = np.dot(U,np.diag(e))
            VT = np.conj(V.T)
        else:
            Mat_temp = Theta.transpose(0,2,1,3).reshape(R1.shape[0]*m1,R2.shape[0]*m2)
            e, U = linalg.eigh(np.dot(Mat_temp,Mat_temp.T.conj()))
            e = e[::-1]
            U = U[:,::-1]
            s =np.zeros(e.size)
            max_e = np.nanmax(e)
            for i in range(0,e.size):
                if (e[i]/max_e > 1e-20):
                    s[i] = np.sqrt(e[i])
                    e[i] = 1.0/np.sqrt(e[i])
                else:
                    s[i] = 0.0
                    e[i] = 0.0

            VT = np.dot(U.T.conj(),Mat_temp)
            VT = np.dot(np.diag(e),VT)
        print "## svd fail"
    ## truncation
    lambda_c = s[0:D_cut]
    ## care for svd error
    for i in xrange(1,lambda_c.shape[0]):
        if np.abs(lambda_c[i]/lambda_c[0]) < 10**( - np.finfo(lambda_c[0].dtype).precision):
            lambda_c[i] = 0.0
        
    Uc = U[:,0:D_cut]
    VTc = VT[0:D_cut,:]

    norm = np.sqrt(np.sum(lambda_c**2))
    for x in np.nditer(lambda_c,flags = ['external_loop', 'buffered'],op_flags=['readwrite']):
        x[...] = np.sqrt(x/norm)


    #VTc_temp = VTc.reshape(dc,VTc.shape[1]/m2,m2)
    #for i in range(dc):
    #    for j in range(VTc.shape[1]/m2):
    #        for  k in range(m2):
    #            print "VTc[i,j,k]=",i,j,k,VTc_temp[i,j,k]
    
    Uc = np.dot(Uc,np.diag(lambda_c))
    VTc = np.dot(np.diag(lambda_c),VTc)
            

    ## Create new tensors
    ## remove lambda effets
    if (Tn1_lambda.shape[0]*Tn1_lambda.shape[1]*Tn1_lambda.shape[2] >= Tn1_lambda.shape[3]*Tn1_lambda.shape[4]):
        if connect1==0:
            Q1 = np.tensordot(np.tensordot(np.tensordot(Q1.reshape(Tn1.shape[1],Tn1.shape[2],Tn1.shape[3],m1*dc),np.diag(lambda1_inv[1]),(0,0)),np.diag(lambda1_inv[2]),(0,0)),np.diag(lambda1_inv[3]),(0,0))
        elif connect1==1:
            Q1 = np.tensordot(np.tensordot(np.tensordot(Q1.reshape(Tn1.shape[0],Tn1.shape[2],Tn1.shape[3],m1*dc),np.diag(lambda1_inv[0]),(0,0)),np.diag(lambda1_inv[2]),(0,0)),np.diag(lambda1_inv[3]),(0,0))
        elif connect1==2:
            Q1 = np.tensordot(np.tensordot(np.tensordot(Q1.reshape(Tn1.shape[0],Tn1.shape[1],Tn1.shape[3],m1*dc),np.diag(lambda1_inv[0]),(0,0)),np.diag(lambda1_inv[1]),(0,0)),np.diag(lambda1_inv[3]),(0,0))
        else :
            Q1 = np.tensordot(np.tensordot(np.tensordot(Q1.reshape(Tn1.shape[0],Tn1.shape[1],Tn1.shape[2],m1*dc),np.diag(lambda1_inv[0]),(0,0)),np.diag(lambda1_inv[1]),(0,0)),np.diag(lambda1_inv[2]),(0,0))
    else:
        if connect1==0:
            Q1 = np.tensordot(np.tensordot(np.tensordot(Q1.reshape(Tn1.shape[1],Tn1.shape[2],Tn1.shape[3],Tn1.shape[1]*Tn1.shape[2]*Tn1.shape[3]),np.diag(lambda1_inv[1]),(0,0)),np.diag(lambda1_inv[2]),(0,0)),np.diag(lambda1_inv[3]),(0,0))
        elif connect1==1:
            Q1 = np.tensordot(np.tensordot(np.tensordot(Q1.reshape(Tn1.shape[0],Tn1.shape[2],Tn1.shape[3],Tn1.shape[0]*Tn1.shape[2]*Tn1.shape[3]),np.diag(lambda1_inv[0]),(0,0)),np.diag(lambda1_inv[2]),(0,0)),np.diag(lambda1_inv[3]),(0,0))
        elif connect1==2:
            Q1 = np.tensordot(np.tensordot(np.tensordot(Q1.reshape(Tn1.shape[0],Tn1.shape[1],Tn1.shape[3],Tn1.shape[0]*Tn1.shape[1]*Tn1.shape[3]),np.diag(lambda1_inv[0]),(0,0)),np.diag(lambda1_inv[1]),(0,0)),np.diag(lambda1_inv[3]),(0,0))
        else :
            Q1 = np.tensordot(np.tensordot(np.tensordot(Q1.reshape(Tn1.shape[0],Tn1.shape[1],Tn1.shape[2],Tn1.shape[0]*Tn1.shape[1]*Tn1.shape[2]),np.diag(lambda1_inv[0]),(0,0)),np.diag(lambda1_inv[1]),(0,0)),np.diag(lambda1_inv[2]),(0,0))
        
        
    if (Tn2_lambda.shape[0]*Tn2_lambda.shape[1]*Tn2_lambda.shape[2] >= Tn2_lambda.shape[3]*Tn2_lambda.shape[4]):
        if connect2==0:
            Q2 = np.tensordot(np.tensordot(np.tensordot(Q2.reshape(Tn2.shape[1],Tn2.shape[2],Tn2.shape[3],m2*dc),np.diag(lambda2_inv[1]),(0,0)),np.diag(lambda2_inv[2]),(0,0)),np.diag(lambda2_inv[3]),(0,0))
        elif connect2==1:
            Q2 = np.tensordot(np.tensordot(np.tensordot(Q2.reshape(Tn2.shape[0],Tn2.shape[2],Tn2.shape[3],m2*dc),np.diag(lambda2_inv[0]),(0,0)),np.diag(lambda2_inv[2]),(0,0)),np.diag(lambda2_inv[3]),(0,0))
        elif connect2==2:
            Q2 = np.tensordot(np.tensordot(np.tensordot(Q2.reshape(Tn2.shape[0],Tn2.shape[1],Tn2.shape[3],m2*dc),np.diag(lambda2_inv[0]),(0,0)),np.diag(lambda2_inv[1]),(0,0)),np.diag(lambda2_inv[3]),(0,0))
        else :
            Q2 = np.tensordot(np.tensordot(np.tensordot(Q2.reshape(Tn2.shape[0],Tn2.shape[1],Tn2.shape[2],m2*dc),np.diag(lambda2_inv[0]),(0,0)),np.diag(lambda2_inv[1]),(0,0)),np.diag(lambda2_inv[2]),(0,0))
    else:
        if connect2==0:
            Q2 = np.tensordot(np.tensordot(np.tensordot(Q2.reshape(Tn2.shape[1],Tn2.shape[2],Tn2.shape[3],Tn2.shape[1]*Tn2.shape[2]*Tn2.shape[3]),np.diag(lambda2_inv[1]),(0,0)),np.diag(lambda2_inv[2]),(0,0)),np.diag(lambda2_inv[3]),(0,0))
        elif connect2==1:
            Q2 = np.tensordot(np.tensordot(np.tensordot(Q2.reshape(Tn2.shape[0],Tn2.shape[2],Tn2.shape[3],Tn2.shape[0]*Tn2.shape[2]*Tn2.shape[3]),np.diag(lambda2_inv[0]),(0,0)),np.diag(lambda2_inv[2]),(0,0)),np.diag(lambda2_inv[3]),(0,0))
        elif connect2==2:
            Q2 = np.tensordot(np.tensordot(np.tensordot(Q2.reshape(Tn2.shape[0],Tn2.shape[1],Tn2.shape[3],Tn2.shape[0]*Tn2.shape[1]*Tn2.shape[3]),np.diag(lambda2_inv[0]),(0,0)),np.diag(lambda2_inv[1]),(0,0)),np.diag(lambda2_inv[3]),(0,0))
        else :
            Q2 = np.tensordot(np.tensordot(np.tensordot(Q2.reshape(Tn2.shape[0],Tn2.shape[1],Tn2.shape[2],Tn2.shape[0]*Tn2.shape[1]*Tn2.shape[2]),np.diag(lambda2_inv[0]),(0,0)),np.diag(lambda2_inv[1]),(0,0)),np.diag(lambda2_inv[2]),(0,0))
        

    if connect1==0:
        Tn1_new = np.tensordot(Q1,Uc.reshape(R1.shape[0],m1,D_cut),(0,0)).transpose(4,0,1,2,3)
    elif connect1 == 1:
        Tn1_new = np.tensordot(Q1,Uc.reshape(R1.shape[0],m1,D_cut),(0,0)).transpose(0,4,1,2,3)
    elif connect1 == 2:
        Tn1_new = np.tensordot(Q1,Uc.reshape(R1.shape[0],m1,D_cut),(0,0)).transpose(0,1,4,2,3)
    else:
        Tn1_new = np.tensordot(Q1,Uc.reshape(R1.shape[0],m1,D_cut),(0,0)).transpose(0,1,2,4,3)
        
    if connect2 == 0:
        Tn2_new = np.tensordot(Q2,VTc.reshape(D_cut,R2.shape[0],m2),(0,1)).transpose(3,0,1,2,4)
    elif connect2 == 1:
        Tn2_new = np.tensordot(Q2,VTc.reshape(D_cut,R2.shape[0],m2),(0,1)).transpose(0,3,1,2,4)
    elif connect2 == 2:
        Tn2_new = np.tensordot(Q2,VTc.reshape(D_cut,R2.shape[0],m2),(0,1)).transpose(0,1,3,2,4)
    elif connect2 == 3:
        Tn2_new = np.tensordot(Q2,VTc.reshape(D_cut,R2.shape[0],m2),(0,1)).transpose(0,1,2,3,4)    
    
    ## return 
    return Tn1_new,Tn2_new,lambda_c

## for full update
def Create_Environment_two_sites(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,Q1,Q2):
    ## C1 - eT1 - eT2 - C2
    ## |    |     |     |
    ##eT6 - Q1-  -Q2  - eT3 
    ## |    |     |     |
    ## C4 - eT5 - eT4 - C3
    ##
    ## Q1,Q2 are double layered
    
##############################
# (((C2*eT2)*((((C3*eT4)*eT3)*Q2c)*Q2))*((C1*eT1)*((((C4*eT6)*eT5)*Q1c)*Q1)))
# cpu_cost= 2.22e+11  memory= 1.00204e+09
# final_bond_order  (tc2, t2, tc1, t1)
##############################
    return np.tensordot(
            np.tensordot(
                np.tensordot(
                    C2, eT2, ([0], [1])
                ), np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                C3, eT4, ([1], [0])
                            ), eT3, ([0], [1])
                        ), Q2.conj(), ([2, 5], [3, 2])
                    ), Q2, ([1, 3], [3, 2])
                ), ([0, 2, 3], [1, 5, 3])
            ), np.tensordot(
                np.tensordot(
                    C1, eT1, ([1], [0])
                ), np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            np.tensordot(
                                C4, eT6, ([1], [0])
                            ), eT5, ([0], [1])
                        ), Q1.conj(), ([2, 5], [0, 2])
                    ), Q1, ([1, 3], [0, 2])
                ), ([0, 2, 3], [0, 4, 2])
            ), ([0, 1], [0, 1])
        ).transpose(3,1,2,0)

def Full_update_bond_horizontal(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,Tn1,Tn2,op12):
    ## Connecting [2] bond of Tn1 and [0] bond of Tn2
    ## QR decomposition

    Tn1_shape = Tn1.shape
    Tn2_shape = Tn2.shape

    D1_out = Tn1_shape[0]*Tn1_shape[1]*Tn1_shape[3]
    D_connect = Tn1_shape[2]
    D2_out = Tn2_shape[1]*Tn2_shape[2]*Tn2_shape[3]
    m1 = Tn1_shape[4]
    m2 = Tn2_shape[4]
    
    if D_connect != Tn2_shape[0]:
        print "## error:bad connection at full update",Tn1.shape[2],Tn2.shape[0]
                
    if (D1_out >= D_connect * m1):
        Q1,R1 = linalg.qr(Tn1.transpose(0,1,3,2,4).reshape(D1_out,D_connect * m1),mode="economic")
    else:
        Q1 = np.identity(D1_out)
        R1 = Tn1.transpose(0,1,3,2,4).reshape(D1_out,D_connect * m1).copy()
    if (D2_out >= D_connect * m2):
        Q2,R2 = linalg.qr(Tn2.transpose(1,2,3,0,4).reshape(D2_out,D_connect * m2),mode="economic")
    else:
        Q2 = np.identity(D2_out)
        R2 = Tn2.transpose(1,2,3,0,4).reshape(D2_out,D_connect * m2).copy()

    envR1 = R1.shape[0]
    envR2 = R2.shape[0]
    
    ## apply time evolution
    ## connetc R1, R2, op
##############################
# ((R1*R2)*op12)
# cpu_cost= 22400  memory= 3216
# final_bond_order  (c1, c2, m1o, m2o)
##############################
    Theta = np.tensordot(
            np.tensordot(
                R1.reshape(envR1, D_connect,m1), R2.reshape(envR2, D_connect, m2), ([1], [1]) 
            ), op12, ([1, 3], [0, 1])
        )

    ## Environment
    # bond order (t1, t2, tc1, tc2)
    
    Environment = Create_Environment_two_sites(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,Q1.reshape(Tn1_shape[0],Tn1_shape[1],Tn1_shape[3],envR1),Q2.reshape(Tn2_shape[1],Tn2_shape[2],Tn2_shape[3],envR2).transpose(3,0,1,2))

    
    ## Hermite
    Environment = 0.5 * (Environment + Environment.transpose(2,3,0,1).conj())

    ## diagonalization
    w, U = linalg.eigh(Environment.reshape(envR1*envR2,envR1*envR2))
    ## w should be positive
    w = abs(w)
    #w_max=np.abs(w[-1])
    w_max=np.nanmax(w)
    for x in np.nditer(w,flags = ['buffered'],op_flags=['readwrite']):
        if abs(x/w_max) > Inverse_Env_cut:
            x[...] = np.sqrt(x)
        else:
            x[...] = 0.0

    
    
    Z = np.dot(U,np.diag(w)).reshape(envR1,envR2,envR1*envR2)
    
    if Full_Gauge_Fix:
        ## gauge fix
        ## note : mode "r" did not work because it did not return simple ndarray
        Q_temp,LR1 = linalg.qr(Z.transpose(2,1,0).reshape(envR1*envR2**2,envR1),mode="economic")
        Q_temp,LR2 = linalg.qr(Z.transpose(2,0,1).reshape(envR1**2*envR2,envR2),mode="economic")

        ## for theta
        Theta = np.tensordot(LR1,np.tensordot(LR2,Theta,(1,1)),(1,1))


        ## for environment
        ## using solve_triangular is unstable
        ## Z = linalg.solve_triangular(LR1,Z.reshape(envR1,envR1*envR2**2),trans="T").reshape(envR1,envR2,envR1*envR2)
        ## Z = linalg.solve_triangular(LR2,Z.transpose(1,0,2).reshape(envR2,envR1**2*envR2),trans="T").reshape(envR2,envR1,envR1*envR2) 

        ## Environment = np.tensordot(Z,Z.conj(),(2,2)).transpose(1,0,3,2)

        u,s,vt = linalg.svd(LR1,full_matrices=False)
        s_max = s[0]
        for x in np.nditer(s,flags = ['buffered'],op_flags=['readwrite']):
            if x/s_max > Inverse_Env_cut:
                x[...] = 1.0/x
            else:
                x[...] = 0.0
        LR1_inv = np.tensordot(np.tensordot(u.conj(),np.diag(s),(1,1)),vt.conj(),(1,0))

        u,s,vt = linalg.svd(LR2,full_matrices=False)
        s_max = s[0]
        for x in np.nditer(s,flags = ['buffered'],op_flags=['readwrite']):
            if x/s_max > Inverse_Env_cut:
                x[...] = 1.0/x
            else:
                x[...] = 0.0
        LR2_inv = np.tensordot(np.tensordot(u.conj(),np.diag(s),(1,1)),vt.conj(),(1,0))

        Z = np.tensordot(np.tensordot(Z,LR1_inv,(0,1)),LR2_inv,(0,1));
        
    
        Environment = np.tensordot(Z,Z.conj(),(0,0)).transpose(1,0,3,2)
    
    else:
        Environment = np.tensordot(Z,Z.conj(),(2,2))


    ##test##
    #print "environment",Environment[:,:,0,0]
    
    convergence = False

    ## create initial guess
    ## SVD of Theta
    U, s, VT = linalg.svd(Theta.transpose(0,2,1,3).reshape(envR1*m1,envR2*m2),full_matrices=False)
    
    ## truncation
    lambda_c = s[0:D_connect]
    Uc = U[:,0:D_connect]
    VTc = VT[0:D_connect,:]

    norm = np.sqrt(np.sum(lambda_c**2))
    for x in np.nditer(lambda_c,flags = ['external_loop', 'buffered'],op_flags=['readwrite']):
        x[...] = np.sqrt(x/norm)

    ##R1_old = np.dot(Uc,np.diag(lambda_c)).reshape(envR1,m1,D_connect).transpose(0,2,1)
    ##R2_old = np.dot(np.diag(lambda_c),VTc).transpose(1,0).reshape(envR2,m2,D_connect).transpose(0,2,1)

    R1 = np.tensordot(Uc,np.diag(lambda_c),(1,0)).reshape(envR1,m1,D_connect).transpose(0,2,1)
    R2 = np.tensordot(VTc,np.diag(lambda_c),(0,1)).reshape(envR2,m2,D_connect).transpose(0,2,1)
        

    count = 0

    C_phi = np.tensordot(np.tensordot(Environment,Theta,([0,1],[0,1])),Theta.conj(),([0,1,2,3],[0,1,2,3])).real
    Old_delta = -2.0 * \
                np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            Environment,Theta,([0,1],[0,1])
                        )
                        ,R2.conj(),([1,3],[0,2])
                    ),
                    R1.conj(),([0,1,2],[0,2,1])
                ).real + \
                np.tensordot(
                    R1, np.tensordot(
                        R2, np.tensordot(
                            Environment, np.tensordot(
                                R1.conj(), R2.conj(), (1, 1)
                            ), ([2, 3], [0, 2])
                        ), ([0, 2], [1, 3])
                    ), ([0, 1, 2], [1, 0, 2])
                ).real 

    #print "C_phi, Old_delta",C_phi,Old_delta
    while (not convergence and count < Full_max_iteration):

        ## for R1
        ## create W
        ## ((envR1 * D_connect,m1)*)
        W_vec = np.tensordot(np.tensordot(Environment,Theta,([0,1],[0,1])),R2.conj(),([1,3],[0,2])).transpose(0,2,1).reshape(envR1*D_connect,m1)

        ## create N
        ## ((envR1, D_connect)*, (envR1,D_connect))
        N_mat = np.tensordot(np.tensordot(Environment,R2,(1,0)),R2.conj(),([2,4],[0,2])).transpose(1,3,0,2).reshape(envR1*D_connect,envR1*D_connect)
        ## test
        s = linalg.svd(N_mat,compute_uv=False)
        ##

        ## Moore-Penrose Psude Inverse (for Hermitian matrix)
        N_mat_inv = linalg.pinvh(N_mat, cond = Full_Inverse_precision)
        
        R1 = np.dot(N_mat_inv,W_vec).reshape(envR1,D_connect,m1)
    
        ## for R2
        ## create W
        ## ((envR2 * D_connect,m2)*)
        W_vec = np.tensordot(np.tensordot(Environment,Theta,([0,1],[0,1])),R1.conj(),([0,2],[0,2])).transpose(0,2,1).reshape(envR2*D_connect,m2)

        ## create N
        ## ((envR2, D_connect)*, (envR2,D_connect))
        N_mat = np.tensordot(np.tensordot(Environment,R1,(0,0)),R1.conj(),([1,4],[0,2])).transpose(1,3,0,2).reshape(envR2*D_connect,envR2*D_connect)

        ## Moore-Penrose Psude Inverse (for Hermitian matrix)
        ## test
        s = linalg.svd(N_mat,compute_uv=False)
        ##
        N_mat_inv = linalg.pinvh(N_mat, cond = Full_Inverse_precision)
        
        R2 = np.dot(N_mat_inv,W_vec).reshape(envR2,D_connect,m2)

        delta = - 2.0 * \
                np.tensordot(
                    np.tensordot(
                        np.tensordot(
                            Environment,Theta,([0,1],[0,1])
                        )
                        ,R2.conj(),([1,3],[0,2])
                    ),
                    R1.conj(),([0,1,2],[0,2,1])
                ).real + \
                np.tensordot(
                    R1, np.tensordot(
                        R2, np.tensordot(
                            Environment, np.tensordot(
                                R1.conj(), R2.conj(), (1, 1)
                            ), ([2, 3], [0, 2])
                        ), ([0, 2], [1, 3])
                    ), ([0, 1, 2], [1, 0, 2])
                ).real

        if np.fabs(Old_delta - delta)/C_phi < Full_Convergence_Epsilon :
            convergence = True

        ## R1_old = R1_new
        ## R2_old = R2_new
        Old_delta = delta
        count += 1
    ## Post processing
    if not convergence:
        print "## warning: Full update iteration was not conveged! count=",count        

    if Debug_flag:
        print "## Full Update: count, delta,original_norm =",count,delta + C_phi,C_phi
    if Full_Gauge_Fix:
        ## remove gauge
        ## using solve_triangular is unstable
        ##R1 = linalg.solve_triangular(LR1,R1.reshape(envR1,D_connect*m1)).reshape(envR1,D_connect,m1)
        ##R2 = linalg.solve_triangular(LR2,R2.reshape(envR2,D_connect*m2)).reshape(envR2,D_connect,m2)

        R1 = np.tensordot(LR1_inv,R1,(0,0))        
        R2 = np.tensordot(LR2_inv,R2,(0,0))        
        
    ## balancing and normalization
    q1, r1 = linalg.qr(R1.transpose(0,2,1).reshape(envR1*m1,D_connect),mode="economic")
    q2, r2 = linalg.qr(R2.transpose(0,2,1).reshape(envR2*m2,D_connect),mode="economic")

    U, s, VT = linalg.svd(np.tensordot(r1,r2,(1,1)))

    norm =np.sqrt(np.sum(s**2))
    for x in np.nditer(s,flags = ['external_loop', 'buffered'],op_flags=['readwrite']):
        x[...] = np.sqrt(x/norm)

    R1 = np.dot(q1, np.dot(U,np.diag(s))).reshape(envR1,m1,D_connect)
    R2 = np.dot(q2, np.dot(np.diag(s),VT).T).reshape(envR2,m2,D_connect)


    Tn1_new = np.tensordot(Q1,R1,(1,0)).reshape(Tn1_shape[0],Tn1_shape[1],Tn1_shape[3],m1,Tn1_shape[2]).transpose(0,1,4,2,3)
    Tn2_new = np.tensordot(Q2,R2,(1,0)).reshape(Tn2_shape[1],Tn2_shape[2],Tn2_shape[3],m2,Tn2_shape[0]).transpose(4,0,1,2,3)
    
    return Tn1_new,Tn2_new

def Full_update_bond(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,Tn1,Tn2,op12,connect1):
    ## 回転して、horizontal 型にする 
    if connect1 == 0:
        #Tn1_rot = Tn1
        #Tn2_rot = Tn2
        Tn1_rot = Tn1.transpose(2,3,0,1,4)
        Tn2_rot = Tn2.transpose(2,3,0,1,4)
    elif connect1 == 1:
        Tn1_rot = Tn1.transpose(3,0,1,2,4)
        Tn2_rot = Tn2.transpose(3,0,1,2,4)
    elif connect1 == 2:
        #Tn1_rot = Tn1.transpose(2,3,0,1,4)
        #Tn2_rot = Tn2.transpose(2,3,0,1,4)
        Tn1_rot = Tn1
        Tn2_rot = Tn2
    else:
        Tn1_rot = Tn1.transpose(1,2,3,0,4)
        Tn2_rot = Tn2.transpose(1,2,3,0,4)

    Tn1_new_rot, Tn2_new_rot = Full_update_bond_horizontal(C1,C2,C3,C4,eT1,eT2,eT3,eT4,eT5,eT6,Tn1_rot,Tn2_rot,op12)

    if connect1 == 0:
        #Tn1_new = Tn1_new_rot
        #Tn2_new = Tn2_new_rot
        Tn1_new = Tn1_new_rot.transpose(2,3,0,1,4)
        Tn2_new = Tn2_new_rot.transpose(2,3,0,1,4)
    elif connect1 == 1:
        Tn1_new = Tn1_new_rot.transpose(1,2,3,0,4)
        Tn2_new = Tn2_new_rot.transpose(1,2,3,0,4)
    elif connect1 == 2:
        #Tn1_new = Tn1_new_rot.transpose(2,3,0,1,4)
        #Tn2_new = Tn2_new_rot.transpose(2,3,0,1,4)
        Tn1_new = Tn1_new_rot
        Tn2_new = Tn2_new_rot
    else:
        Tn1_new = Tn1_new_rot.transpose(3,0,1,2,4)
        Tn2_new = Tn2_new_rot.transpose(3,0,1,2,4)

    return Tn1_new, Tn2_new

