# coding:utf-8
## Constant Parameters for PEPS
import numpy as np
import fractions 

## Lattice
# At least one of the LX_ori and LY_ori must be even.
LX_ori = 2
LY_ori = 2
N_UNIT = LX_ori * LY_ori

## true periodicity
## definition of slide when we apply the slided bounday
## User have to set correct (acceptable) valuses
LX_diff = LX_ori
LY_diff = LY_ori


## Tensors
D = 4
CHI = D**2
TENSOR_DTYPE=np.dtype(complex)

## Debug
Debug_flag = False

## Simple update
Inverse_lambda_cut = 1e-12

## Full update
Full_Inverse_precision = 1e-12
Full_Convergence_Epsilon = 1e-12
Full_max_iteration = 1000
Full_Gauge_Fix = True
Full_Use_FFU = True

## Environment
Inverse_projector_cut = 1e-12
Inverse_Env_cut = Inverse_projector_cut
CTM_Convergence_Epsilon = 1e-10
Max_CTM_Iteration = 100
CTM_Projector_corner = True
Use_Partial_SVD = False
Use_Interporative_SVD = False
Use_Orthogonalization = True

## Following condition is tentative. It works only for limited situations.
if (LX_ori%2 ==1):
    ## skew boundary along x direction    
    ## fractions.gcd returns Greatest Common Divisor
    LX = (LY_ori / fractions.gcd(LY_ori, LX_diff)) * LX_ori
    LY = LY_ori
elif (LY_ori%2 ==1):
    ## skew boundary along x direction    
    ## fractions.gcd returns Greatest Common Divisor
    LX = LX_ori
    LY = (LX_ori / fractions.gcd(LX_ori, LY_diff)) * LY_ori
else:
    LX = LX_ori
    LY = LY_ori

def output_parameter(file):
    ## Lattice
    file.write("LX_ori = "+repr(LX_ori) + "\n")
    file.write("LY_ori = "+repr(LY_ori) + "\n")
    file.write("N_UNIT = "+repr(N_UNIT) + "\n")
    
    file.write("LX_diff = "+repr(LX_diff) + "\n")
    file.write("LY_diff = "+repr(LY_diff) + "\n")

    file.write("LX = "+repr(LX) + "\n")
    file.write("LY = "+repr(LY) + "\n")

    ## Tensors
    file.write("D = "+repr(D) + "\n")
    file.write("CHI = "+repr(CHI) + "\n")
    file.write("TENSOR_DTYPE = "+repr(TENSOR_DTYPE) + "\n")

    ## Debug
    file.write("Debug_flag = "+repr(Debug_flag) + "\n")

    ## Simple update
    file.write("Inverse_lambda_cut = "+repr(Inverse_lambda_cut) + "\n")

    ## Full update
    file.write("Full_Inverse_precision = "+repr(Full_Inverse_precision) + "\n")
    file.write("Full_Convergence_Epsilon = "+repr(Full_Convergence_Epsilon) + "\n")
    file.write("Full_max_iteration = "+repr(Full_max_iteration) + "\n")
    file.write("Full_Gauge_Fix = "+repr(Full_Gauge_Fix) + "\n")
    file.write("Full_Use_FFU = "+repr(Full_Use_FFU) + "\n")
    
    ## Environment
    file.write("Inverse_projector_cut = "+repr(Inverse_projector_cut) + "\n")
    file.write("Inverse_Env_cut = "+repr(Inverse_Env_cut) + "\n")
    file.write("CTM_Convergence_Epsilon = "+repr(CTM_Convergence_Epsilon) + "\n")
    file.write("Max_CTM_Iteration = "+repr(Max_CTM_Iteration) + "\n")
    file.write("CTM_Projector_corner = "+repr(CTM_Projector_corner) + "\n")
    file.write("Use_Partial_SVD = "+repr(Use_Partial_SVD) + "\n")
    file.write("Use_Interporative_SVD = "+repr(Use_Interporative_SVD) + "\n")
    file.write("Use_Orthogonalization = "+repr(Use_Orthogonalization) + "\n")

