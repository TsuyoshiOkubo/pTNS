# coding:utf-8
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import time
import sys
import argparse
import os
import os.path


## import basic routines
from PEPS_Basics import *
from PEPS_Parameters import *
from Square_lattice_CTM import *

def parse_args():
    parser = argparse.ArgumentParser(description='iPEPS simulation for Kagome lattice model')
    parser.add_argument('-Dz', metavar='Dz',dest='Dz', type=float, default=0.0,
                        help='set DM vector along z ')
    parser.add_argument('-Dp', metavar='Dp',dest='Dp', type=float, default=0.0,
                        help='set DM vector perpendicular z ')
    parser.add_argument('-hz', metavar='hz',dest='hz', type=float, default=0.0,
                        help='set hz')
    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=0.0,
                        help='set hx')
    parser.add_argument('-s', metavar='seed',dest='seed', type=int, default=11,
                        help='set random seed')

    parser.add_argument('-Jab', metavar='Jab',dest='Jab', type=float, default=1.0,
                        help='set Heisenberg coupling on ab bond')
    parser.add_argument('-Jbc', metavar='Jbc',dest='Jbc', type=float, default=1.0,
                        help='set Heisenberg coupling on bc bond')
    parser.add_argument('-Jca', metavar='Jca',dest='Jca', type=float, default=1.0,
                        help='set Heisenberg coupling on ca bond')

    parser.add_argument('-J2', metavar='J2',dest='J2', type=float, default=0.0,
                        help='set Heisenberg coupling on the 2nd neighbors')

    parser.add_argument('-ts', metavar='tau',dest='tau', type=float, default=0.01,
                        help='set tau for simple update')

    parser.add_argument('-tf', metavar='tau_full',dest='tau_full', type=float, default=0.01,
                        help='set tau for full update')

    parser.add_argument('-ss', metavar='tau_step',dest='tau_step', type=int, default=1000,
                        help='set step for simple update')

    parser.add_argument('-sf', metavar='tau_full_step',dest='tau_full_step', type=int, default=0,
                        help='set step for full update')
    
    parser.add_argument('-i', metavar='initial_type',dest='initial_type', type=int, default=0,
                        help='set initial state type (0:random, 1:ferro z-direction, 2:120 (q=0),3:120 (\sqrt{3} \times \sqrt{3}), 4:uud(q=0) z-direction, 5:uud(\sqrt{3} \times \sqrt{3}) z-direction, -1 or negative value:from file (default input_Tn.npz)')

    parser.add_argument('-ir', metavar='Random_amp',dest='Random_amp', type=float, default=0.01,
                        help='set amplitude of random noise for the inisital state')
    
    parser.add_argument('--second_ST', action='store_const', dest="second_ST",const=True, default=False,
                        help='use second order Suzuki Trotter decomposition for ITE')

    parser.add_argument('--2body', action='store_const', dest="Calc_all_2body",const=True, default=False,
                        help='Calculate all two body interactions which are not necessary for the Hamiltonian')
    
    parser.add_argument('-o', metavar='output_file',dest='output_file', default="output",
                        help='set output filename prefix for optimized tensor')
    
    parser.add_argument('-in', metavar='input_file',dest='input_file', default="input_Tn.npz",
                        help='set input filename for initial tensors')
    parser.add_argument('-tag', metavar='tag',dest='tag', default="",
                        help='set tag for output dirrectory')
    parser.add_argument('--append_output', action='store_const', const=True, default=False,
                        help='append output date to existing files instead creating new file')
    return parser.parse_args()


def Set_Hamiltonian(Jab,Jbc,Jca,hx,hz,Dp,Dz):    
    Sz = np.zeros((2,2))
    Sz[0,0] = 0.5
    Sz[1,1] = -0.5
    Sp = np.zeros((2,2))
    Sp[0,1] = 1.0
    Sm = np.zeros((2,2))
    Sm[1,0] = 1.0
    I = np.identity(2)
    def Ham_ops(ops):
        ## (0,1,2,3,4,5) = (i1,i2,i3,j1,j2,j3)
        return np.kron(np.kron(np.kron(np.kron(np.kron(ops[0],ops[1]),ops[2]),ops[3]),ops[4]),ops[5])        
    def Create_one_site_ops(op,ip):
        ops=[]
        for i in range(6):
            if i == ip:
                ops.append(op)
            else:
                ops.append(I)
        return ops
    def Create_two_site_ops(op_i,op_j,ip,jp):
        ops=[]
        for i in range(6):
            if i == ip:
                ops.append(op_i)
            elif i == jp:
                ops.append(op_j)
            else:
                ops.append(I)
        return ops        

    def Heisenberg(ip,jp):
        ops_Sz = Create_two_site_ops(Sz,Sz,ip,jp)
        ops_Spm = Create_two_site_ops(Sp,Sm,ip,jp)
        ops_Smp = Create_two_site_ops(Sm,Sp,ip,jp)

        return Ham_ops(ops_Sz) + 0.5 * (Ham_ops(ops_Spm) + Ham_ops(ops_Smp))


    def hx_field(ip,jp):
        ops_Spi = Create_one_site_ops(Sp,ip)
        ops_Spj = Create_one_site_ops(Sp,jp)
        ops_Smi = Create_one_site_ops(Sm,ip)
        ops_Smj = Create_one_site_ops(Sm,jp)
    
        return -0.125 * (Ham_ops(ops_Spi) + Ham_ops(ops_Spj) + Ham_ops(ops_Smi) + Ham_ops(ops_Smj))

    def hz_field(ip,jp):
        ops_Szi = Create_one_site_ops(Sz,ip)
        ops_Szj = Create_one_site_ops(Sz,jp)
    
        return -0.25 * (Ham_ops(ops_Szi) + Ham_ops(ops_Szj))

    def DM_Dz(ip,jp):
        ops_Spm = Create_two_site_ops(Sp,Sm,ip,jp)
        ops_Smp = Create_two_site_ops(Sm,Sp,ip,jp)
    
        return 0.5j * (Ham_ops(ops_Spm) - Ham_ops(ops_Smp))

    def DM_Dp(ip,jp,theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        i_e_pi_theta = 1.0j * cos_theta - sin_theta
        i_e_mi_theta = 1.0j * cos_theta + sin_theta

        ops_Spz = Create_two_site_ops(Sp,Sz,ip,jp)
        ops_Smz = Create_two_site_ops(Sm,Sz,ip,jp)
        ops_Szp = Create_two_site_ops(Sz,Sp,ip,jp)
        ops_Szm = Create_two_site_ops(Sz,Sm,ip,jp)
    
        return 0.5 * (i_e_mi_theta * (-Ham_ops(ops_Spz) + Ham_ops(ops_Szp)) + i_e_pi_theta * (Ham_ops(ops_Smz) - Ham_ops(ops_Szm)))
    
    def Create_Hij(ip,jp,J,Dp,Dz,theta):
        if (Dz == 0.0 and Dp == 0.0):
            Hij = J * Heisenberg(ip,jp) + hx * hx_field(ip,jp) + hz * hz_field(ip,jp)
        elif Dp == 0.0:
            Hij = J * Heisenberg(ip,jp) + hx * hx_field(ip,jp) + hz * hz_field(ip,jp) + Dz * DM_Dz(ip,jp)
        elif Dz == 0.0:
            Hij =  J * Heisenberg(ip,jp) + hx * hx_field(ip,jp) + hz * hz_field(ip,jp) + Dp * DM_Dp(ip,jp,theta)
        else:
            Hij =  J * Heisenberg(ip,jp) + hx * hx_field(ip,jp) + hz * hz_field(ip,jp) + Dp * DM_Dp(ip,jp,theta) + Dz * DM_Dz(ip,jp)
        return Hij

    
    #H_AA_i = A1*A2 + A2*A3 = 
    Ham_AA_i = Create_Hij(0,1,Jbc,Dp,Dz,7.0 * np.pi/6.0) + Create_Hij(1,2,Jbc,Dp,-Dz,np.pi/6.0) 

    #H_AA_j = A1*A2 + A2*A3  
    Ham_AA_j = Create_Hij(3,4,Jbc,Dp,Dz,7.0 * np.pi/6.0) + Create_Hij(4,5,Jbc,Dp,-Dz,np.pi/6.0) 

    #H_BB_i = B1*B2 + B2*B3
    Ham_BB_i = Create_Hij(0,1,Jca,Dp,-Dz,5.0 * np.pi/6.0) + Create_Hij(1,2,Jca,Dp,Dz,11.0 * np.pi/6.0) 

    #H_BB_j = B1*B2 + B2*B3
    Ham_BB_j = Create_Hij(3,4,Jca,Dp,-Dz,5.0 * np.pi/6.0) + Create_Hij(4,5,Jca,Dp,Dz,11.0 * np.pi/6.0) 
        
    #H_ABx = A2B3 + A1B3 + 0.25 * (H_AA_i + H_BB_j)
    Ham_ABx = Create_Hij(1,5,Jab,Dp,Dz,11.0 * np.pi/6.0) + Create_Hij(0,5,Jab,Dp,-Dz, 3.0 * np.pi/2.0) + 0.25 * (Ham_AA_i + Ham_BB_j) 

    #H_ABy = A1B2 + A1B3 + 0.25 * (H_AA_i + H_BB_j)
    Ham_ABy = Create_Hij(0,4,Jbc,Dp,Dz,7.0 * np.pi/6.0) + Create_Hij(0,5,Jab,Dp,-Dz, 3.0 * np.pi/2.0) + 0.25 * (Ham_AA_i + Ham_BB_j) 


    #H_BAx = B1A2 + B1A3 + 0.25 * (H_BB_i + H_AA_j)
    Ham_BAx = Create_Hij(0,4,Jca,Dp,-Dz,5.0*np.pi/6.0) + Create_Hij(0,5,Jab,Dp,Dz,  np.pi/2.0) + 0.25 * (Ham_BB_i + Ham_AA_j) 
    
    #H_BAy = B2A3 + B1A3 + 0.25 * (H11 + H22)
    Ham_BAy = Create_Hij(1,5,Jbc,Dp,-Dz,np.pi/6.0) + Create_Hij(0,5,Jab,Dp,Dz, np.pi/2.0) + 0.25 * (Ham_BB_i + Ham_AA_j) 

    return Ham_ABx,Ham_ABy,Ham_BAx,Ham_BAy

def Initialize_Tensors(Tn,initial_type=0,seed=11,random_amp=0.01,Dz=0.0):
    ## Dz is used only for the case of q=0 120degree structure
    np.random.seed(seed)
    for num in range(N_UNIT):
        if isinstance(Tn[num][0,0,0,0,0],complex):
            Tn[num][:] = random_amp* ((np.random.rand(D,D,D,D,8)-0.5) + 1.0j * (np.random.rand(D,D,D,D,8)-0.5))
        else:
            Tn[num][:] = random_amp* (np.random.rand(D,D,D,D,8)-0.5)
    def make_120_structure(theta1,theta2,theta3):
        T_120 = np.zeros(8,dtype=complex)
        T_120[0] = 1.0
        T_120[1] = np.cos(theta3) + 1.0j * np.sin(theta3)
        T_120[2] = np.cos(theta2) + 1.0j * np.sin(theta2)
        T_120[3] = np.cos(theta2+theta3) + 1.0j * np.sin(theta2+theta3)
        T_120[4] = np.cos(theta1) + 1.0j * np.sin(theta1)
        T_120[5] = np.cos(theta1+theta3) + 1.0j * np.sin(theta1+theta3)
        T_120[6] = np.cos(theta1+theta2) + 1.0j * np.sin(theta1+theta2)
        T_120[7] = np.cos(theta1+theta2+theta3) + 1.0j * np.sin(theta1+theta2+theta3)
        return T_120
            
    if initial_type == 1:
        #ferro
        for num  in range(N_UNIT):
            Tn[num][0,0,0,0,0]  = 1.0
    elif (initial_type ==2):
        #q=0 120 degree sttructure on xy plane
        if Dz <= 0.0:
            theta_A1 = 2.0 * np.pi / 3.0
            theta_A2 = 4.0 * np.pi / 3.0
            theta_A3 = 2.0 * np.pi / 3.0

            theta_B1 = 0.0
            theta_B2 = 4.0 * np.pi / 3.0
            theta_B3 = 0.0
        else:
            theta_A1 = 4.0 * np.pi / 3.0
            theta_A2 = 2.0 * np.pi / 3.0
            theta_A3 = 4.0 * np.pi / 3.0

            theta_B1 = 0.0
            theta_B2 = 2.0 * np.pi / 3.0
            theta_B3 = 0.0


        A_120 = make_120_structure(theta_A1,theta_A2,theta_A3)
        B_120 = make_120_structure(theta_B1,theta_B2,theta_B3)
        
        for i  in range(N_UNIT/2):            
            ## A-Tensor
            num = A_sub_list[i]
            Tn[num][0,0,0,0] = A_120
            ## B-Tensor
            num = B_sub_list[i]
            Tn[num][0,0,0,0] = B_120
            
    elif (initial_type ==3):
        #\sqrt{3} \times \sqrt{3} 120 degree sttructure on xy plane
        # Note! We need enough # of unit cells

        
        theta_a = 0.0
        theta_b = 2.0 * np.pi / 3.0
        theta_c = 4.0 * np.pi / 3.0 

        for num in range(N_UNIT):
            ix = Tensor_position[num,0]
            iy = Tensor_position[num,1]
            theta_type= (iy +ix) % 6
            if theta_type == 0:
                #A-sub
                Tn[num][0,0,0,0] = make_120_structure(theta_c,theta_b,theta_a)
            elif theta_type ==1:
                #B-sub
                Tn[num][0,0,0,0] = make_120_structure(theta_c,theta_a,theta_b)
            elif theta_type ==2:
                #A-sub
                Tn[num][0,0,0,0] = make_120_structure(theta_a,theta_c,theta_b)
            elif theta_type ==3:
                #B-sub
                Tn[num][0,0,0,0] = make_120_structure(theta_a,theta_b,theta_c)
            elif theta_type ==4:
                #A-sub
                Tn[num][0,0,0,0] = make_120_structure(theta_b,theta_a,theta_c)
            elif theta_type ==5:
                #B-sub
                Tn[num][0,0,0,0] = make_120_structure(theta_b,theta_c,theta_a)
    elif initial_type ==4:
        #q=0 up,up,down
        for num  in range(N_UNIT):            
            Tn[num][0,0,0,0,2] = 1.0
    elif initial_type ==5:
        #\sqrt{3} \times \sqrt{3} up,up,down
        for num in range(N_UNIT):
            ix = Tensor_position[num,0]
            iy = Tensor_position[num,1]
            theta_type= (iy +ix) % 6
            if theta_type == 0:
                #A-sub
                Tn[num][0,0,0,0,4] = 1.0
            elif theta_type ==1:
                #B-sub
                Tn[num][0,0,0,0,4] = 1.0
            elif theta_type ==2:
                #A-sub
                Tn[num][0,0,0,0,2] = 1.0
            elif theta_type ==3:
                #B-sub
                Tn[num][0,0,0,0,1] = 1.0
            elif theta_type ==4:
                #A-sub
                Tn[num][0,0,0,0,1] = 1.0
            elif theta_type ==5:
                #B-sub
                Tn[num][0,0,0,0,2] = 1.0
    # else :
        ## random
def main():
    ## timers
    time_simple_update=0.0
    time_full_update=0.0
    time_env=0.0
    time_obs=0.0
    
    ## Parameters
    args=parse_args()
    Jab = args.Jab
    Jbc = args.Jbc
    Jca = args.Jca
    Dz = args.Dz
    Dp = args.Dp
    hx = args.hx
    hz = args.hz

    Random_amp = args.Random_amp
    seed = args.seed

    ##
    initial_type = args.initial_type
    second_ST = args.second_ST

    ##
    Calc_all_2body = args.Calc_all_2body
    tau = args.tau
    tau_step =args.tau_step

    tau_full = args.tau_full
    tau_full_step = args.tau_full_step

    ## output files    
    output_prefix=args.output_file
    input_file = args.input_file

    output_data_dir = "output_data"+args.tag
    append_output = args.append_output

    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)
    if append_output:
        file_Energy = open(output_data_dir+"/Energy.dat","a")
        file_Energy_sub = open(output_data_dir+"/Energy_Sub.dat","a")
        file_Correlation = open(output_data_dir+"/Correlation.dat","a")
        file_Mag = open(output_data_dir+"/Magnetization.dat","a")
        file_Mag_sub = open(output_data_dir+"/Magnetization_Sub.dat","a")
        file_Mag_all = open(output_data_dir+"/Magnetization_All.dat","a")
        file_params = open(output_data_dir+"/output_params.dat","a")
        file_Timer = open(output_data_dir+"/Timer.dat","a")
    else:
        file_Energy = open(output_data_dir+"/Energy.dat","w")
        file_Energy_sub = open(output_data_dir+"/Energy_Sub.dat","w")
        file_Correlation = open(output_data_dir+"/Correlation.dat","w")
        file_Mag = open(output_data_dir+"/Magnetization.dat","w")
        file_Mag_sub = open(output_data_dir+"/Magnetization_Sub.dat","w")
        file_Mag_all = open(output_data_dir+"/Magnetization_All.dat","w")
        file_params = open(output_data_dir+"/output_params.dat","w")
        file_Timer = open(output_data_dir+"/Timer.dat","w")


    ##
    print "## Logs: Jab =",Jab
    print "## Logs: Jbc =",Jbc
    print "## Logs: Jca =",Jca
    print "## Logs: Dz =",Dz
    print "## Logs: Dp =",Dp
    print "## Logs: hz =",hz
    print "## Logs: hx =",hx
    print "## Logs: seed =",seed
    print "## Logs: Random_amp =",Random_amp
    print "## Logs: tau =",tau
    print "## Logs: tau_step =",tau_step
    print "## Logs: tau_full =",tau_full
    print "## Logs: tau_full_step =",tau_full_step
    print "## Logs: second_ST =",second_ST
    print "## Logs: initial_type =",initial_type
    if initial_type < 0:
        print "## Logs: input_file =",input_file
    print "## Logs: output_file =",output_prefix
    print "## Logs: output_data_dir =",output_data_dir
    print "## Logs: append_output =",append_output

    file_params.write("Jab = "+repr(Jab) + "\n")
    file_params.write("Jbc = "+repr(Jbc) + "\n")
    file_params.write("Jca = "+repr(Jca) + "\n")
    file_params.write("Dz = "+repr(Dz) + "\n")
    file_params.write("Dp = "+repr(Dp) + "\n")
    file_params.write("hz = "+repr(hz) + "\n")
    file_params.write("hx = "+repr(hx) + "\n")
    file_params.write("seed = "+repr(seed) + "\n")
    file_params.write("Random_amp = "+repr(Random_amp) + "\n")
    file_params.write("tau = "+repr(tau) + "\n")
    file_params.write("tau_step = "+repr(tau_step) + "\n")
    file_params.write("tau_full = "+repr(tau_full) + "\n")
    file_params.write("tau_full_step = "+repr(tau_full_step) + "\n")
    file_params.write("second_ST = "+repr(second_ST) + "\n")
    file_params.write("initial_type = "+repr(initial_type) + "\n")
    if initial_type < 0:
        file_params.write("input_file = "+repr(input_file) + "\n")
    file_params.write("output_file = "+repr(output_prefix) + "\n")
    file_params.write("output_data_dir = "+repr(output_data_dir) + "\n")
    file_params.write("append_output = "+repr(append_output) + "\n")

    output_parameter(file_params)
    file_params.write("\n")

        
    ## Tensors
    
    Tn=[np.zeros((D,D,D,D,8),dtype=TENSOR_DTYPE)]
    eTt=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTr=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTb=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTl=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]

    C1=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C2=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C3=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C4=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)

    for i in range(1,N_UNIT):
        Tn.append(np.zeros((D,D,D,D,8),dtype=TENSOR_DTYPE))
        eTt.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
        eTr.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
        eTb.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
        eTl.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))


    ## Initialize tensor every step
    if initial_type >= 0:
        Initialize_Tensors(Tn,initial_type,seed,Random_amp,Dz)
    else:
        #input from file
        input_tensors = np.load(input_file)
        for i in range(N_UNIT):
            key='arr_'+repr(i)
            Tn_in = input_tensors[key]
            Tn[i][0:min(Tn[i].shape[0],Tn_in.shape[0]),0:min(Tn[i].shape[1],Tn_in.shape[1]),0:min(Tn[i].shape[2],Tn_in.shape[2]),0:min(Tn[i].shape[3],Tn_in.shape[3]),0:min(Tn[i].shape[4],Tn_in.shape[4])] = Tn_in[0:min(Tn[i].shape[0],Tn_in.shape[0]),0:min(Tn[i].shape[1],Tn_in.shape[1]),0:min(Tn[i].shape[2],Tn_in.shape[2]),0:min(Tn[i].shape[3],Tn_in.shape[3]),0:min(Tn[i].shape[4],Tn_in.shape[4])]
        
    lambda_tensor = np.ones((N_UNIT,4,D),dtype=float)

    Ham_ABx,Ham_ABy,Ham_BAx,Ham_BAy = Set_Hamiltonian(Jab,Jbc,Jca,hx,hz,Dp,Dz)
    s_ABx,U_ABx = linalg.eigh(Ham_ABx)

    opAB_x =  np.dot(np.dot(U_ABx,np.diag(np.exp(-tau * s_ABx))),U_ABx.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)
    opAB_x_2 =  np.dot(np.dot(U_ABx,np.diag(np.exp(-tau*0.5 * s_ABx))),U_ABx.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)

    s_ABy,U_ABy = linalg.eigh(Ham_ABy)

    opAB_y =  np.dot(np.dot(U_ABy,np.diag(np.exp(-tau * s_ABy))),U_ABy.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)
    opAB_y_2 =  np.dot(np.dot(U_ABy,np.diag(np.exp(-tau*0.5 * s_ABy))),U_ABy.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)

    s_BAx,U_BAx = linalg.eigh(Ham_BAx)

    opBA_x =  np.dot(np.dot(U_BAx,np.diag(np.exp(-tau * s_BAx))),U_BAx.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)
    opBA_x_2 =  np.dot(np.dot(U_BAx,np.diag(np.exp(-tau*0.5 * s_BAx))),U_BAx.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)

    s_BAy,U_BAy = linalg.eigh(Ham_BAy)

    opBA_y =  np.dot(np.dot(U_BAy,np.diag(np.exp(-tau * s_BAy))),U_BAy.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)
    opBA_y_2 =  np.dot(np.dot(U_BAy,np.diag(np.exp(-tau*0.5 * s_BAy))),U_BAy.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)

    start_simple=time.time()


    for int_tau in range(0,tau_step):
        if second_ST:
            ## simple update
            ## x-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opAB_x_2,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c

            ## x-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opBA_x_2,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c


            ## y-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opAB_y_2,1)
                lambda_tensor[num,1] = lambda_c
                lambda_tensor[num_j,3] = lambda_c

            ## y-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opBA_y,1)
                lambda_tensor[num,1] = lambda_c
                lambda_tensor[num_j,3] = lambda_c

            ## y-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opAB_y_2,1)
                lambda_tensor[num,1] = lambda_c
                lambda_tensor[num_j,3] = lambda_c

            ## x-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opBA_x_2,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c

            ## x-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opAB_x_2,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c

        else:
            ## simple update
            ## x-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opAB_x,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c

            ## x-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opBA_x,2)
                lambda_tensor[num,2] = lambda_c
                lambda_tensor[num_j,0] = lambda_c


            ## y-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opAB_y,1)
                lambda_tensor[num,1] = lambda_c
                lambda_tensor[num_j,3] = lambda_c

            ## y-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],opBA_y,1)
                lambda_tensor[num,1] = lambda_c
                lambda_tensor[num_j,3] = lambda_c


        ## done simple update
    time_simple_update += time.time() - start_simple


    ## Start full update
    if tau_full_step > 0:

        opAB_x =  np.dot(np.dot(U_ABx,np.diag(np.exp(-tau_full * s_ABx))),U_ABx.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)
        opAB_x_2 =  np.dot(np.dot(U_ABx,np.diag(np.exp(-tau_full*0.5 * s_ABx))),U_ABx.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)

        opAB_y =  np.dot(np.dot(U_ABy,np.diag(np.exp(-tau_full * s_ABy))),U_ABy.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)
        opAB_y_2 =  np.dot(np.dot(U_ABy,np.diag(np.exp(-tau_full*0.5 * s_ABy))),U_ABy.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)

        opBA_x =  np.dot(np.dot(U_BAx,np.diag(np.exp(-tau_full * s_BAx))),U_BAx.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)
        opBA_x_2 =  np.dot(np.dot(U_BAx,np.diag(np.exp(-tau_full*0.5 * s_BAx))),U_BAx.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)

        opBA_y =  np.dot(np.dot(U_BAy,np.diag(np.exp(-tau_full * s_BAy))),U_BAy.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)
        opBA_y_2 =  np.dot(np.dot(U_BAy,np.diag(np.exp(-tau_full*0.5 * s_BAy))),U_BAy.conj().T).reshape(8,8,8,8).transpose(2,3,0,1)


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

                Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],opAB_x_2,2)

                if Full_Use_FFU:
                    #ix = num%LX_ori
                    #ix_j = num_j%LX_ori
                    ix = Tensor_position[num,0]
                    ix_j = Tensor_position[num_j,0]
                    Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                    Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                

                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## x-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],opBA_x_2,2)

                if Full_Use_FFU:
                    #ix = num%LX_ori
                    #ix_j = num_j%LX_ori
                    ix = Tensor_position[num,0]
                    ix_j = Tensor_position[num_j,0]
                    Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                    Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## y-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],opAB_y_2,1)

                if Full_Use_FFU:
                    #iy = num/LX_ori
                    #iy_j = num_j/LX_ori
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)


            ## y-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]


                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],opBA_y,1)

                if Full_Use_FFU:
                    #iy = num/LX_ori
                    #iy_j = num_j/LX_ori
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
            ## y-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],opAB_y_2,1)

                if Full_Use_FFU:
                    #iy = num/LX_ori
                    #iy_j = num_j/LX_ori
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## x-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],opBA_x_2,2)

                if Full_Use_FFU:
                    #ix = num%LX_ori
                    #ix_j = num_j%LX_ori
                    ix = Tensor_position[num,0]
                    ix_j = Tensor_position[num_j,0]
                    Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                    Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## x-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],opAB_x_2,2)

                if Full_Use_FFU:
                    #ix = num%LX_ori
                    #ix_j = num_j%LX_ori
                    ix = Tensor_position[num,0]
                    ix_j = Tensor_position[num_j,0]
                    Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                    Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                

                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

        else:
            ## x-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],opAB_x,2)

                if Full_Use_FFU:
                    #ix = num%LX_ori
                    #ix_j = num_j%LX_ori
                    ix = Tensor_position[num,0]
                    ix_j = Tensor_position[num_j,0]
                    Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                    Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                

                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## x-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],opBA_x,2)

                if Full_Use_FFU:
                    #ix = num%LX_ori
                    #ix_j = num_j%LX_ori
                    ix = Tensor_position[num,0]
                    ix_j = Tensor_position[num_j,0]
                    Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                    Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## y-bond A sub-lattice
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],opAB_y,1)

                if Full_Use_FFU:
                    #iy = num/LX_ori
                    #iy_j = num_j/LX_ori
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)


            ## y-bond B sub-lattice
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]


                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],opBA_y,1)

                if Full_Use_FFU:
                    #iy = num/LX_ori
                    #iy_j = num_j/LX_ori
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)


        ## done full update
    time_full_update += time.time()-start_full

    ## output optimized tensor
    output_file = output_prefix+'_Tn.npz'
    np.savez(output_file, *Tn)    

    ## Calc physical quantities
    start_env = time.time()
    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
    time_env += time.time() - start_env

    def Create_one_op(op1,op2,op3):
        return np.kron(np.kron(op1,op2),op3)


    Sz = np.zeros((2,2))
    Sz[0,0] = 0.5
    Sz[1,1] = -0.5

    Sx = np.zeros((2,2))
    Sx[0,1] = 0.5
    Sx[1,0] = 0.5

    #!!note operator should be transposed due to the definition
    Sy = np.zeros((2,2),dtype=complex)
    Sy[0,1] = 0.5j
    Sy[1,0] = -0.5j


    I = np.identity(2)

    S_op=[Sx]
    S_op.append(Sy)
    S_op.append(Sz)

    S_1 = []
    S_2 = []
    S_3 = []
    for i in range(3):
        S_1.append(Create_one_op(S_op[i],I,I))
        S_2.append(Create_one_op(I,S_op[i],I))
        S_3.append(Create_one_op(I,I,S_op[i]))


    S_12 =[]
    S_23 =[]
    for i in range(9):
        i1 = i / 3
        i2 = i % 3
        S_12.append(Create_one_op(S_op[i1],S_op[i2],I))
        S_23.append(Create_one_op(I,S_op[i1],S_op[i2]))

    I_123 = np.identity(8)

    m_1 = np.zeros((N_UNIT,3))
    m_2 = np.zeros((N_UNIT,3))
    m_3 = np.zeros((N_UNIT,3))

    m_12 = np.zeros((N_UNIT,9))
    m_23 = np.zeros((N_UNIT,9))

    m2_12 = np.zeros((N_UNIT,9))
    m2_23 = np.zeros((N_UNIT,9))
    m2_31_1 = np.zeros((N_UNIT,9))
    m2_31_2 = np.zeros((N_UNIT,9))

    s_chirality = np.zeros((N_UNIT/2,6))
    ## list for necessary elements of two-body interactions
    if (not Calc_all_2body):
        if (Dz == 0.0 and Dp ==0.0):
            num_2body = 3
            list_2body = np.zeros(num_2body,dtype=int)
            list_2body[0] = 0
            list_2body[1] = 4
            list_2body[2] = 8
        elif (Dz ==0.0):
            num_2body = 7
            list_2body = np.zeros(num_2body,dtype=int)
            list_2body[0] = 0
            list_2body[1] = 2
            list_2body[2] = 4
            list_2body[3] = 5
            list_2body[4] = 6
            list_2body[5] = 7
            list_2body[6] = 8
        elif (Dp ==0.0):
            num_2body = 5
            list_2body = np.zeros(num_2body,dtype=int)
            list_2body[0] = 0
            list_2body[1] = 1
            list_2body[2] = 3
            list_2body[3] = 4
            list_2body[4] = 8
        else:
            num_2body = 9
            list_2body = np.arange(num_2body,dtype=int)
    else:    
        num_2body = 9
        list_2body = np.arange(num_2body)



    start_obs = time.time()
    for i in range(N_UNIT):
        ## site
        norm = Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],I_123)
        for n in range(3):
            m_1[i,n] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],S_1[n])/norm)
            m_2[i,n] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],S_2[n])/norm)
            m_3[i,n] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],S_3[n])/norm)

        for n2 in range(num_2body):
            n = list_2body[n2]
            m_12[i,n] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],S_12[n])/norm)
            m_23[i,n] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],S_23[n])/norm)


    for n_sub in range(0,N_UNIT/2):        

        ## A sub lattice
        num = A_sub_list[n_sub]

        ## x direction
        num_j = NN_Tensor[num,2]
        norm_x = Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],I_123,I_123)

        for n in range(num_2body):
            i = list_2body[n]
            i1 = i / 3
            i2 = i % 3

            m2_23[num,i] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],S_2[i1],S_3[i2])/norm_x)
            m2_31_1[num_j,i] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],S_1[i2],S_3[i1])/norm_x)


        ## y direction
        num_j = NN_Tensor[num,3]        
        norm_y = Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],I_123,I_123)

        for n in range(num_2body):
            i = list_2body[n]
            i1 = i / 3
            i2 = i % 3

            m2_23[num_j,i] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],S_3[i2],S_2[i1])/norm_y)
            m2_31_1[num,i] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],S_3[i1],S_1[i2])/norm_y)




        ## B sub lattice
        num = B_sub_list[n_sub]

        ## x direction
        num_j = NN_Tensor[num,2]
        norm_x = Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],I_123,I_123)

        for n in range(num_2body):
            i = list_2body[n]
            i1 = i / 3
            i2 = i % 3

            m2_12[num,i] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],S_1[i1],S_2[i2])/norm_x)
            m2_31_2[num_j,i] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],S_1[i2],S_3[i1])/norm_x)


        ## y direction
        num_j = NN_Tensor[num,3]        
        norm_y = Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],I_123,I_123)

        for n in range(num_2body):
            i = list_2body[n]
            i1 = i / 3
            i2 = i % 3

            m2_12[num_j,i] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],S_2[i2],S_1[i1])/norm_y)
            m2_31_2[num,i] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],S_3[i1],S_1[i2])/norm_y)





    ## scalar chirality
    list_3body = np.zeros((2,6),dtype=int)
    list_3body[0,0] = 1
    list_3body[0,1] = 2
    list_3body[0,2] = 3
    list_3body[0,3] = 5
    list_3body[0,4] = 6
    list_3body[0,5] = 7

    list_3body[1,0] = 2
    list_3body[1,1] = 1
    list_3body[1,2] = 2
    list_3body[1,3] = 0
    list_3body[1,4] = 1
    list_3body[1,5] = 0

    for n_sub in range(0,N_UNIT/2):        

        ## A sub lattice
        num = A_sub_list[n_sub]

        ## x direction
        num_j = NN_Tensor[num,2]
        norm_x = Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],I_123,I_123)

        for n in range(6):
            i = list_3body[0,n]
            j = list_3body[1,n]

            s_chirality[n_sub,n] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],S_12[i],S_3[j])/norm_x)

        print "## s_chirality ",num,hx,hz,Dz,Dp,s_chirality[n_sub]

    Scalar_Chirality = np.zeros(N_UNIT/2)
    for i in range(N_UNIT/2):
        Scalar_Chirality[i] = s_chirality[i,0] + s_chirality[i,3] + s_chirality[i,4] - (s_chirality[i,1] + s_chirality[i,2] + s_chirality[i,5])
        print "## Scalar Chirality ",i,hx,hz,Dz,Dp,Scalar_Chirality[i]

    Total_Scalar_Chirality = sum(Scalar_Chirality)/(N_UNIT/2)

    for i in range(N_UNIT):
        print "## Mag 1",i,hx,hz,Dz,Dp,m_1[i,0],m_1[i,1],m_1[i,2],np.sqrt(np.sum(m_1[i]**2))
        print "## Mag 2",i,hx,hz,Dz,Dp,m_2[i,0],m_2[i,1],m_2[i,2],np.sqrt(np.sum(m_2[i]**2))
        print "## Mag 3",i,hx,hz,Dz,Dp,m_3[i,0],m_3[i,1],m_3[i,2],np.sqrt(np.sum(m_3[i]**2))
        file_Mag_all.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(m_1[i,0])+" "+repr(m_1[i,1])+" "+repr(m_1[i,2])+" "+repr(np.sqrt(np.sum(m_1[i]**2)))+"\n")
        file_Mag_all.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(m_2[i,0])+" "+repr(m_2[i,1])+" "+repr(m_2[i,2])+" "+repr(np.sqrt(np.sum(m_2[i]**2)))+"\n")
        file_Mag_all.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(m_3[i,0])+" "+repr(m_3[i,1])+" "+repr(m_3[i,2])+" "+repr(np.sqrt(np.sum(m_3[i]**2)))+"\n")

    for i in range(N_UNIT):

        print "## Cor 12",i,hx,hz,Dz,Dp,m_12[i]
        print "## Cor2 12",i,hx,hz,Dz,Dp,m2_12[i]
        print "## Cor 23",i,hx,hz,Dz,Dp,m_23[i]
        print "## Cor2 23",i,hx,hz,Dz,Dp,m2_23[i]
        print "## Cor2 31_1",i,hx,hz,Dz,Dp,m2_31_1[i]
        print "## Cor2 31_2",i,hx,hz,Dz,Dp,m2_31_2[i]

        file_Correlation.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(m_12[i])+"\n") 
        file_Correlation.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(m2_12[i])+"\n") 
        file_Correlation.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(m_23[i])+"\n") 
        file_Correlation.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(m2_23[i])+"\n")
        file_Correlation.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(m2_31_1[i])+"\n")
        file_Correlation.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(m2_31_2[i])+"\n")
      
    ## Magnetization
    mx_a = 0.0
    mx_b = 0.0
    mx_c = 0.0

    my_a = 0.0
    my_b = 0.0
    my_c = 0.0

    mz_a = 0.0
    mz_b = 0.0
    mz_c = 0.0

    m_abs_a = 0.0
    m_abs_b = 0.0
    m_abs_c = 0.0
    
    ## Energy
    Energy_Heisenberg_ab = 0.0
    Energy_Heisenberg_bc = 0.0
    Energy_Heisenberg_ca = 0.0

    Energy_Dz_ab = 0.0
    Energy_Dz_bc = 0.0
    Energy_Dz_ca = 0.0

    Energy_Dp_ab = 0.0
    Energy_Dp_bc = 0.0
    Energy_Dp_ca = 0.0

    for n_sub in range(N_UNIT/2):
        ## A sub lattice
        i = A_sub_list[n_sub]

        mx_b += m_1[i,0] + m_3[i,0]
        my_b += m_1[i,1] + m_3[i,1]
        mz_b += m_1[i,2] + m_3[i,2]
        m_abs_b += np.sqrt(np.sum(m_1[i]**2)) + np.sqrt(np.sum(m_3[i]**2))
        
        mx_c += m_2[i,0]
        my_c += m_2[i,1]
        mz_c += m_2[i,2]
        m_abs_c += np.sqrt(np.sum(m_2[i]**2))

        Energy_Heisenberg_ab += (m2_31_1[i,0] + m2_31_1[i,4] + m2_31_1[i,8]) + (m2_31_2[i,0] + m2_31_2[i,4] + m2_31_2[i,8])
        Energy_Heisenberg_bc += (m_12[i,0] + m_12[i,4] + m_12[i,8]) + (m_23[i,0] + m_23[i,4] + m_23[i,8]) + (m2_12[i,0] + m2_12[i,4] + m2_12[i,8])
        Energy_Heisenberg_ca += (m2_23[i,0] + m2_23[i,4] + m2_23[i,8])


        if (Dz !=0.0):
            Energy_Dz_ab += -(m2_31_1[i,1] - m2_31_1[i,3]) -  (m2_31_2[i,1] - m2_31_2[i,3])
            Energy_Dz_bc += (m_12[i,1] - m_12[i,3]) - (m_23[i,1] - m_23[i,3]) + (m2_12[i,1] - m2_12[i,3])
            Energy_Dz_ca += (m2_23[i,1] - m2_23[i,3])

        if (Dp !=0.0):
            Energy_Dp_ab += np.cos(np.pi/2.0) * (-(m2_31_1[i,5] - m2_31_1[i,7]) -  (m2_31_2[i,5] - m2_31_2[i,7]))
            Energy_Dp_ab += np.sin(np.pi/2.0) * (-(m2_31_1[i,6] - m2_31_1[i,2]) -  (m2_31_2[i,6] - m2_31_2[i,2]))
            Energy_Dp_bc += np.cos(7.0*np.pi/6.0) * ((m_12[i,5] - m_12[i,7]) - (m_23[i,5] - m_23[i,7]) + (m2_12[i,5] - m2_12[i,7]))
            Energy_Dp_bc += np.sin(7.0*np.pi/6.0) * ((m_12[i,6] - m_12[i,2]) - (m_23[i,6] - m_23[i,2]) + (m2_12[i,6] - m2_12[i,2]))
            Energy_Dp_ca += np.cos(11.0*np.pi/6.0) * (m2_23[i,5] - m2_23[i,7])
            Energy_Dp_ca += np.sin(11.0*np.pi/6.0) * (m2_23[i,6] - m2_23[i,2])

        ## B sub lattice
        i = B_sub_list[n_sub]

        mx_a += m_1[i,0] + m_3[i,0]
        my_a += m_1[i,1] + m_3[i,1]
        mz_a += m_1[i,2] + m_3[i,2]
        m_abs_a += np.sqrt(np.sum(m_1[i]**2)) + np.sqrt(np.sum(m_3[i]**2))

        mx_c += m_2[i,0]
        my_c += m_2[i,1]
        mz_c += m_2[i,2]
        m_abs_c += np.sqrt(np.sum(m_2[i]**2))


        Energy_Heisenberg_ab += (m2_31_1[i,0] + m2_31_1[i,4] + m2_31_1[i,8]) + (m2_31_2[i,0] + m2_31_2[i,4] + m2_31_2[i,8])
        Energy_Heisenberg_bc += (m2_23[i,0] + m2_23[i,4] + m2_23[i,8])
        Energy_Heisenberg_ca += (m_12[i,0] + m_12[i,4] + m_12[i,8]) + (m_23[i,0] + m_23[i,4] + m_23[i,8]) + (m2_12[i,0] + m2_12[i,4] + m2_12[i,8])


        if (Dz !=0.0):
            Energy_Dz_ab += (m2_31_1[i,1] - m2_31_1[i,3]) +  (m2_31_2[i,1] - m2_31_2[i,3])
            Energy_Dz_bc += -(m2_23[i,1] - m2_23[i,3])
            Energy_Dz_ca += -(m_12[i,1] - m_12[i,3]) + (m_23[i,1] - m_23[i,3]) - (m2_12[i,1] - m2_12[i,3])

        if (Dp !=0.0):
            Energy_Dp_ab += np.cos(np.pi/2.0) * ((m2_31_1[i,5] - m2_31_1[i,7]) +  (m2_31_2[i,5] - m2_31_2[i,7]))
            Energy_Dp_ab += np.sin(np.pi/2.0) * ((m2_31_1[i,6] - m2_31_1[i,2]) +  (m2_31_2[i,6] - m2_31_2[i,2]))
            Energy_Dp_bc += np.cos(7.0*np.pi/6.0) * (-(m2_23[i,5] - m2_23[i,7]))
            Energy_Dp_bc += np.sin(7.0*np.pi/6.0) * (-(m2_23[i,6] - m2_23[i,2]))
            Energy_Dp_ca += np.cos(11.0*np.pi/6.0) * (-(m_12[i,5] - m_12[i,7]) + (m_23[i,5] - m_23[i,7]) - (m2_12[i,5] - m2_12[i,7]))
            Energy_Dp_ca += np.sin(11.0*np.pi/6.0) * (-(m_12[i,6] - m_12[i,2]) + (m_23[i,6] - m_23[i,2]) - (m2_12[i,6] - m2_12[i,2]))

    ## per unit
    Energy_Heisenberg_ab *= Jab / N_UNIT
    Energy_Heisenberg_bc *= Jbc / N_UNIT
    Energy_Heisenberg_ca *= Jca / N_UNIT

    Energy_Dz_ab *=Dz / N_UNIT
    Energy_Dz_bc *=Dz / N_UNIT
    Energy_Dz_ca *=Dz / N_UNIT

    Energy_Dp_ab *=Dp / N_UNIT
    Energy_Dp_bc *=Dp / N_UNIT
    Energy_Dp_ca *=Dp / N_UNIT

    ##
    mx_a /= N_UNIT
    mx_b /= N_UNIT
    mx_c /= N_UNIT
    my_a /= N_UNIT
    my_b /= N_UNIT
    my_c /= N_UNIT
    mz_a /= N_UNIT
    mz_b /= N_UNIT
    mz_c /= N_UNIT

    m_abs_a /= N_UNIT
    m_abs_b /= N_UNIT
    m_abs_c /= N_UNIT

    Energy_hx_a = -hx * mx_a
    Energy_hx_b = -hx * mx_b
    Energy_hx_c = -hx * mx_c

    Energy_hz_a = -hz * mz_a
    Energy_hz_b = -hz * mz_b
    Energy_hz_c = -hz * mz_c

    time_obs += time.time() - start_obs

    print "## Mag_total a",hx,hz,Dz,Dp,mx_a,my_a,mz_a,np.sqrt(mx_a**2 +my_a**2 + mz_a **2),m_abs_a
    print "## Mag_total b",hx,hz,Dz,Dp,mx_b,my_b,mz_b,np.sqrt(mx_b**2 +my_b**2 + mz_b **2),m_abs_b
    print "## Mag_total c",hx,hz,Dz,Dp,mx_c,my_c,mz_c,np.sqrt(mx_c**2 +my_c**2 + mz_c **2),m_abs_c
    file_Mag_sub.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(mx_a)+" "+repr(my_a)+" "+repr(mz_a)+" "+repr(np.sqrt(mx_a**2+my_a**2+mz_a**2))+" "+repr(m_abs_a)+"\n")
    file_Mag_sub.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(mx_b)+" "+repr(my_b)+" "+repr(mz_b)+" "+repr(np.sqrt(mx_b**2+my_b**2+mz_b**2))+" "+repr(m_abs_b)+"\n")
    file_Mag_sub.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(mx_c)+" "+repr(my_c)+" "+repr(mz_c)+" "+repr(np.sqrt(mx_c**2+my_c**2+mz_c**2))+" "+repr(m_abs_c)+"\n")


    
    print "## Energy ab",hx,hz,Dz,Dp,Energy_Heisenberg_ab + Energy_Dz_ab +Energy_Dp_ab,Energy_Heisenberg_ab,Energy_Dz_ab, Energy_Dp_ab
    print "## Energy bc",hx,hz,Dz,Dp,Energy_Heisenberg_bc + Energy_Dz_bc +Energy_Dp_bc,Energy_Heisenberg_bc,Energy_Dz_bc, Energy_Dp_bc
    print "## Energy ca",hx,hz,Dz,Dp,Energy_Heisenberg_ca + Energy_Dz_ca +Energy_Dp_ca,Energy_Heisenberg_ca,Energy_Dz_ca, Energy_Dp_ca

    print "## Energy_h a",hx,hz,Dz,Dp,Energy_hx_a,Energy_hz_a
    print "## Energy_h b",hx,hz,Dz,Dp,Energy_hx_b,Energy_hz_b
    print "## Energy_h c",hx,hz,Dz,Dp,Energy_hx_c,Energy_hz_c

    file_Energy_sub.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(Energy_Heisenberg_ab+Energy_Dz_ab+Energy_Dp_ab)+" "+repr(Energy_Heisenberg_ab)+" "+repr(Energy_Dz_ab)+" "+repr(Energy_Dp_ab)+"\n")
    file_Energy_sub.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(Energy_Heisenberg_bc+Energy_Dz_bc+Energy_Dp_bc)+" "+repr(Energy_Heisenberg_bc)+" "+repr(Energy_Dz_bc)+" "+repr(Energy_Dp_bc)+"\n")
    file_Energy_sub.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(Energy_Heisenberg_ca+Energy_Dz_ca+Energy_Dp_ca)+" "+repr(Energy_Heisenberg_ca)+" "+repr(Energy_Dz_ca)+" "+repr(Energy_Dp_ca)+"\n")

    file_Energy_sub.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(Energy_hx_a)+" "+repr(Energy_hz_a)+"\n")
    file_Energy_sub.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(Energy_hx_b)+" "+repr(Energy_hz_b)+"\n")
    file_Energy_sub.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(Energy_hx_c)+" "+repr(Energy_hz_c)+"\n")

    
    print "## Scalar Chirality",hx,hz,Dz,Dp, Scalar_Chirality


    mx = (mx_a + mx_b + mx_c)/3.0
    my = (my_a + my_b + my_c)/3.0
    mz = (mz_a + mz_b + mz_c)/3.0
    m_abs = (m_abs_a + m_abs_b + m_abs_c) /3.0
    
    print "## Total Magnetization"
    print hx,hz,Dz,Dp,mx,my,mz,np.sqrt(mx**2 +my**2 + mz **2),m_abs
    file_Mag.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(mx)+" "+repr(my)+" "+repr(mz)+" "+repr(np.sqrt(mx**2+my**2+mz**2))+" "+repr(m_abs)+"\n")
    
    total_Energy_Heisenberg = (Energy_Heisenberg_ab + Energy_Heisenberg_bc + Energy_Heisenberg_ca) / 3.0
    total_Energy_Dz = (Energy_Dz_ab + Energy_Dz_bc + Energy_Dz_ca) / 3.0
    total_Energy_Dp = (Energy_Dp_ab + Energy_Dp_bc + Energy_Dp_ca) / 3.0
    total_Energy_hx = (Energy_hx_a + Energy_hx_b + Energy_hx_c) / 3.0
    total_Energy_hz = (Energy_hz_a + Energy_hz_b + Energy_hz_c) / 3.0

    total_Energy = total_Energy_Heisenberg + total_Energy_Dz + total_Energy_Dp + total_Energy_hx + total_Energy_hz

    print "## Total Energy"
    print hx,hz,Dz,Dp,total_Energy, total_Energy_Heisenberg, total_Energy_Dz, total_Energy_Dp, total_Energy_hx, total_Energy_hz
    file_Energy.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(total_Energy)+" "+repr(total_Energy_Heisenberg)+" "+" "+repr(total_Energy_Dz)+" "+" "+repr(total_Energy_Dp)+" "+" "+repr(total_Energy_hx)+" "+repr(total_Energy_hx)+"\n")

    print "## Total Scalar Chirality"
    print hx,hz,Dz,Dp,Total_Scalar_Chirality
    print "## time simple update=",time_simple_update
    print "## time full update=",time_full_update
    print "## time environment=",time_env
    print "## time observable=",time_obs

    file_Timer.write(repr(hx)+" "+repr(hz)+" "+repr(Dz)+" "+repr(Dp)+" "+repr(time_simple_update)+" "+repr(time_full_update)+" "+repr(time_env)+" "+repr(time_obs)+"\n")

if __name__ == "__main__":
    main()
    
