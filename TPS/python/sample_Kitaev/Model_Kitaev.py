# coding:utf-8
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import time
import argparse
import os
import os.path

## import basic routines
from PEPS_Basics import *
from PEPS_Parameters import *
from Square_lattice_CTM import *

def parse_args():
    parser = argparse.ArgumentParser(description='iPEPS simulation for Kitaev Heisenberg model')
    parser.add_argument('-Kx', metavar='Kx',dest='Kx', type=float, default=-1.0,
                        help='set Kx (default = -1.0)')
    parser.add_argument('-Ky', metavar='Ky',dest='Ky', type=float, default=-1.0,
                        help='set Ky (default = -1.0)' )
    parser.add_argument('-Kz', metavar='Kz',dest='Kz', type=float, default=-1.0,
                        help='set Kz (default = -1.0)')
    parser.add_argument('-hxyz', metavar='hxyz',dest='hxyz', type=float, default=0.0,
                        help='set hxyz (default = 0.0)')
    parser.add_argument('-hp', metavar='hp',dest='hp', type=float, default=0.0,
                        help='set hp (default = 0.0)')
    parser.add_argument('-hz', metavar='hz',dest='hz', type=float, default=0.0,
                        help='set hz (default = 0.0)')
    parser.add_argument('-s', metavar='seed',dest='seed', type=int, default=11,
                        help='set random seed (default = 11)')
    parser.add_argument('-J', metavar='J',dest='J', type=float, default=0.0,
                        help='set Heisenberg coupling J (default = 0.0)')

    parser.add_argument('-ts', metavar='tau',dest='tau', type=float, default=0.01,
                        help='set tau for simple update (default = 0.01)')

    parser.add_argument('-tf', metavar='tau_full',dest='tau_full', type=float, default=0.01,
                        help='set tau for full update (default = 0.01)')

    parser.add_argument('-ss', metavar='tau_step',dest='tau_step', type=int, default=5000,
                        help='set step for simple update (default = 5000)')

    parser.add_argument('-sf', metavar='tau_full_step',dest='tau_full_step', type=int, default=0,
                        help='set step for full update (default = 0)')
    
    parser.add_argument('-i', metavar='initial_type',dest='initial_type', type=int, default=0,
                        help='set initial state type (0:random, 1:ferro, 2:AF, 3:zigzag, 4:stripy, -1 or negative value:from file (default input_Tn.npz)) (default = 0)')

    parser.add_argument('-id', metavar='initial_direction',dest='initial_direction', type=int, default=0,
                        help='set spin direction for (classical) initial state (0:z, 1:x, 2:y, 3:(1,1,1), 4:(1,-1,0), 5:(1,1,0) 6:(1,1,-1) (default=0))')

    parser.add_argument('-ir', metavar='random_amp',dest='random_amp', type=float, default=0.01,
                        help='set amplitude of random noise for the inisital state (default = 0.01)')

    parser.add_argument('--second_ST', action='store_const', const=True, default=False,
                        help='use second order Suzuki Trotter decomposition for ITE (default = False)')

    parser.add_argument('-o', metavar='output_file',dest='output_file', default="output",
                        help='set output filename prefix for optimized tensor (default = "output")')
    
    parser.add_argument('-in', metavar='input_file',dest='input_file', default="input_Tn.npz",
                        help='set input filename for initial tensors (default = "input_Tn.npz")')
    parser.add_argument('-tag', metavar='tag',dest='tag', default="",
                        help='set tag for output dirrectory (default = "")')
    parser.add_argument('--append_output', action='store_const', const=True, default=False,
                        help='append output date to existing files instead creating new file (default = False)')
    return parser.parse_args()


def Set_Hamiltonian(hxyz,hp,hz,J,K,direction=0):
    ## 0:x, 1:y, else:z
    Ham = np.zeros((4,4),dtype=complex)

    if direction==0:
    # x-bond
        Ham[0,0] = 0.25 * J
        Ham[0,1] = 0.0
        Ham[0,2] = 0.0
        Ham[0,3] = 0.25 * K

        Ham[1,0] = 0.0
        Ham[1,1] = -0.25 * J
        Ham[1,2] = 0.25 * K + 0.5 * J
        Ham[1,3] = 0.0

        Ham[2,0] = 0.0
        Ham[2,1] = 0.25 * K + 0.5 * J
        Ham[2,2] = -0.25 * J
        Ham[2,3] = 0.0

        Ham[3,0] = 0.25 * K
        Ham[3,1] = 0.0
        Ham[3,2] = 0.0
        Ham[3,3] = 0.25 * J
    elif direction==1:
    # y-bond
        Ham[0,0] = 0.25 * J
        Ham[0,1] = 0.0
        Ham[0,2] = 0.0
        Ham[0,3] = -0.25 * K

        Ham[1,0] = 0.0
        Ham[1,1] = -0.25 * J
        Ham[1,2] = 0.25 * K + 0.5 * J
        Ham[1,3] = 0.0

        Ham[2,0] = 0.0
        Ham[2,1] = 0.25 * K + 0.5 * J
        Ham[2,2] = -0.25 * J
        Ham[2,3] = 0.0


        Ham[3,0] = -0.25 * K
        Ham[3,1] = 0.0
        Ham[3,2] = 0.0
        Ham[3,3] = 0.25 * J

    else:
    # z-bond
        Ham[0,0] = 0.25 * K + 0.25 * J
        Ham[0,3] = 0.0

        Ham[1,1] = -0.25 * K - 0.25 * J
        Ham[1,2] = 0.5 * J

        Ham[2,2] = -0.25 * K - 0.25 * J
        Ham[2,1] = 0.5 * J

        Ham[3,0] = 0.0
        Ham[3,3] =  0.25 * K + 0.25 * J

    ## hxyz field (1,1,1) direction
    if not (hxyz ==0.0):
        factor = 0.5/np.sqrt(3.0)/3.0

        Ham[0,0] += - 2.0 * factor * hxyz
        Ham[0,1] += - (1.0 - 1.0j) * factor * hxyz
        Ham[0,2] += - (1.0 - 1.0j) * factor * hxyz

        Ham[1,0] += - (1.0 + 1.0j) * factor * hxyz
        Ham[1,3] += - (1.0 - 1.0j) * factor * hxyz

        Ham[2,0] += - (1.0 + 1.0j) * factor * hxyz
        Ham[2,3] += - (1.0 - 1.0j) * factor * hxyz

        Ham[3,1] += - (1.0 + 1.0j) * factor * hxyz
        Ham[3,2] += - (1.0 + 1.0j) * factor * hxyz
        Ham[3,3] += 2.0 * factor * hxyz
    
    ## hp field (1,-1,0) direction
    if not (hp ==0.0):
        factor = 0.5/np.sqrt(2.0)/3.0

        Ham[0,1] += - (1.0 + 1.0j) * factor * hp
        Ham[0,2] += - (1.0 + 1.0j) * factor * hp

        Ham[1,0] += - (1.0 - 1.0j) * factor * hp
        Ham[1,3] += - (1.0 + 1.0j) * factor * hp

        Ham[2,0] += - (1.0 - 1.0j) * factor * hp
        Ham[2,3] += - (1.0 + 1.0j) * factor * hp

        Ham[3,1] += - (1.0 - 1.0j) * factor * hp
        Ham[3,2] += - (1.0 - 1.0j) * factor * hp
    if not (hz ==0.0):
        Ham[0,0] += -hz/3.0
        Ham[3,3] += +hp/3.0

    return Ham

def Initialize_Tensors(Tn,seed,initial_type=0,initial_direction=0,random_amp=0.01):

    np.random.seed(seed)

    def Create_initial(D):
        for i in range(0,N_UNIT):
            ix = i%LX_ori
            iy = i/LX_ori
            if isinstance(Tn[i][0,0,0,0,0],complex):
                if (ix + iy)%2 == 0:
                    Tn[i] = np.zeros((1,D,D,D,2),dtype=complex)
                    Tn_temp_real = (np.random.rand(1,D,D,D,2)-0.5) * random_amp
                    Tn_temp_imag = (np.random.rand(1,D,D,D,2)-0.5) * random_amp
                    Tn[i][:] = Tn_temp_real + 1.0j * Tn_temp_imag
                else: 
                    Tn[i] = np.zeros((D,D,1,D,2),dtype=complex)
                    Tn_temp_real = (np.random.rand(D,D,1,D,2)-0.5) * random_amp
                    Tn_temp_imag = (np.random.rand(D,D,1,D,2)-0.5) * random_amp
                    Tn[i][:] = Tn_temp_real + 1.0j * Tn_temp_imag
            else:
                if (ix + iy)%2 == 0:
                    Tn[i] = np.zeros((1,D,D,D,2))
                    Tn[i][:] = (np.random.rand(1,D,D,D,2)-0.5) * random_amp
                else:
                    Tn[i] = np.zeros((D,D,1,D,2))
                    Tn[i][:] = (np.random.rand(D,D,1,D,2)-0.5) * random_amp

    local_vector_p = np.zeros((2),dtype=Tn[0].dtype)
    local_vector_m = np.zeros((2),dtype=Tn[0].dtype)
    
    if initial_direction==1:
        ## x direction
        local_vector_p[0] = 1.0
        local_vector_p[1] = 1.0

        local_vector_m[0] = 1.0
        local_vector_m[1] = -1.0
    elif initial_direction ==2:
        ## y direction
        local_vector_p[0] = 1.0
        local_vector_p[1] = 1.0j

        local_vector_m[0] = 1.0
        local_vector_m[1] = -1.0j
    elif initial_direction ==3:
        ## (1,1,1) direction
        local_vector_p[0] = 1.0
        local_vector_p[1] = (np.sqrt(3.0)-1.0)*0.5 * (1.0 +1.0j)

        local_vector_m[0] = 1.0
        local_vector_m[1] = (-np.sqrt(3.0)-1.0)*0.5 * (1.0 +1.0j)
    elif initial_direction ==4:
        ## (1,-1,0) direction
        local_vector_p[0] = 1.0
        local_vector_p[1] = np.sqrt(2.0)*0.5 * (1.0-1.0j)

        local_vector_m[0] = 1.0
        local_vector_m[1] = -np.sqrt(2.0)*0.5 * (1.0-1.0j)
    elif initial_direction ==5:
        ## (1,1,0) direction
        local_vector_p[0] = 1.0
        local_vector_p[1] = np.sqrt(2.0)*0.5 * (1.0+1.0j)

        local_vector_m[0] = 1.0
        local_vector_m[1] = -np.sqrt(2.0)*0.5 * (1.0+1.0j)
    elif initial_direction ==6:
        ## (1,1,-1) direction
        local_vector_p[0] = 1.0
        local_vector_p[1] = (np.sqrt(3.0)+1.0)*0.5 * (1.0+1.0j)

        local_vector_m[0] = 1.0
        local_vector_m[1] = (-np.sqrt(3.0)+1.0)*0.5 * (1.0+1.0j)
    else:
        ## z direction
        local_vector_p[0] = 1.0
        local_vector_p[1] = 0.0

        local_vector_m[0] = 0.0
        local_vector_m[1] = 1.0

    local_vector_p /= np.sqrt(np.abs(np.dot(local_vector_p,local_vector_p.conj())))
    local_vector_m /= np.sqrt(np.abs(np.dot(local_vector_m,local_vector_m.conj())))
    if initial_type == 1:
        ## ferro
        Create_initial(2)
        for i in range(0,N_UNIT):
            Tn[i][0,0,0,0]+=local_vector_p.copy()

    elif initial_type == 2:
        ## AF
        Create_initial(2)
        for i in range(0,N_UNIT):
            ix = i%LX_ori
            iy = i/LX_ori
            if (ix + iy)%2 == 0:
                Tn[i][0,0,0,0]+=local_vector_p.copy()
            else:
                Tn[i][0,0,0,0]+=local_vector_m.copy()
    elif initial_type == 3:
        ## zigzag
        Create_initial(2)
        for i in range(0,N_UNIT):
            ix = i%LX_ori
            iy = i/LX_ori
            if ix%2 == 0:
                Tn[i][0,0,0,0]+=local_vector_p.copy()
            else:
                Tn[i][0,0,0,0]+=local_vector_m.copy()

    elif initial_type == 4:
        ## stripy
        Create_initial(2)
        for i in range(0,N_UNIT):
            ix = i%LX_ori
            iy = i/LX_ori
            if iy%2 == 0:
                Tn[i][0,0,0,0,0]+=1.0
                Tn[i][0,0,0,0,1]+=0.0
            else:
                Tn[i][0,0,0,0,0]+=0.0
                Tn[i][0,0,0,0,1]+=1.0
    else:
        Create_initial(D)



                                                                                                                                                                                 
def main():                
    ## timers
    time_simple_update=0.0
    time_full_update=0.0
    time_env=0.0
    time_obs=0.0

    ## Parameters
    args = parse_args()
    Kx = args.Kx
    Ky = args.Ky
    Kz = args.Kz

    J = args.J
    
    hxyz = args.hxyz
    hp = args.hp
    hz = args.hz

    seed = args.seed
    random_amp = args.random_amp
    initial_direction = args.initial_direction

                        
    tau = args.tau
    tau_step = args.tau_step

    tau_full = args.tau_full
    tau_full_step = args.tau_full_step

    second_ST = args.second_ST

    initial_type = args.initial_type

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
        file_Mag = open(output_data_dir+"/Magnetization.dat","a")
        file_Mag_sub = open(output_data_dir+"/Magnetization_Sub.dat","a")
        file_flux = open(output_data_dir+"/flux.dat","a")
        file_params = open(output_data_dir+"/output_params.dat","a")
        file_Timer = open(output_data_dir+"/Timer.dat","a")
    else:
        file_Energy = open(output_data_dir+"/Energy.dat","w")
        file_Energy_sub = open(output_data_dir+"/Energy_Sub.dat","w")
        file_Mag = open(output_data_dir+"/Magnetization.dat","w")
        file_Mag_sub = open(output_data_dir+"/Magnetization_Sub.dat","w")
        file_flux = open(output_data_dir+"/flux.dat","w")
        file_params = open(output_data_dir+"/output_params.dat","w")
        file_Timer = open(output_data_dir+"/Timer.dat","w")

        

    ##
    
    ##
    print "## Logs: Kx =",Kx
    print "## Logs: Ky =",Ky
    print "## Logs: Kz =",Kz
    print "## Logs: J =",J
    print "## Logs: hxyz =",hxyz
    print "## Logs: hp =",hp
    print "## Logs: hz =",hz
    print "## Logs: seed =",seed
    print "## Logs: random_amp =",random_amp
    print "## Logs: initial_direction =",initial_direction
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

    file_params.write("Kx = "+repr(Kx) + "\n")
    file_params.write("Ky = "+repr(Ky) + "\n")
    file_params.write("Kz = "+repr(Kz) + "\n")
    file_params.write("J = "+repr(J) + "\n")
    file_params.write("hxyz = "+repr(hxyz) + "\n")
    file_params.write("hp = "+repr(hp) + "\n")
    file_params.write("hz = "+repr(hz) + "\n")
    file_params.write("seed = "+repr(seed) + "\n")
    file_params.write("random_amp = "+repr(random_amp) + "\n")
    file_params.write("initial_direction = "+repr(initial_direction) + "\n")
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
    Tn=[np.zeros((1,D,D,D,2),dtype=TENSOR_DTYPE)]
    eTt=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTr=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTb=[np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE)]
    eTl=[np.zeros((CHI,CHI,1,1),dtype=TENSOR_DTYPE)]

    C1=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C2=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C3=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)
    C4=np.zeros((N_UNIT,CHI,CHI),dtype=TENSOR_DTYPE)



    for i in range(1,N_UNIT):
        ix = i%LX_ori
        iy = i/LX_ori

        if (ix + iy)%2 == 0:
            Tn.append(np.zeros((1,D,D,D,2),dtype=TENSOR_DTYPE))
            eTt.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
            eTr.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
            eTb.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
            eTl.append(np.zeros((CHI,CHI,1,1),dtype=TENSOR_DTYPE))
        else:
            Tn.append(np.zeros((D,D,1,D,2),dtype=TENSOR_DTYPE))
            eTt.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
            eTr.append(np.zeros((CHI,CHI,1,1),dtype=TENSOR_DTYPE))
            eTb.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
            eTl.append(np.zeros((CHI,CHI,D,D),dtype=TENSOR_DTYPE))
    

    if initial_type >= 0:
        Initialize_Tensors(Tn,seed,initial_type,initial_direction,random_amp)
    else:
        #input from file
        input_tensors = np.load(input_file)
        for i in range(N_UNIT):
            key='arr_'+repr(i)
            Tn_in = input_tensors[key]
            Tn[i][0:min(Tn[i].shape[0],Tn_in.shape[0]),0:min(Tn[i].shape[1],Tn_in.shape[1]),0:min(Tn[i].shape[2],Tn_in.shape[2]),0:min(Tn[i].shape[3],Tn_in.shape[3]),0:min(Tn[i].shape[4],Tn_in.shape[4])] = Tn_in[0:min(Tn[i].shape[0],Tn_in.shape[0]),0:min(Tn[i].shape[1],Tn_in.shape[1]),0:min(Tn[i].shape[2],Tn_in.shape[2]),0:min(Tn[i].shape[3],Tn_in.shape[3]),0:min(Tn[i].shape[4],Tn_in.shape[4])]
        

    
    lambda_D = np.ones(Tn[0].shape[1],dtype=float)
    lambda_1 = np.ones(1,dtype=float)

    lambda_type_A = [lambda_1.copy(),lambda_D.copy(),lambda_D.copy(),lambda_D.copy()]
    lambda_tensor = [lambda_type_A]
    for i in range(1,N_UNIT):
        ix = i%LX_ori
        iy = i/LX_ori
        if (ix + iy)%2 == 0:
            lambda_type_A = [lambda_1.copy(),lambda_D.copy(),lambda_D.copy(),lambda_D.copy()]
            lambda_tensor.append(lambda_type_A)
        else:
            lambda_type_B = [lambda_D.copy(),lambda_D.copy(),lambda_1.copy(),lambda_D.copy()]
            lambda_tensor.append(lambda_type_B)

        
    Ham_x = Set_Hamiltonian(hxyz,hp,hz,J,Kx,direction=0)
    Ham_y = Set_Hamiltonian(hxyz,hp,hz,J,Ky,direction=1)
    Ham_z = Set_Hamiltonian(hxyz,hp,hz,J,Kz,direction=2)


    s_x,U_x = linalg.eigh(Ham_x)
    s_y,U_y = linalg.eigh(Ham_y)
    s_z,U_z = linalg.eigh(Ham_z)
    

    op12_x =  np.dot(np.dot(U_x,np.diag(np.exp(-tau * s_x))),U_x.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12_y =  np.dot(np.dot(U_y,np.diag(np.exp(-tau * s_y))),U_y.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12_z =  np.dot(np.dot(U_z,np.diag(np.exp(-tau * s_z))),U_z.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)


    op12_x_2 =  np.dot(np.dot(U_x,np.diag(np.exp(-tau * s_x * 0.5))),U_x.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
    op12_y_2 =  np.dot(np.dot(U_y,np.diag(np.exp(-tau * s_y * 0.5))),U_y.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)

    
    start_simple=time.time()


    ## simple update
    if second_ST:
        for int_tau in range(0,tau_step):

            ## x-bond 
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_x_2,1,D)
                lambda_tensor[num][1] = lambda_c
                lambda_tensor[num_j][3] = lambda_c

            ## y-bond
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_y_2,1,D)
                lambda_tensor[num][1] = lambda_c
                lambda_tensor[num_j][3] = lambda_c

            ## z-bond
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_z,2,D)
                lambda_tensor[num][2] = lambda_c
                lambda_tensor[num_j][0] = lambda_c

            ## y-bond
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_y_2,1,D)
                lambda_tensor[num][1] = lambda_c
                lambda_tensor[num_j][3] = lambda_c

            ## x-bond 
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_x_2,1,D)
                lambda_tensor[num][1] = lambda_c
                lambda_tensor[num_j][3] = lambda_c
    else:
        for int_tau in range(0,tau_step):
            ## x-bond 
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_x,1,D)
                lambda_tensor[num][1] = lambda_c
                lambda_tensor[num_j][3] = lambda_c

            ## y-bond
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_y,1,D)
                lambda_tensor[num][1] = lambda_c
                lambda_tensor[num_j][3] = lambda_c

            ## z-bond
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j], lambda_c = Simple_update_bond(Tn[num],Tn[num_j],lambda_tensor[num],lambda_tensor[num_j],op12_z,2,D)
                lambda_tensor[num][2] = lambda_c
                lambda_tensor[num_j][0] = lambda_c


        ## done simple update
    time_simple_update += time.time() - start_simple

    ## Start full update
    if tau_full_step > 0:

        op12_x =  np.dot(np.dot(U_x,np.diag(np.exp(-tau_full * s_x))),U_x.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
        op12_y =  np.dot(np.dot(U_y,np.diag(np.exp(-tau_full * s_y))),U_y.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
        op12_z =  np.dot(np.dot(U_z,np.diag(np.exp(-tau_full * s_z))),U_z.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)

        op12_x_2 =  np.dot(np.dot(U_x,np.diag(np.exp(-tau_full * s_x * 0.5))),U_x.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
        op12_y_2 =  np.dot(np.dot(U_y,np.diag(np.exp(-tau_full * s_y * 0.5))),U_y.conj().T).reshape(2,2,2,2).transpose(2,3,0,1)
        ## Environment 
        start_env = time.time()
        Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)
        time_env += time.time() - start_env

    start_full = time.time()
    if second_ST:
        for int_tau in range(0,tau_full_step):

            ## x-bond 
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_x_2,1)

                if Full_Use_FFU:
                    #iy = num/LX
                    #iy_j = num_j/LX
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## y-bond
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_y_2,1)

                if Full_Use_FFU:
                    #iy = num/LX
                    #iy_j = num_j/LX
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## z-bond
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_z,2)

                if Full_Use_FFU:
                    #ix = num%LX
                    #ix_j = num_j%LX
                    ix = Tensor_position[num,0]
                    ix_j = Tensor_position[num_j,0]
                    Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                    Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                

                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## y-bond
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_y_2,1)

                if Full_Use_FFU:
                    #iy = num/LX
                    #iy_j = num_j/LX
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## x-bond 
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_x_2,1)

                if Full_Use_FFU:
                    #iy = num/LX
                    #iy_j = num_j/LX
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)


    else:
        for int_tau in range(0,tau_full_step):

            ## x-bond 
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_x,1)

                if Full_Use_FFU:
                    #iy = num/LX
                    #iy_j = num_j/LX
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## y-bond
            for i in range(0,N_UNIT/2):
                num = B_sub_list[i]
                num_j = NN_Tensor[num,1]

                Tn[num],Tn[num_j] = Full_update_bond(C4[num],C1[num_j],C2[num_j],C3[num],eTl[num],eTl[num_j],eTt[num_j],eTr[num_j],eTr[num],eTb[num],Tn[num],Tn[num_j],op12_y,1)

                if Full_Use_FFU:
                    #iy = num/LX
                    #iy_j = num_j/LX
                    iy = Tensor_position[num,1]
                    iy_j = Tensor_position[num_j,1]
                    Bottom_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy)
                    Top_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,iy_j)                
                else:
                    Calc_CTM_Environment(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn)

            ## z-bond
            for i in range(0,N_UNIT/2):
                num = A_sub_list[i]
                num_j = NN_Tensor[num,2]

                Tn[num],Tn[num_j] = Full_update_bond(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op12_z,2)

                if Full_Use_FFU:
                    #ix = num%LX
                    #ix_j = num_j%LX
                    ix = Tensor_position[num,0]
                    ix_j = Tensor_position[num_j,0]
                    Left_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix)
                    Right_move(C1,C2,C3,C4,eTt,eTr,eTb,eTl,Tn,ix_j)                

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

    op_identity = np.identity(2)

    op_mz = np.zeros((2,2))
    op_mz[0,0] = 0.5
    op_mz[1,1] = -0.5

    op_mx = np.zeros((2,2))
    op_mx[1,0] = 0.5
    op_mx[0,1] = 0.5

    op_my = np.zeros((2,2),dtype=complex)
    op_my[0,1] = 0.5j
    op_my[1,0] = -0.5j
    
    mz = np.zeros(N_UNIT)
    mx = np.zeros(N_UNIT)
    my = np.zeros(N_UNIT)

    zx = np.zeros((N_UNIT/2,3))
    zy = np.zeros((N_UNIT/2,3))
    zz = np.zeros((N_UNIT/2,3))
    xx = np.zeros((N_UNIT/2,3))
    xy = np.zeros((N_UNIT/2,3))
    xz = np.zeros((N_UNIT/2,3))
    yx = np.zeros((N_UNIT/2,3))
    yy = np.zeros((N_UNIT/2,3))
    yz = np.zeros((N_UNIT/2,3))

    start_obs = time.time()
    norm = np.zeros((N_UNIT,),dtype=complex)
    for i in range(0,N_UNIT):        
        norm[i] = Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_identity)
        mz[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_mz)/norm[i])
        mx[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_mx)/norm[i])
        my[i] = np.real(Contract_one_site(C1[i],C2[i],C3[i],C4[i],eTt[i],eTr[i],eTb[i],eTl[i],Tn[i],op_my)/norm[i])


        print "## Mag ",J, hxyz,hp,hz,i,norm[i],mx[i],my[i],mz[i],np.sqrt(mx[i]**2+my[i]**2+mz[i]**2)

    norm_x = np.zeros(N_UNIT/2,dtype=complex)
    norm_y = np.zeros(N_UNIT/2,dtype=complex)
    norm_z = np.zeros(N_UNIT/2,dtype=complex)
    for i in range(0,N_UNIT/2):
        num = A_sub_list[i]
        ## x bond
        num_j = NN_Tensor[num,1]

        norm_x[i] = Contract_two_sites_vertical(C1[num_j],C2[num_j],C3[num],C4[num],eTt[num_j],eTr[num_j],eTr[num],eTb[num],eTl[num],eTl[num_j],Tn[num_j],Tn[num],op_identity,op_identity)
        zx[i,0] = np.real(Contract_two_sites_vertical(C1[num_j],C2[num_j],C3[num],C4[num],eTt[num_j],eTr[num_j],eTr[num],eTb[num],eTl[num],eTl[num_j],Tn[num_j],Tn[num],op_mz,op_mx)/norm_x[i])
        zy[i,0] = np.real(Contract_two_sites_vertical(C1[num_j],C2[num_j],C3[num],C4[num],eTt[num_j],eTr[num_j],eTr[num],eTb[num],eTl[num],eTl[num_j],Tn[num_j],Tn[num],op_mz,op_my)/norm_x[i])
        zz[i,0] = np.real(Contract_two_sites_vertical(C1[num_j],C2[num_j],C3[num],C4[num],eTt[num_j],eTr[num_j],eTr[num],eTb[num],eTl[num],eTl[num_j],Tn[num_j],Tn[num],op_mz,op_mz)/norm_x[i])
        xx[i,0] = np.real(Contract_two_sites_vertical(C1[num_j],C2[num_j],C3[num],C4[num],eTt[num_j],eTr[num_j],eTr[num],eTb[num],eTl[num],eTl[num_j],Tn[num_j],Tn[num],op_mx,op_mx)/norm_x[i])
        xy[i,0] = np.real(Contract_two_sites_vertical(C1[num_j],C2[num_j],C3[num],C4[num],eTt[num_j],eTr[num_j],eTr[num],eTb[num],eTl[num],eTl[num_j],Tn[num_j],Tn[num],op_mx,op_my)/norm_x[i])
        xz[i,0] = np.real(Contract_two_sites_vertical(C1[num_j],C2[num_j],C3[num],C4[num],eTt[num_j],eTr[num_j],eTr[num],eTb[num],eTl[num],eTl[num_j],Tn[num_j],Tn[num],op_mx,op_mz)/norm_x[i])
        yx[i,0] = np.real(Contract_two_sites_vertical(C1[num_j],C2[num_j],C3[num],C4[num],eTt[num_j],eTr[num_j],eTr[num],eTb[num],eTl[num],eTl[num_j],Tn[num_j],Tn[num],op_my,op_mx)/norm_x[i])
        yy[i,0] = np.real(Contract_two_sites_vertical(C1[num_j],C2[num_j],C3[num],C4[num],eTt[num_j],eTr[num_j],eTr[num],eTb[num],eTl[num],eTl[num_j],Tn[num_j],Tn[num],op_my,op_my)/norm_x[i])
        yz[i,0] = np.real(Contract_two_sites_vertical(C1[num_j],C2[num_j],C3[num],C4[num],eTt[num_j],eTr[num_j],eTr[num],eTb[num],eTl[num],eTl[num_j],Tn[num_j],Tn[num],op_my,op_mz)/norm_x[i])

        ## y direction
        num_j = NN_Tensor[num,3]        

        norm_y[i] = Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_identity,op_identity)
        zx[i,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mz,op_mx)/norm_y[i])
        zy[i,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mz,op_my)/norm_y[i])
        zz[i,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mz,op_mz)/norm_y[i])
        xx[i,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mx,op_mx)/norm_y[i])
        xy[i,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mx,op_my)/norm_y[i])
        xz[i,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_mx,op_mz)/norm_y[i])
        yx[i,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_my,op_mx)/norm_y[i])
        yy[i,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_my,op_my)/norm_y[i])
        yz[i,1] = np.real(Contract_two_sites_vertical(C1[num],C2[num],C3[num_j],C4[num_j],eTt[num],eTr[num],eTr[num_j],eTb[num_j],eTl[num_j],eTl[num],Tn[num],Tn[num_j],op_my,op_mz)/norm_y[i])

        ## z direction
        num_j = NN_Tensor[num,2]        

        norm_z[i] = Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_identity,op_identity)
        zx[i,2] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mz,op_mx)/norm_z[i])
        zy[i,2] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mz,op_my)/norm_z[i])
        zz[i,2] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mz,op_mz)/norm_z[i])
        xx[i,2] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mx,op_mx)/norm_z[i])
        xy[i,2] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mx,op_my)/norm_z[i])
        xz[i,2] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_mx,op_mz)/norm_z[i])
        yx[i,2] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_my,op_mx)/norm_z[i])
        yy[i,2] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_my,op_my)/norm_z[i])
        yz[i,2] = np.real(Contract_two_sites_holizontal(C1[num],C2[num_j],C3[num_j],C4[num],eTt[num],eTt[num_j],eTr[num_j],eTb[num_j],eTb[num],eTl[num],Tn[num],Tn[num_j],op_my,op_mz)/norm_z[i])
        

        print "## Dot", J, hxyz,hp,hz,i,norm_x[i],norm_y[i],norm_z[i],xx[i,0],yy[i,0],zz[i,0],xx[i,1],yy[i,1],zz[i,1],xx[i,2],yy[i,2],zz[i,2]
        print "## Dot Off", J, hxyz,hp,hz,i,norm_x[i],norm_y[i],norm_z[i],xy[i,0],xz[i,0],yx[i,0],yz[i,0],zx[i,0],zy[i,0],xy[i,1],xz[i,1],yx[i,1],yz[i,1],zx[i,1],zy[i,1],xy[i,2],xz[i,2],yx[i,2],yz[i,2],zx[i,2],zy[i,2]
        
    time_obs += time.time() - start_obs

    Energy_hxyz = 0.0
    Energy_hp = 0.0
    Energy_hz = 0.0
    Mag_x = 0.0
    Mag_y = 0.0
    Mag_z = 0.0
    Mag_abs = 0.0
    for i in range(N_UNIT):
        Mag_x += mx[i]
        Mag_y += my[i]
        Mag_z += mz[i]
        Mag_abs += np.sqrt(mx[i]**2 + my[i]**2 + mz[i]**2)

        file_Mag_sub.write(repr(J)+" "+repr(hxyz)+" "+repr(hp)+" "+repr(hz)+" "+repr(i)+" "+repr(mx[i])+" "+repr(my[i])+" "+" "+repr(mz[i])+" "+" "+repr((mx[i]+my[i]+mz[i])/np.sqrt(3.0))+" "+" "+repr((mx[i]-my[i])/np.sqrt(2.0))+" "+repr(norm[i])+"\n")

        Energy_hxyz += -hxyz * (mx[i]+my[i]+mz[i])/np.sqrt(3.0)
        Energy_hp += -hp * (mx[i]-my[i])/np.sqrt(2.0)
        Energy_hz += -hz * mz[i]
        

    Mag_x /= N_UNIT
    Mag_y /= N_UNIT
    Mag_z /= N_UNIT
    Mag_abs /= N_UNIT
    
    file_Mag_sub.write("\n")
    print "## Magnetization:"
    print J,hxyz, hp, hz,(Mag_x + Mag_y + Mag_z)/np.sqrt(3.0),(Mag_x - Mag_y)/np.sqrt(2.0),Mag_abs,Mag_x,Mag_y,Mag_z
    file_Mag.write(repr(J)+" "+repr(hxyz)+" "+repr(hp)+" "+repr(hz)+" "+repr((Mag_x + Mag_y + Mag_z)/np.sqrt(3.0))+" "+repr((Mag_x - Mag_y)/np.sqrt(2.0))+" "+" "+repr(Mag_abs)+" "+" "+repr(Mag_x)+" "+" "+repr(Mag_y)+" "+repr(Mag_z)+"\n")

    Energy_x = 0.0
    Energy_y = 0.0
    Energy_z = 0.0
    for i in range(N_UNIT/2):
        Ex = Kx * xx[i,0]  + J * (xx[i,0] + yy[i,0] + zz[i,0])
        Ey = Ky * yy[i,1]  + J * (xx[i,1] + yy[i,1] + zz[i,1])
        Ez = Kz * zz[i,2]  + J * (xx[i,2] + yy[i,2] + zz[i,2])
        
        file_Energy_sub.write(repr(J)+" "+repr(hxyz)+" "+repr(hp)+" "+repr(hz)+" "+repr(i)+" "+repr(Ex)+" "+repr(Ey)+" "+" "+repr(Ez)+" "+" "+repr(norm_x[i])+" "+" "+repr(norm_y[i])+" "+repr(norm_z[i])+"\n")

        Energy_x += Ex
        Energy_y += Ey
        Energy_z += Ez

    
    file_Energy_sub.write("\n")

    Energy_x /= N_UNIT
    Energy_y /= N_UNIT
    Energy_z /= N_UNIT
    Energy_hxyz /= N_UNIT
    Energy_hp /= N_UNIT
    Energy_hz /= N_UNIT

    Energy = Energy_x + Energy_y + Energy_z + Energy_hxyz + Energy_hp + Energy_hz

    print "## Energy per site:"
    print J,hxyz, hp, hz,Energy,Energy_x,Energy_y,Energy_z,Energy_hxyz,Energy_hp,Energy_hz
    file_Energy.write(repr(J)+" "+repr(hxyz)+" "+repr(hp)+" "+repr(hz)+" "+repr(Energy)+" "+repr(Energy_x)+" "+" "+repr(Energy_y)+" "+" "+repr(Energy_z)+" "+" "+repr(Energy_hxyz)+" "+repr(Energy_hp)+" "+repr(Energy_hz)+"\n")



    print "## time simple update=",time_simple_update
    print "## time full update=",time_full_update
    print "## time environment=",time_env
    print "## time observable=",time_obs

    file_Timer.write(repr(J)+" "+repr(hxyz)+" "+repr(hp)+" "+repr(time_simple_update)+" "+repr(time_full_update)+" "+repr(time_env)+" "+repr(time_obs)+"\n")
    
if __name__ == "__main__":
    main()
