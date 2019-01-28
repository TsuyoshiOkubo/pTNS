#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <mpi.h>
#include <mptensor/complex.hpp>
#include <mptensor/rsvd.hpp>
#include <mptensor/tensor.hpp>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include <Lattice.hpp>
#include <PEPS_Basics.hpp>
#include <PEPS_Parameters.hpp>
#include <Square_lattice_CTM.hpp>

#ifdef CPP11
#include <random>
#elif BOOST
#include <boost/random.hpp>
#elif DSFMT
#include <dSFMT.h>
#endif

using namespace mptensor;
#ifdef REAL_TENSOR
typedef double tensor_value_type;
#else
typedef complex tensor_value_type;
#endif
typedef Tensor<scalapack::Matrix, tensor_value_type> ptensor;
typedef Tensor<lapack::Matrix, tensor_value_type> tensor;

/* for MPI */
int mpirank;
int mpisize;

class Lattice_two_sub : public Lattice_skew {
public:
  std::vector<int> A_sub_list, B_sub_list;
  Lattice_two_sub() : Lattice_skew() {
    A_sub_list.resize(N_UNIT / 2);
    B_sub_list.resize(N_UNIT / 2);
  }
  void set_lattice_info() {
    // Lattice setting
    int a_num = 0;
    int b_num = 0;

    A_sub_list.resize(N_UNIT / 2);
    B_sub_list.resize(N_UNIT / 2);

    Tensor_position =
        std::vector<std::vector<int> >(N_UNIT, std::vector<int>(2));
    Tensor_list = std::vector<std::vector<int> >(LX, std::vector<int>(LY));
    NN_Tensor = std::vector<std::vector<int> >(N_UNIT, std::vector<int>(4));

    int num;
    for (int ix = 0; ix < LX_ori; ++ix) {
      for (int iy = 0; iy < LY_ori; ++iy) {
        num = iy * LX_ori + ix;
        Tensor_list[ix][iy] = num;
        Tensor_position[num][0] = ix;
        Tensor_position[num][1] = iy;
        if ((ix + iy) % 2 == 0) {
          A_sub_list[a_num] = num;
          a_num += 1;
        } else {
          B_sub_list[b_num] = num;
          b_num += 1;
        }
      }
    };
    // extend for larger "periodic" unit cell
    if (LX > LX_ori) {
      // assuming LY_ori = LY
      for (int ix = LX_ori; ix < LX; ++ix) {
        int slide = LX_diff * (ix / LX_ori);
        int ix_ori = ix % LX_ori;
        for (int iy = 0; iy < LY_ori; iy++) {
          int iy_ori = (iy - slide + LY_ori) % LY_ori;
          int num = Tensor_list[ix_ori][iy_ori];
          Tensor_list[ix][iy] = num;
        }
      }
    } else if (LY > LY_ori) {
      // assuming LX_ori = LX
      for (int iy = LY_ori; iy < LY; iy++) {
        int slide = LY_diff * (iy / LY_ori);
        int iy_ori = iy % LY_ori;
        for (int ix = 0; ix < LX_ori; ix++) {
          int ix_ori = (ix - slide + LX_ori) % LX_ori;
          int num = Tensor_list[ix_ori][iy_ori];
          Tensor_list[ix][iy] = num;
        }
      }
    }
    // else LX = LX_ori, LY = LY_ori

    int ix, iy;
    for (int i = 0; i < N_UNIT; ++i) {
      ix = i % LX_ori;
      iy = i / LX_ori;

      NN_Tensor[i][0] = Tensor_list[(ix - 1 + LX) % LX][iy];
      NN_Tensor[i][1] = Tensor_list[ix][(iy + 1) % LY];
      NN_Tensor[i][2] = Tensor_list[(ix + 1) % LX][iy];
      NN_Tensor[i][3] = Tensor_list[ix][(iy - 1 + LY) % LY];
    }
  }
};

class Local_parameters {
public:
  int random_seed_global;
  int random_seed;

  double Dz, Dp, hx, hz;
  bool Calc_all_2body;
  //
  bool Read_Initial;
  int Initial_type;
  double Random_amp;

  //
  bool second_ST;
  double tau;
  int tau_step;
  double tau_full;
  int tau_full_step;

  // environment
  bool Env_calc_before_full;
  bool Env_calc_before_obs;
  bool Obs_calc_mag;
  bool Obs_calc_energy;

  // file
  bool Output_file_append;
  Local_parameters() {
    random_seed_global = 13;

    Dz = 0.0;
    Dp = 0.0;
    hx = 0.0;
    hz = 0.0;
    Calc_all_2body = false;

    Read_Initial = false;
    Initial_type = 0;
    Random_amp = 0.02;

    second_ST = false;
    tau = 0.01;
    tau_step = 1000;

    tau_full = 0.01;
    tau_full_step = 0;

    // environment
    Env_calc_before_full = true;
    Env_calc_before_obs = true;
    Obs_calc_mag = true;
    Obs_calc_energy = true;

    // file
    Output_file_append = false;
  }

  void read_parameters(const char *filename) {
    std::ifstream input_file;
    input_file.open(filename, std::ios::in);
    std::string reading_line_buffer;

    while (!input_file.eof()) {
      std::getline(input_file, reading_line_buffer);
      // std::cout << reading_line_buffer << std::endl;
      std::stringstream buf(reading_line_buffer);
      std::vector<std::string> result;
      while (buf >> reading_line_buffer) {
        result.push_back(reading_line_buffer);
      }

      if (result.size() > 1) {
        if (result[0].compare("random_seed_global") == 0) {
          std::istringstream is(result[1]);
          is >> random_seed_global;
        } else if (result[0].compare("Dz") == 0) {
          std::istringstream is(result[1]);
          is >> Dz;
        } else if (result[0].compare("Dp") == 0) {
          std::istringstream is(result[1]);
          is >> Dp;
        } else if (result[0].compare("hx") == 0) {
          std::istringstream is(result[1]);
          is >> hx;
        } else if (result[0].compare("hz") == 0) {
          std::istringstream is(result[1]);
          is >> hz;
        } else if (result[0].compare("Calc_all_2body") == 0) {
          std::istringstream is(result[1]);
          is >> Calc_all_2body;
        } else if (result[0].compare("Read_Initial") == 0) {
          std::istringstream is(result[1]);
          is >> Read_Initial;
        } else if (result[0].compare("Initial_type") == 0) {
          std::istringstream is(result[1]);
          is >> Initial_type;
        } else if (result[0].compare("Random_amp") == 0) {
          std::istringstream is(result[1]);
          is >> Random_amp;
        } else if (result[0].compare("second_ST") == 0) {
          std::istringstream is(result[1]);
          is >> second_ST;
        } else if (result[0].compare("tau") == 0) {
          std::istringstream is(result[1]);
          is >> tau;
        } else if (result[0].compare("tau_step") == 0) {
          std::istringstream is(result[1]);
          is >> tau_step;
        } else if (result[0].compare("tau_full") == 0) {
          std::istringstream is(result[1]);
          is >> tau_full;
        } else if (result[0].compare("tau_full_step") == 0) {
          std::istringstream is(result[1]);
          is >> tau_full_step;
        } else if (result[0].compare("Env_calc_before_full") == 0) {
          std::istringstream is(result[1]);
          is >> Env_calc_before_full;
        } else if (result[0].compare("Env_calc_before_obs") == 0) {
          std::istringstream is(result[1]);
          is >> Env_calc_before_obs;
        } else if (result[0].compare("Obs_calc_mag") == 0) {
          std::istringstream is(result[1]);
          is >> Obs_calc_mag;
        } else if (result[0].compare("Obs_calc_energy") == 0) {
          std::istringstream is(result[1]);
          is >> Obs_calc_energy;
        } else if (result[0].compare("Output_file_append") == 0) {
          std::istringstream is(result[1]);
          is >> Output_file_append;
        }
        // std::cout<< "## input data: "<<result[0]<<" =
        // "<<result[1]<<std::endl;
      }
    }
  };

  void output_parameters(const char *filename, bool append) {
    std::ofstream ofs;
    if (append) {
      ofs.open(filename, std::ios::out | std::ios::app);
    } else {
      ofs.open(filename, std::ios::out);
    }
    // Tensor
    ofs << "random_seed_global " << random_seed_global << std::endl;

    ofs << "Dz " << Dz << std::endl;
    ofs << "Dp " << Dp << std::endl;
    ofs << "hx " << hx << std::endl;
    ofs << "hz " << hz << std::endl;
    ofs << "Calc_all_2body " << Calc_all_2body << std::endl;

    ofs << "Read_Initial " << Read_Initial << std::endl;
    ofs << "Initial_type " << Initial_type << std::endl;
    ofs << "Random_amp " << Random_amp << std::endl;

    ofs << "second_ST " << second_ST << std::endl;
    ofs << "tau " << tau << std::endl;
    ofs << "tau_step " << tau_step << std::endl;
    ofs << "tau_full " << tau_full << std::endl;
    ofs << "tau_full_step " << tau_full_step << std::endl;

    ofs << "Env_calc_before_full " << Env_calc_before_full << std::endl;
    ofs << "Env_calc_before_obs " << Env_calc_before_obs << std::endl;
    ofs << "Obs_calc_mag " << Obs_calc_mag << std::endl;
    ofs << "Obs_calc_energy " << Obs_calc_energy << std::endl;

    ofs << "Output_file_append " << Output_file_append << std::endl;

    ofs.close();
  }
  void output_parameters(const char *filename) {
    output_parameters(filename, false);
  }
  void output_parameters_append(const char *filename) {
    output_parameters(filename, true);
  }

  void Bcast_parameters(MPI_Comm comm) {
    int irank;
    MPI_Comm_rank(MPI_COMM_WORLD, &irank);

    std::vector<double> params_double(7);
    std::vector<int> params_int(12);

    if (irank == 0) {
      params_int[0] = random_seed_global;
      params_int[1] = Read_Initial;
      params_int[2] = Initial_type;
      params_int[3] = second_ST;
      params_int[4] = tau_step;
      params_int[5] = tau_full_step;
      params_int[6] = Env_calc_before_full;
      params_int[7] = Env_calc_before_obs;
      params_int[8] = Obs_calc_mag;
      params_int[9] = Obs_calc_energy;
      params_int[10] = Output_file_append;
      params_int[11] = Calc_all_2body;

      params_double[0] = Dz;
      params_double[1] = Dp;
      params_double[2] = hx;
      params_double[3] = hz;
      params_double[4] = Random_amp;
      params_double[5] = tau;
      params_double[6] = tau_full;

      MPI_Bcast(&params_int.front(), 12, MPI_INT, 0, comm);
      MPI_Bcast(&params_double.front(), 7, MPI_DOUBLE, 0, comm);
    } else {
      MPI_Bcast(&params_int.front(), 12, MPI_INT, 0, comm);
      MPI_Bcast(&params_double.front(), 7, MPI_DOUBLE, 0, comm);

      random_seed_global = params_int[0];
      Read_Initial = params_int[1];
      Initial_type = params_int[2];
      second_ST = params_int[3];
      tau_step = params_int[4];
      tau_full_step = params_int[5];

      Dz = params_double[0];
      Dp = params_double[1];
      hx = params_double[2];
      hz = params_double[3];

      Calc_all_2body = params_int[11];

      Random_amp = params_double[4];
      tau = params_double[5];
      tau_full = params_double[6];

      Env_calc_before_full = params_int[6];
      Env_calc_before_obs = params_int[7];
      Obs_calc_mag = params_int[8];
      Obs_calc_energy = params_int[9];

      Output_file_append = params_int[10];
    }
  };
};

// functions for Set_Hamiltonian

ptensor kron(const ptensor &A, const ptensor &B) {
  assert(A.shape().size() == B.shape().size());
  int AB_size = A.shape().size();
  Shape shape_A = A.shape();
  Shape shape_B = B.shape();
  shape_A.push(1);
  shape_B.push(1);

  Axes tans_C;
  Shape shape_C;
  for (int i = 0; i < AB_size; i++) {
    tans_C.push(i);
    tans_C.push(i + AB_size);
    shape_C.push(shape_A[i] * shape_B[i]);
  }

  return reshape(tensordot(reshape(A, shape_A), reshape(B, shape_B),
                           Axes(AB_size), Axes(AB_size))
                     .transpose(tans_C),
                 shape_C);
}

ptensor Ham_ops(const std::vector<ptensor> &ops) {
  assert(ops.size() >= 6);
  return kron(kron(kron(kron(kron(ops[0], ops[1]), ops[2]), ops[3]), ops[4]),
              ops[5]);
}

std::vector<ptensor> Create_one_site_ops(const ptensor &op, const ptensor &I,
                                         const int ip) {
  std::vector<ptensor> ops;
  for (int i = 0; i < 6; i++) {
    if (i == ip) {
      ops.push_back(op);
    } else {
      ops.push_back(I);
    }
  }
  return ops;
}
std::vector<ptensor> Create_two_site_ops(const ptensor &op_i,
                                         const ptensor &op_j, const ptensor &I,
                                         const int ip, const int jp) {
  std::vector<ptensor> ops;
  for (int i = 0; i < 6; i++) {
    if (i == ip) {
      ops.push_back(op_i);
    } else if (i == jp) {
      ops.push_back(op_j);
    } else {
      ops.push_back(I);
    }
  }
  return ops;
}

ptensor Heisenberg(const ptensor &Sz, const ptensor &Sp, const ptensor &Sm,
                   const ptensor &I, const int ip, const int jp) {
  std::vector<ptensor> ops_Sz = Create_two_site_ops(Sz, Sz, I, ip, jp);
  std::vector<ptensor> ops_Spm = Create_two_site_ops(Sp, Sm, I, ip, jp);
  std::vector<ptensor> ops_Smp = Create_two_site_ops(Sm, Sp, I, ip, jp);

  return Ham_ops(ops_Sz) + 0.5 * (Ham_ops(ops_Spm) + Ham_ops(ops_Smp));
}

ptensor hx_field(const ptensor &Sp, const ptensor &Sm, const ptensor &I,
                 const int ip, const int jp) {
  std::vector<ptensor> ops_Spi = Create_one_site_ops(Sp, I, ip);
  std::vector<ptensor> ops_Spj = Create_one_site_ops(Sp, I, jp);
  std::vector<ptensor> ops_Smi = Create_one_site_ops(Sm, I, ip);
  std::vector<ptensor> ops_Smj = Create_one_site_ops(Sm, I, jp);

  return -0.125 * (Ham_ops(ops_Spi) + Ham_ops(ops_Spj) + Ham_ops(ops_Smi) +
                   Ham_ops(ops_Smj));
}

ptensor hz_field(const ptensor &Sz, const ptensor &I, const int ip,
                 const int jp) {
  std::vector<ptensor> ops_Szi = Create_one_site_ops(Sz, I, ip);
  std::vector<ptensor> ops_Szj = Create_one_site_ops(Sz, I, jp);

  return -0.25 * (Ham_ops(ops_Szi) + Ham_ops(ops_Szj));
}
#ifndef REAL_TENSOR
ptensor DM_Dz(const ptensor &Sp, const ptensor &Sm, const ptensor &I,
              const int ip, const int jp) {
  std::vector<ptensor> ops_Spm = Create_two_site_ops(Sp, Sm, I, ip, jp);
  std::vector<ptensor> ops_Smp = Create_two_site_ops(Sm, Sp, I, ip, jp);

  return std::complex<double>(0.0, 0.5) * (Ham_ops(ops_Spm) - Ham_ops(ops_Smp));
}
ptensor DM_Dp(const ptensor &Sz, const ptensor &Sp, const ptensor &Sm,
              const ptensor &I, const int ip, const int jp,
              const double theta) {
  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);

  std::complex<double> i_e_pi_theta(-sin_theta, cos_theta);
  std::complex<double> i_e_mi_theta(sin_theta, cos_theta);

  std::vector<ptensor> ops_Spz = Create_two_site_ops(Sp, Sz, I, ip, jp);
  std::vector<ptensor> ops_Smz = Create_two_site_ops(Sm, Sz, I, ip, jp);
  std::vector<ptensor> ops_Szp = Create_two_site_ops(Sz, Sp, I, ip, jp);
  std::vector<ptensor> ops_Szm = Create_two_site_ops(Sz, Sm, I, ip, jp);

  return 0.5 * (i_e_mi_theta * (-Ham_ops(ops_Spz) + Ham_ops(ops_Szp)) +
                i_e_pi_theta * (Ham_ops(ops_Smz) - Ham_ops(ops_Szm)));
}
#endif
ptensor Create_Hij(const ptensor &Sz, const ptensor &Sp, const ptensor &Sm,
                   const ptensor &I, const int ip, const int jp,
                   const double hx, const double hz, const double Dp,
                   const double Dz, const double theta) {
#ifdef REAL_TENSOR
  // ignore DM interactions
  return Heisenberg(Sz, Sp, Sm, I, ip, jp) + hx * hx_field(Sp, Sm, I, ip, jp) +
         hz * hz_field(Sz, I, ip, jp);
#else
  return Heisenberg(Sz, Sp, Sm, I, ip, jp) + hx * hx_field(Sp, Sm, I, ip, jp) +
         hz * hz_field(Sz, I, ip, jp) +
         Dp * DM_Dp(Sz, Sp, Sm, I, ip, jp, theta) +
         Dz * DM_Dz(Sp, Sm, I, ip, jp);
#endif
}

void Set_Hamiltonian(ptensor &Ham_ABx, ptensor &Ham_ABy, ptensor &Ham_BAx,
                     ptensor &Ham_BAy, const double hx, const double hz,
                     const double Dp, const double Dz) {
  /* メモ：mptensor の初期値はzero */
  ptensor Sz(Shape(2, 2)), Sp(Shape(2, 2)), Sm(Shape(2, 2)), I(Shape(2, 2));
  Sz.set_value(Index(0, 0), 0.5);
  Sz.set_value(Index(1, 1), -0.5);
  Sp.set_value(Index(0, 1), 1.0);
  Sm.set_value(Index(1, 0), 1.0);
  I.set_value(Index(0, 0), 1.0);
  I.set_value(Index(1, 1), 1.0);

  // H_AA_i = A1*A2 + A2*A3 =
  ptensor Ham_AA_i =
      Create_Hij(Sz, Sp, Sm, I, 0, 1, hx, hz, Dp, Dz, 7.0 * M_PI / 6.0) +
      Create_Hij(Sz, Sp, Sm, I, 1, 2, hx, hz, Dp, -Dz, M_PI / 6.0);
  // HH_AA_j = A1*A2 + A2*A3
  ptensor Ham_AA_j =
      Create_Hij(Sz, Sp, Sm, I, 3, 4, hx, hz, Dp, Dz, 7.0 * M_PI / 6.0) +
      Create_Hij(Sz, Sp, Sm, I, 4, 5, hx, hz, Dp, -Dz, M_PI / 6.0);

  // H_BB_i = B1*B2 + B2*B3 =
  ptensor Ham_BB_i =
      Create_Hij(Sz, Sp, Sm, I, 0, 1, hx, hz, Dp, -Dz, 5.0 * M_PI / 6.0) +
      Create_Hij(Sz, Sp, Sm, I, 1, 2, hx, hz, Dp, Dz, 11.0 * M_PI / 6.0);
  // HH_BB_j = B1*B2 + B2*B3
  ptensor Ham_BB_j =
      Create_Hij(Sz, Sp, Sm, I, 3, 4, hx, hz, Dp, -Dz, 5.0 * M_PI / 6.0) +
      Create_Hij(Sz, Sp, Sm, I, 4, 5, hx, hz, Dp, Dz, 11.0 * M_PI / 6.0);

  //H_ABx = A2B3 + A1B3 + 0.25 * (H_AA_i + H_BB_j)
  Ham_ABx = Create_Hij(Sz, Sp, Sm, I, 1, 5, hx, hz, Dp, Dz, 11.0 * M_PI / 6.0) +
            Create_Hij(Sz, Sp, Sm, I, 0, 5, hx, hz, Dp, -Dz, 3.0 * M_PI / 2.0) +
            0.25 * (Ham_AA_i + Ham_BB_j);

  //H_ABy = A1B2 + A1B3 + 0.25 * (H_AA_i + H_BB_j)
  Ham_ABy = Create_Hij(Sz, Sp, Sm, I, 0, 4, hx, hz, Dp, Dz, 7.0 * M_PI / 6.0) +
            Create_Hij(Sz, Sp, Sm, I, 0, 5, hx, hz, Dp, -Dz, 3.0 * M_PI / 2.0) +
            0.25 * (Ham_AA_i + Ham_BB_j);

  //H_BAx = B1A2 + B1A3 + 0.25 * (H_BB_i + H_AA_j)
  Ham_BAx = Create_Hij(Sz, Sp, Sm, I, 0, 4, hx, hz, Dp, -Dz, 5.0 * M_PI / 6.0) +
            Create_Hij(Sz, Sp, Sm, I, 0, 5, hx, hz, Dp, Dz, M_PI / 2.0) +
            0.25 * (Ham_BB_i + Ham_AA_j);

  //H_BAy = B2A3 + B1A3 + 0.25 * (H11 + H22)
  Ham_BAy = Create_Hij(Sz, Sp, Sm, I, 1, 5, hx, hz, Dp, -Dz, M_PI / 6.0) +
            Create_Hij(Sz, Sp, Sm, I, 0, 5, hx, hz, Dp, Dz, M_PI / 2.0) +
            0.25 * (Ham_BB_i + Ham_AA_j);
  return;
}

void Initialize_Tensors(std::vector<ptensor> &Tn,
                        const Local_parameters local_parameters,
                        const Lattice_two_sub lattice) {
  int D = Tn[0].shape()[0];

// Random tensors
#ifdef CPP11
  std::mt19937 gen(local_parameters.random_seed);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
#elif BOOST
  boost::mt19937 gen(local_parameters.random_seed);
  boost::uniform_real<double> dist(-0.5, 0.5);
#elif DSFMT
  dsfmt_t dsfmt;
  dsfmt_init_gen_rand(&dsfmt, local_parameters.random_seed);
#endif

  Index index;
  tensor_value_type values;
  double real_part, imag_part;

  if (local_parameters.Initial_type == 1) {
    // q=0 up,up,down
    for (int i = 0; i < lattice.N_UNIT; ++i) {
      for (int n = 0; n < Tn[i].local_size(); ++n) {
        index = Tn[i].global_index(n);
        // data is stored in "fortran order"
        if (index == Index(0, 0, 0, 0, 4)) {
          Tn[i].set_value(index, 1.0);
        } else {
#ifdef DSFMT
          real_part = local_parameters.Random_amp *
                      (dsfmt_genrand_close_open(&dsfmt) - 0.5);
          imag_part = local_parameters.Random_amp *
                      (dsfmt_genrand_close_open(&dsfmt) - 0.5);
#else
          real_part = local_parameters.Random_amp * dist(gen);
          imag_part = local_parameters.Random_amp * dist(gen);
#endif

#ifdef REAL_TENSOR
          values = real_part;
#else
          values = std::complex<double>(real_part, imag_part);
#endif
          Tn[i].set_value(index, values);
        }
      }
    }
  } else {
    for (int i = 0; i < lattice.N_UNIT; ++i) {
      for (int n = 0; n < Tn[i].local_size(); ++n) {
        index = Tn[i].global_index(n);
#ifdef DSFMT
        real_part = local_parameters.Random_amp *
                    (dsfmt_genrand_close_open(&dsfmt) - 0.5);
        imag_part = local_parameters.Random_amp *
                    (dsfmt_genrand_close_open(&dsfmt) - 0.5);
#else
        real_part = local_parameters.Random_amp * dist(gen);
        imag_part = local_parameters.Random_amp * dist(gen);
#endif
#ifdef REAL_TENSOR
        values = real_part;
#else
        values = std::complex<double>(real_part, imag_part);
#endif
        Tn[i].set_value(index, values);
      }
    }
  }
}

// for physicsl quantities
ptensor Create_one_op(const ptensor &op1, const ptensor &op2,
                      const ptensor &op3) {
  return kron(kron(op1, op2), op3);
}

double get_real_part(const std::complex<double> &value) { return value.real(); }
double get_real_part(const double value) { return value; }

double get_imag_part(const std::complex<double> &value) { return value.imag(); }
double get_imag_part(const double value) { return 0.0; }

int main(int argc, char **argv) {

  /* MPI initialization */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  double time_simple_update = 0.0;
  double time_full_update = 0.0;
  double time_env = 0.0;
  double time_obs = 0.0;
  double start_time;

  std::cout << std::setprecision(16);

  // Parameters
  PEPS_Parameters peps_parameters;
  Lattice_two_sub lattice;
  Local_parameters local_parameters;
  int random_seed;

  if (mpirank == 0) {
    local_parameters.read_parameters("input.dat");
    peps_parameters.read_parameters("input.dat");
    lattice.read_parameters("input.dat");
    lattice.N_UNIT = lattice.LX_ori * lattice.LY_ori;
#ifdef REAL_TENSOR
    // ignore DM interactions
    local_parameters.Dz = 0.0;
    local_parameters.Dp = 0.0;
#endif
  }

  local_parameters.Bcast_parameters(MPI_COMM_WORLD);
  peps_parameters.Bcast_parameters(MPI_COMM_WORLD);
  lattice.Bcast_parameters(MPI_COMM_WORLD);

  lattice.set_lattice_info();

  // output debug or warning info only from process 0
  if (mpirank != 0) {
    peps_parameters.Debug_flag = false;
    peps_parameters.Warning_flag = false;
  }

  if (mpirank == 0) {
    // folder check
    struct stat status;
    if (stat("output_data", &status) != 0) {
      mkdir("output_data", 0755);
    }

    if (local_parameters.Output_file_append) {
      peps_parameters.output_parameters_append("output_data/output_params.dat");
    } else {
      peps_parameters.output_parameters("output_data/output_params.dat");
    }
    lattice.output_parameters_append("output_data/output_params.dat");
    local_parameters.output_parameters_append("output_data/output_params.dat");
    std::ofstream ofs;
    ofs.open("output_data/output_params.dat", std::ios::out | std::ios::app);
    ofs << std::endl;
    ofs.close();
  }

  // for convenience//
  int D = peps_parameters.D;
  int CHI = peps_parameters.CHI;

  int LX = lattice.LX;
  int LY = lattice.LY;
  int LX_ori = lattice.LX_ori;
  int LY_ori = lattice.LY_ori;
  int N_UNIT = lattice.N_UNIT;

  double hx = local_parameters.hx;
  double hz = local_parameters.hz;
  double Dp = local_parameters.Dp;
  double Dz = local_parameters.Dz;

#ifdef CPP11
  std::mt19937 gen(local_parameters.random_seed_global);
  std::uniform_int_distribution<> dist(0, INT_MAX);
#elif BOOST
  boost::mt19937 gen(local_parameters.random_seed_global);
  boost::uniform_int<> dist(0, INT_MAX);
#elif DSFMT
  dsfmt_t dsfmt;
  dsfmt_init_gen_rand(&dsfmt, local_parameters.random_seed_global);
#endif
  std::vector<int> seed_dist(mpisize);
  if (mpirank == 0) {
    for (int i = 0; i < mpisize; ++i) {
#ifdef DSFMT
      seed_dist[i] = dsfmt_genrand_uint32(&dsfmt);
#else
      seed_dist[i] = dist(gen);
#endif
    }
    MPI_Scatter(&seed_dist.front(), 1, MPI_INT, &random_seed, 1, MPI_INT, 0,
                MPI_COMM_WORLD);
  } else {
    MPI_Scatter(&seed_dist.front(), 1, MPI_INT, &random_seed, 1, MPI_INT, 0,
                MPI_COMM_WORLD);
  }
  local_parameters.random_seed = random_seed;

  // set seed for randomized svd
  random_tensor::set_seed(local_parameters.random_seed);

  // Tensors
  std::vector<ptensor> Tn(N_UNIT, ptensor(Shape(D, D, D, D, 8)));
  std::vector<ptensor> eTt(N_UNIT, ptensor(Shape(CHI, CHI, D, D))),
      eTr(N_UNIT, ptensor(Shape(CHI, CHI, D, D))),
      eTb(N_UNIT, ptensor(Shape(CHI, CHI, D, D))),
      eTl(N_UNIT, ptensor(Shape(CHI, CHI, D, D)));
  std::vector<ptensor> C1(N_UNIT, ptensor(Shape(CHI, CHI))),
      C2(N_UNIT, ptensor(Shape(CHI, CHI))),
      C3(N_UNIT, ptensor(Shape(CHI, CHI))),
      C4(N_UNIT, ptensor(Shape(CHI, CHI)));

  std::vector<std::vector<std::vector<double> > > lambda_tensor(
      N_UNIT, std::vector<std::vector<double> >(4, std::vector<double>(D)));

  if (!local_parameters.Read_Initial) {
    Initialize_Tensors(Tn, local_parameters, lattice);

    for (int i1 = 0; i1 < N_UNIT; ++i1) {
      for (int i2 = 0; i2 < 4; ++i2) {
        for (int i3 = 0; i3 < D; ++i3) {
          lambda_tensor[i1][i2][i3] = 1.0;
        }
      }
    }
  } else {
    for (int i = 0; i < N_UNIT; i++) {
      std::stringstream ss;
      ss << i;
      std::string filename;

      filename = "input_tensors/Tn" + ss.str();
      Tn[i].load(filename.c_str());

      filename = "input_tensors/C1Mat" + ss.str();
      C1[i].load(filename.c_str());
      filename = "input_tensors/C2Mat" + ss.str();
      C2[i].load(filename.c_str());
      filename = "input_tensors/C3Mat" + ss.str();
      C3[i].load(filename.c_str());
      filename = "input_tensors/C4Mat" + ss.str();
      C4[i].load(filename.c_str());

      filename = "input_tensors/eTt" + ss.str();
      eTt[i].load(filename.c_str());
      filename = "input_tensors/eTr" + ss.str();
      eTr[i].load(filename.c_str());
      filename = "input_tensors/eTb" + ss.str();
      eTb[i].load(filename.c_str());
      filename = "input_tensors/eTl" + ss.str();
      eTl[i].load(filename.c_str());
    }
    // extention
    int D_read = Tn[0].shape()[0];
    int CHI_read = C1[0].shape()[0];
    bool D_extend_flag = false;
    bool CHI_extend_flag = false;
    if (D_read < D)
      D_extend_flag = true;
    if (CHI_read < CHI)
      CHI_extend_flag = true;

    if (D_extend_flag) {
      for (int i = 0; i < N_UNIT; i++) {
        Tn[i] = extend(Tn[i], Shape(D, D, D, D, 3));
      }
    }
    if (CHI_extend_flag) {
      for (int i = 0; i < N_UNIT; i++) {
        C1[i] = extend(C1[i], Shape(CHI, CHI));
        C2[i] = extend(C2[i], Shape(CHI, CHI));
        C3[i] = extend(C3[i], Shape(CHI, CHI));
        C4[i] = extend(C4[i], Shape(CHI, CHI));
      }
    }

    if (D_extend_flag || CHI_extend_flag) {
      for (int i = 0; i < N_UNIT; i++) {
        eTt[i] = extend(eTt[i], Shape(CHI, CHI, D, D));
        eTr[i] = extend(eTr[i], Shape(CHI, CHI, D, D));
        eTb[i] = extend(eTb[i], Shape(CHI, CHI, D, D));
        eTl[i] = extend(eTl[i], Shape(CHI, CHI, D, D));
      }
    }

    std::vector<double> lambda_load(N_UNIT * 4 * D_read);
    if (mpirank == 0) {
      std::string filename = "input_tensors/lambdas";
      std::ifstream fin(filename.c_str(), std::ifstream::binary);
      fin.read(reinterpret_cast<char *>(&lambda_load[0]),
               sizeof(double) * N_UNIT * 4 * D_read);
      fin.close();

      MPI_Bcast(&lambda_load.front(), N_UNIT * 4 * D_read, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    } else {
      MPI_Bcast(&lambda_load.front(), N_UNIT * 4 * D_read, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    }
    int num;
    for (int i1 = 0; i1 < N_UNIT; ++i1) {
      for (int i2 = 0; i2 < 4; ++i2) {
        for (int i3 = 0; i3 < D_read; ++i3) {
          num = i1 * 4 * D_read + i2 * D_read + i3;
          lambda_tensor[i1][i2][i3] = lambda_load[num];
        }
      }
    }
  }

  ptensor Ham_ABx, Ham_ABy, Ham_BAx, Ham_BAy;
  Set_Hamiltonian(Ham_ABx, Ham_ABy, Ham_BAx, Ham_BAy, hx, hz, Dp, Dz);
  ptensor U_ABx, Ud_ABx, Us_ABx, U_ABy, Ud_ABy, Us_ABy;
  ptensor U_BAx, Ud_BAx, Us_BAx, U_BAy, Ud_BAy, Us_BAy;
  std::vector<double> s_ABx, s_ABy, s_BAx, s_BAy;

  int info = eigh(Ham_ABx, s_ABx, U_ABx);
  std::vector<double> exp_s(64);

  Ud_ABx = conj(U_ABx);
  for (int i = 0; i < 64; ++i) {
    exp_s[i] = exp(-local_parameters.tau * s_ABx[i]);
  }
  Us_ABx = U_ABx;
  Us_ABx.multiply_vector(exp_s, 1);

  ptensor opAB_x =
      reshape(tensordot(Us_ABx, Ud_ABx, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
          .transpose(Axes(2, 3, 0, 1));

  for (int i = 0; i < 64; ++i) {
    exp_s[i] = exp(-0.5 * local_parameters.tau * s_ABx[i]);
  }
  Us_ABx = U_ABx;
  Us_ABx.multiply_vector(exp_s, 1);

  ptensor opAB_x_2 =
      reshape(tensordot(Us_ABx, Ud_ABx, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
          .transpose(Axes(2, 3, 0, 1));

  info = eigh(Ham_ABy, s_ABy, U_ABy);
  Ud_ABy = conj(U_ABy);
  for (int i = 0; i < 64; ++i) {
    exp_s[i] = exp(-local_parameters.tau * s_ABy[i]);
  }
  Us_ABy = U_ABy;
  Us_ABy.multiply_vector(exp_s, 1);

  ptensor opAB_y =
      reshape(tensordot(Us_ABy, Ud_ABy, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
          .transpose(Axes(2, 3, 0, 1));

  for (int i = 0; i < 64; ++i) {
    exp_s[i] = exp(-0.5 * local_parameters.tau * s_ABy[i]);
  }
  Us_ABy = U_ABy;
  Us_ABy.multiply_vector(exp_s, 1);

  ptensor opAB_y_2 =
      reshape(tensordot(Us_ABy, Ud_ABy, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
          .transpose(Axes(2, 3, 0, 1));

  info = eigh(Ham_BAx, s_BAx, U_BAx);
  Ud_BAx = conj(U_BAx);
  for (int i = 0; i < 64; ++i) {
    exp_s[i] = exp(-local_parameters.tau * s_BAx[i]);
  }
  Us_BAx = U_BAx;
  Us_BAx.multiply_vector(exp_s, 1);

  ptensor opBA_x =
      reshape(tensordot(Us_BAx, Ud_BAx, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
          .transpose(Axes(2, 3, 0, 1));

  for (int i = 0; i < 64; ++i) {
    exp_s[i] = exp(-0.5 * local_parameters.tau * s_BAx[i]);
  }
  Us_BAx = U_BAx;
  Us_BAx.multiply_vector(exp_s, 1);

  ptensor opBA_x_2 =
      reshape(tensordot(Us_BAx, Ud_BAx, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
          .transpose(Axes(2, 3, 0, 1));

  info = eigh(Ham_BAy, s_BAy, U_BAy);
  Ud_BAy = conj(U_BAy);
  for (int i = 0; i < 64; ++i) {
    exp_s[i] = exp(-local_parameters.tau * s_BAy[i]);
  }
  Us_BAy = U_BAy;
  Us_BAy.multiply_vector(exp_s, 1);

  ptensor opBA_y =
      reshape(tensordot(Us_BAy, Ud_BAy, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
          .transpose(Axes(2, 3, 0, 1));

  for (int i = 0; i < 64; ++i) {
    exp_s[i] = exp(-0.5 * local_parameters.tau * s_BAy[i]);
  }
  Us_BAy = U_BAy;
  Us_BAy.multiply_vector(exp_s, 1);

  ptensor opBA_y_2 =
      reshape(tensordot(Us_BAy, Ud_BAy, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
          .transpose(Axes(2, 3, 0, 1));

  ptensor Tn1_new, Tn2_new;
  std::vector<double> lambda_c;

  start_time = MPI_Wtime();
  for (int int_tau = 0; int_tau < local_parameters.tau_step; ++int_tau) {
    int num, num_j;
    // simple update
    if (local_parameters.second_ST) {
      // x-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opAB_x_2, 2, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][2] = lambda_c;
        lambda_tensor[num_j][0] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // x-bond B sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opBA_x_2, 2, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][2] = lambda_c;
        lambda_tensor[num_j][0] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }
      // y-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opAB_y_2, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // y-bond B sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opBA_y, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // y-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opAB_y_2, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // x-bond B sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opBA_x_2, 2, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][2] = lambda_c;
        lambda_tensor[num_j][0] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // x-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opAB_x_2, 2, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][2] = lambda_c;
        lambda_tensor[num_j][0] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

    } else {
      // x-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opAB_x, 2, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][2] = lambda_c;
        lambda_tensor[num_j][0] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // x-bond B sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opBA_x, 2, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][2] = lambda_c;
        lambda_tensor[num_j][0] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }
      // y-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opAB_y, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // y-bond B sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], opBA_y, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }
    }
  }
  time_simple_update += MPI_Wtime() - start_time;
  // done simple update

  // Start full update

  int count_CTM_env;
  bool file_count_CTM_exist = false;
  if (local_parameters.tau_full_step > 0) {

    for (int i = 0; i < 64; ++i) {
      exp_s[i] = exp(-local_parameters.tau_full * s_ABx[i]);
    }
    Us_ABx = U_ABx;
    Us_ABx.multiply_vector(exp_s, 1);

    opAB_x =
        reshape(tensordot(Us_ABx, Ud_ABx, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
            .transpose(Axes(2, 3, 0, 1));

    for (int i = 0; i < 64; ++i) {
      exp_s[i] = exp(-0.5 * local_parameters.tau_full * s_ABx[i]);
    }
    Us_ABx = U_ABx;
    Us_ABx.multiply_vector(exp_s, 1);

    opAB_x_2 =
        reshape(tensordot(Us_ABx, Ud_ABx, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
            .transpose(Axes(2, 3, 0, 1));

    for (int i = 0; i < 64; ++i) {
      exp_s[i] = exp(-local_parameters.tau_full * s_ABy[i]);
    }
    Us_ABy = U_ABy;
    Us_ABy.multiply_vector(exp_s, 1);

    opAB_y =
        reshape(tensordot(Us_ABy, Ud_ABy, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
            .transpose(Axes(2, 3, 0, 1));

    for (int i = 0; i < 64; ++i) {
      exp_s[i] = exp(-0.5 * local_parameters.tau_full * s_ABy[i]);
    }
    Us_ABy = U_ABy;
    Us_ABy.multiply_vector(exp_s, 1);

    opAB_y_2 =
        reshape(tensordot(Us_ABy, Ud_ABy, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
            .transpose(Axes(2, 3, 0, 1));

    for (int i = 0; i < 64; ++i) {
      exp_s[i] = exp(-local_parameters.tau_full * s_BAx[i]);
    }
    Us_BAx = U_BAx;
    Us_BAx.multiply_vector(exp_s, 1);

    opBA_x =
        reshape(tensordot(Us_BAx, Ud_BAx, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
            .transpose(Axes(2, 3, 0, 1));

    for (int i = 0; i < 64; ++i) {
      exp_s[i] = exp(-0.5 * local_parameters.tau_full * s_BAx[i]);
    }
    Us_BAx = U_BAx;
    Us_BAx.multiply_vector(exp_s, 1);

    opBA_x_2 =
        reshape(tensordot(Us_BAx, Ud_BAx, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
            .transpose(Axes(2, 3, 0, 1));

    for (int i = 0; i < 64; ++i) {
      exp_s[i] = exp(-local_parameters.tau_full * s_BAy[i]);
    }
    Us_BAy = U_BAy;
    Us_BAy.multiply_vector(exp_s, 1);

    opBA_y =
        reshape(tensordot(Us_BAy, Ud_BAy, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
            .transpose(Axes(2, 3, 0, 1));

    for (int i = 0; i < 64; ++i) {
      exp_s[i] = exp(-0.5 * local_parameters.tau_full * s_BAy[i]);
    }
    Us_BAy = U_BAy;
    Us_BAy.multiply_vector(exp_s, 1);

    opBA_y_2 =
        reshape(tensordot(Us_BAy, Ud_BAy, Axes(1), Axes(1)), Shape(8, 8, 8, 8))
            .transpose(Axes(2, 3, 0, 1));

    // Environment
    if (local_parameters.tau_step > 0) {
      start_time = MPI_Wtime();
      count_CTM_env = Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl,
                                           Tn, peps_parameters, lattice, true);
      time_env += MPI_Wtime() - start_time;

      if (mpirank == 0) {
        std::ofstream ofs_env;
        ofs_env << std::setprecision(16);
        if (local_parameters.Output_file_append) {
          ofs_env.open("output_data/CTM_count.dat",
                       std::ios::out | std::ios::app);
        } else {
          ofs_env.open("output_data/CTM_count.dat", std::ios::out);
        }
        ofs_env << "#Befor_full " << hx << " " << hz << " " << Dz << " " << Dp
                << " " << count_CTM_env << std::endl;
        file_count_CTM_exist = true;
        ofs_env.close();
      }

    } else if (local_parameters.Env_calc_before_full) {
      {
        start_time = MPI_Wtime();
        count_CTM_env = Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl,
                                             Tn, peps_parameters, lattice,
                                             !local_parameters.Read_Initial);
        time_env += MPI_Wtime() - start_time;
        if (mpirank == 0) {
          std::ofstream ofs_env;
          ofs_env << std::setprecision(16);
          if (local_parameters.Output_file_append) {
            ofs_env.open("output_data/CTM_count.dat",
                         std::ios::out | std::ios::app);
          } else {
            ofs_env.open("output_data/CTM_count.dat", std::ios::out);
          }
          ofs_env << "#Befor_full " << hx << " " << hz << " " << Dz << " " << Dp
                  << " " << count_CTM_env << std::endl;
          file_count_CTM_exist = true;
          ofs_env.close();
        }
      }
    }
  }

  start_time = MPI_Wtime();
  for (int int_tau = 0; int_tau < local_parameters.tau_full_step; ++int_tau) {
    int num, num_j, ix, ix_j, iy, iy_j;
    if (local_parameters.second_ST) {
      // x-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Full_update_bond(C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                         eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                         Tn[num], Tn[num_j], opAB_x_2, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // ix = num % LX;
          // ix_j = num_j % LX;
          ix = lattice.Tensor_position[num][0];
          ix_j = lattice.Tensor_position[num_j][0];
          Left_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix, peps_parameters,
                    lattice);
          Right_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix_j,
                     peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }

      // x-bond B sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Full_update_bond(C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                         eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                         Tn[num], Tn[num_j], opBA_x_2, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // ix = num % LX;
          // ix_j = num_j % LX;
          ix = lattice.Tensor_position[num][0];
          ix_j = lattice.Tensor_position[num_j][0];
          Left_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix, peps_parameters,
                    lattice);
          Right_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix_j,
                     peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }

      // y-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], opAB_y_2, 1, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // iy = num / LX;
          // iy_j = num_j / LX;
          iy = lattice.Tensor_position[num][1];
          iy_j = lattice.Tensor_position[num_j][1];
          Bottom_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, iy,
                      peps_parameters, lattice);
          Top_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, iy_j,
                   peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }

      // y-bond B sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], opBA_y, 1, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // iy = num / LX;
          // iy_j = num_j / LX;
          iy = lattice.Tensor_position[num][1];
          iy_j = lattice.Tensor_position[num_j][1];
          Bottom_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, iy,
                      peps_parameters, lattice);
          Top_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, iy_j,
                   peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }

      // y-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], opAB_y_2, 1, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // iy = num / LX;
          // iy_j = num_j / LX;
          iy = lattice.Tensor_position[num][1];
          iy_j = lattice.Tensor_position[num_j][1];
          Bottom_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, iy,
                      peps_parameters, lattice);
          Top_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, iy_j,
                   peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }
      // x-bond B sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Full_update_bond(C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                         eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                         Tn[num], Tn[num_j], opBA_x_2, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // ix = num % LX;
          // ix_j = num_j % LX;
          ix = lattice.Tensor_position[num][0];
          ix_j = lattice.Tensor_position[num_j][0];
          Left_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix, peps_parameters,
                    lattice);
          Right_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix_j,
                     peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }
      // x-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Full_update_bond(C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                         eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                         Tn[num], Tn[num_j], opAB_x_2, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // ix = num % LX;
          // ix_j = num_j % LX;
          ix = lattice.Tensor_position[num][0];
          ix_j = lattice.Tensor_position[num_j][0];
          Left_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix, peps_parameters,
                    lattice);
          Right_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix_j,
                     peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }
    } else {
      // x-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Full_update_bond(C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                         eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                         Tn[num], Tn[num_j], opAB_x, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // ix = num % LX;
          // ix_j = num_j % LX;
          ix = lattice.Tensor_position[num][0];
          ix_j = lattice.Tensor_position[num_j][0];
          Left_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix, peps_parameters,
                    lattice);
          Right_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix_j,
                     peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }

      // x-bond B sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Full_update_bond(C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                         eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                         Tn[num], Tn[num_j], opBA_x, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // ix = num % LX;
          // ix_j = num_j % LX;
          ix = lattice.Tensor_position[num][0];
          ix_j = lattice.Tensor_position[num_j][0];
          Left_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix, peps_parameters,
                    lattice);
          Right_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, ix_j,
                     peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }

      // y-bond A sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], opAB_y, 1, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // iy = num / LX;
          // iy_j = num_j / LX;
          iy = lattice.Tensor_position[num][1];
          iy_j = lattice.Tensor_position[num_j][1];
          Bottom_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, iy,
                      peps_parameters, lattice);
          Top_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, iy_j,
                   peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }

      // y-bond B sub-lattice
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], opBA_y, 1, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          // iy = num / LX;
          // iy_j = num_j / LX;
          iy = lattice.Tensor_position[num][1];
          iy_j = lattice.Tensor_position[num_j][1];
          Bottom_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, iy,
                      peps_parameters, lattice);
          Top_move(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, iy_j,
                   peps_parameters, lattice);
        } else {
          Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn,
                               peps_parameters, lattice, false);
        }
      }
    }
  }
  time_full_update += MPI_Wtime() - start_time;
  // done full update

  // output tensor
  if (mpirank == 0) {
    // folder check
    struct stat status;
    if (stat("output_tensors", &status) != 0) {
      mkdir("output_tensors", 0755);
    }
  }
  for (int i = 0; i < N_UNIT; i++) {
    std::stringstream ss;
    ss << i;
    std::string filename;

    filename = "output_tensors/Tn" + ss.str();
    Tn[i].save(filename.c_str());

    filename = "output_tensors/C1Mat" + ss.str();
    C1[i].save(filename.c_str());
    filename = "output_tensors/C2Mat" + ss.str();
    C2[i].save(filename.c_str());
    filename = "output_tensors/C3Mat" + ss.str();
    C3[i].save(filename.c_str());
    filename = "output_tensors/C4Mat" + ss.str();
    C4[i].save(filename.c_str());

    filename = "output_tensors/eTt" + ss.str();
    eTt[i].save(filename.c_str());
    filename = "output_tensors/eTr" + ss.str();
    eTr[i].save(filename.c_str());
    filename = "output_tensors/eTb" + ss.str();
    eTb[i].save(filename.c_str());
    filename = "output_tensors/eTl" + ss.str();
    eTl[i].save(filename.c_str());
  }
  if (mpirank == 0) {
    std::vector<double> lambda_save(N_UNIT * 4 * D);
    int num;
    for (int i1 = 0; i1 < N_UNIT; ++i1) {
      for (int i2 = 0; i2 < 4; ++i2) {
        for (int i3 = 0; i3 < D; ++i3) {
          num = i1 * 4 * D + i2 * D + i3;
          lambda_save[num] = lambda_tensor[i1][i2][i3];
        }
      }
    }
    std::string filename = "output_tensors/lambdas";
    std::ofstream fout(filename.c_str(), std::ofstream::binary);
    fout.write(reinterpret_cast<const char *>(&lambda_save[0]),
               sizeof(double) * N_UNIT * 4 * D);
    fout.close();
  }

  // Calc physical quantities

  if (local_parameters.Env_calc_before_obs) {
    if ((local_parameters.Read_Initial &&
         local_parameters.tau_full_step + local_parameters.tau_step == 0) ||
        local_parameters.tau_full_step > 0) {
      start_time = MPI_Wtime();
      count_CTM_env = Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl,
                                           Tn, peps_parameters, lattice, false);
      time_env += MPI_Wtime() - start_time;

      if (mpirank == 0) {
        std::ofstream ofs_env;
        ofs_env << std::setprecision(16);
        if (local_parameters.Output_file_append || file_count_CTM_exist) {
          ofs_env.open("output_data/CTM_count.dat",
                       std::ios::out | std::ios::app);
        } else {
          ofs_env.open("output_data/CTM_count.dat", std::ios::out);
        }
        ofs_env << "#Befor_Obs " << hx << " " << hz << " " << Dz << " " << Dp
                << " " << count_CTM_env << std::endl;
        ofs_env.close();
      }
    } else {
      start_time = MPI_Wtime();
      count_CTM_env = Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl,
                                           Tn, peps_parameters, lattice, true);
      time_env += MPI_Wtime() - start_time;
      if (mpirank == 0) {
        std::ofstream ofs_env;
        ofs_env << std::setprecision(16);
        if (local_parameters.Output_file_append || file_count_CTM_exist) {
          ofs_env.open("output_data/CTM_count.dat",
                       std::ios::out | std::ios::app);
        } else {
          ofs_env.open("output_data/CTM_count.dat", std::ios::out);
        }
        ofs_env << "#Befor_Obs " << hx << " " << hz << " " << Dz << " " << Dp
                << " " << count_CTM_env << std::endl;
        ofs_env.close();
      }
    }
  } else {
    if (local_parameters.tau_full_step == 0 &&
        (local_parameters.tau_step > 0 || !local_parameters.Read_Initial)) {
      // previously no environment calculation
      start_time = MPI_Wtime();
      count_CTM_env = Calc_CTM_Environment(C1, C2, C3, C4, eTt, eTr, eTb, eTl,
                                           Tn, peps_parameters, lattice, true);
      time_env += MPI_Wtime() - start_time;
      if (mpirank == 0) {
        std::ofstream ofs_env;
        ofs_env << std::setprecision(16);
        if (local_parameters.Output_file_append || file_count_CTM_exist) {
          ofs_env.open("output_data/CTM_count.dat",
                       std::ios::out | std::ios::app);
        } else {
          ofs_env.open("output_data/CTM_count.dat", std::ios::out);
        }
        ofs_env << "#Befor_Obs " << hx << " " << hz << " " << Dz << " " << Dp
                << " " << count_CTM_env << std::endl;
        ofs_env.close();
      }
    };
  };

  // output environment tensor
  for (int i = 0; i < N_UNIT; i++) {
    std::stringstream ss;
    ss << i;
    std::string filename;
    filename = "output_tensors/C1Mat" + ss.str();
    C1[i].save(filename.c_str());
    filename = "output_tensors/C2Mat" + ss.str();
    C2[i].save(filename.c_str());
    filename = "output_tensors/C3Mat" + ss.str();
    C3[i].save(filename.c_str());
    filename = "output_tensors/C4Mat" + ss.str();
    C4[i].save(filename.c_str());

    filename = "output_tensors/eTt" + ss.str();
    eTt[i].save(filename.c_str());
    filename = "output_tensors/eTr" + ss.str();
    eTr[i].save(filename.c_str());
    filename = "output_tensors/eTb" + ss.str();
    eTb[i].save(filename.c_str());
    filename = "output_tensors/eTl" + ss.str();
    eTl[i].save(filename.c_str());
  }

  // Calc physical quantities
  ptensor I(Shape(2, 2)), Sz(Shape(2, 2)), Sx(Shape(2, 2)), Sy(Shape(2, 2)),
      I_123(Shape(8, 8));

  Sz.set_value(Index(0, 0), 0.5);
  Sz.set_value(Index(1, 1), -0.5);

  Sx.set_value(Index(0, 1), 0.5);
  Sx.set_value(Index(1, 0), 0.5);

  //!!note operator should be transposed due to the definition
  // for convenience, "i" is tempolaly removed
  Sy.set_value(Index(0, 1), 0.5);
  Sy.set_value(Index(1, 0), -0.5);

  I.set_value(Index(0, 0), 1.0);
  I.set_value(Index(1, 1), 1.0);

  std::vector<ptensor> S_op, S_1, S_2, S_3, S_12, S_23;
  S_op.push_back(Sx);
  S_op.push_back(Sy);
  S_op.push_back(Sz);

  for (int i = 0; i < 3; i++) {
    S_1.push_back(Create_one_op(S_op[i], I, I));
    S_2.push_back(Create_one_op(I, S_op[i], I));
    S_3.push_back(Create_one_op(I, I, S_op[i]));
  }

  for (int i = 0; i < 9; i++) {
    int i1 = i / 3;
    int i2 = i % 3;
    S_12.push_back(Create_one_op(S_op[i1], S_op[i2], I));
    S_23.push_back(Create_one_op(I, S_op[i1], S_op[i2]));
  }

  for (int i = 0; i < 8; i++) {
    I_123.set_value(Index(i, i), 1.0);
  }

  std::vector<std::vector<double> > m_1(N_UNIT, std::vector<double>(3)),
      m_2(N_UNIT, std::vector<double>(3)), m_3(N_UNIT, std::vector<double>(3));
  std::vector<std::vector<double> > m_12(N_UNIT, std::vector<double>(9)),
      m_23(N_UNIT, std::vector<double>(9));
  std::vector<std::vector<double> > m2_12(N_UNIT, std::vector<double>(9)),
      m2_23(N_UNIT, std::vector<double>(9)),
      m2_31_1(N_UNIT, std::vector<double>(9)),
      m2_31_2(N_UNIT, std::vector<double>(9));

  // list for necessary elements of two-body interactions
  int num_2body;
  std::vector<int> list_2body;
  if (!local_parameters.Calc_all_2body) {
    if (Dz == 0.0 && Dp == 0.0) {
      num_2body = 3;
      list_2body.resize(3);
      list_2body[0] = 0;
      list_2body[1] = 4;
      list_2body[2] = 8;
    } else if (Dz == 0.0) {
      num_2body = 7;
      list_2body.resize(7);
      list_2body[0] = 0;
      list_2body[1] = 2;
      list_2body[2] = 4;
      list_2body[3] = 5;
      list_2body[4] = 6;
      list_2body[5] = 7;
      list_2body[6] = 8;
    } else if (Dp == 0.0) {
      num_2body = 5;
      list_2body.resize(5);
      list_2body[0] = 0;
      list_2body[1] = 1;
      list_2body[2] = 3;
      list_2body[3] = 4;
      list_2body[4] = 8;
    } else {
      num_2body = 9;
      list_2body.resize(9);
      for (int i = 0; i < 9; i++) {
        list_2body[i] = i;
      }
    }
  } else {
    num_2body = 9;
    list_2body.resize(9);
    for (int i = 0; i < 9; i++) {
      list_2body[i] = i;
    }
  }

  std::vector<tensor_value_type> norm(N_UNIT), norm_x(N_UNIT), norm_y(N_UNIT);
  start_time = MPI_Wtime();
  if (local_parameters.Obs_calc_mag || local_parameters.Obs_calc_energy) {
    for (int i = 0; i < N_UNIT; ++i) {
      norm[i] = Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                                  eTb[i], eTl[i], Tn[i], I_123);
      for (int n = 0; n < 3; n++) {
        tensor_value_type m_1_temp, m_2_temp, m_3_temp;
        m_1_temp = Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                                     eTb[i], eTl[i], Tn[i], S_1[n]) /
                   norm[i];
        m_2_temp = Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                                     eTb[i], eTl[i], Tn[i], S_2[n]) /
                   norm[i];
        m_3_temp = Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                                     eTb[i], eTl[i], Tn[i], S_3[n]) /
                   norm[i];

        if (n == 1) {
          // Sy
          m_1[i][n] = -get_imag_part(m_1_temp);
          m_2[i][n] = -get_imag_part(m_2_temp);
          m_3[i][n] = -get_imag_part(m_3_temp);
        } else {
          // Sx,Sz
          m_1[i][n] = get_real_part(m_1_temp);
          m_2[i][n] = get_real_part(m_2_temp);
          m_3[i][n] = get_real_part(m_3_temp);
        }
      };
      if (mpirank == 0) {
      }
    }
  }
  if (local_parameters.Obs_calc_energy) {
    for (int i = 0; i < N_UNIT; ++i) {
      for (int n2 = 0; n2 < num_2body; n2++) {
        int n = list_2body[n2];
        tensor_value_type m_12_temp, m_23_temp;

        m_12_temp = Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i],
                                      eTr[i], eTb[i], eTl[i], Tn[i], S_12[n]) /
                    norm[i];
        m_23_temp = Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i],
                                      eTr[i], eTb[i], eTl[i], Tn[i], S_23[n]) /
                    norm[i];
        if (n == 0 || n == 2 || n == 6 || n == 8) {
          // xx, xz, zx, zz
          m_12[i][n] = get_real_part(m_12_temp);
          m_23[i][n] = get_real_part(m_23_temp);
        } else if (n == 4) {
          // yy
          m_12[i][n] = -get_real_part(m_12_temp);
          m_23[i][n] = -get_real_part(m_23_temp);
        } else {
          // xy, yx, yz,zy
          m_12[i][n] = -get_imag_part(m_12_temp);
          m_23[i][n] = -get_imag_part(m_23_temp);
        }
      }

      if (mpirank == 0) {
      }
    }
  }
  if (local_parameters.Obs_calc_energy) {
    for (int n_sub = 0; n_sub < N_UNIT / 2; ++n_sub) {
      // A sub lattice
      int num = lattice.A_sub_list[n_sub];

      // x direction
      int num_j = lattice.NN_Tensor[num][2];
      norm_x[num] = Contract_two_sites_holizontal(
          C1[num], C2[num_j], C3[num_j], C4[num], eTt[num], eTt[num_j],
          eTr[num_j], eTb[num_j], eTb[num], eTl[num], Tn[num], Tn[num_j], I_123,
          I_123);

      for (int n = 0; n < num_2body; n++) {
        int i = list_2body[n];
        int i1 = i / 3;
        int i2 = i % 3;

        tensor_value_type m2_23_temp, m2_31_1_temp;
        m2_23_temp = Contract_two_sites_holizontal(
                         C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                         eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                         Tn[num], Tn[num_j], S_2[i1], S_3[i2]) /
                     norm_x[num];
        m2_31_1_temp = Contract_two_sites_holizontal(
                           C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                           eTt[num_j], eTr[num_j], eTb[num_j], eTb[num],
                           eTl[num], Tn[num], Tn[num_j], S_1[i2], S_3[i1]) /
                       norm_x[num];
        if (i == 0 || i == 2 || i == 6 || i == 8) {
          // xx, xz, zx, zz
          m2_23[num][i] = get_real_part(m2_23_temp);
          m2_31_1[num_j][i] = get_real_part(m2_31_1_temp);
        } else if (i == 4) {
          m2_23[num][i] = -get_real_part(m2_23_temp);
          m2_31_1[num_j][i] = -get_real_part(m2_31_1_temp);
        } else {
          m2_23[num][i] = -get_imag_part(m2_23_temp);
          m2_31_1[num_j][i] = -get_imag_part(m2_31_1_temp);
        }
      }
      // y direction
      num_j = lattice.NN_Tensor[num][3];

      norm_y[num] = Contract_two_sites_vertical(
          C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
          eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num], Tn[num_j],
          I_123, I_123);

      for (int n = 0; n < num_2body; n++) {
        int i = list_2body[n];
        int i1 = i / 3;
        int i2 = i % 3;

        tensor_value_type m2_23_temp, m2_31_1_temp;
        m2_23_temp = Contract_two_sites_vertical(
                         C1[num], C2[num], C3[num_j], C4[num_j], eTt[num],
                         eTr[num], eTr[num_j], eTb[num_j], eTl[num_j], eTl[num],
                         Tn[num], Tn[num_j], S_3[i2], S_2[i1]) /
                     norm_y[num];
        m2_31_1_temp = Contract_two_sites_vertical(
                           C1[num], C2[num], C3[num_j], C4[num_j], eTt[num],
                           eTr[num], eTr[num_j], eTb[num_j], eTl[num_j],
                           eTl[num], Tn[num], Tn[num_j], S_3[i1], S_1[i2]) /
                       norm_y[num];
        if (i == 0 || i == 2 || i == 6 || i == 8) {
          // xx, xz, zx, zz
          m2_23[num_j][i] = get_real_part(m2_23_temp);
          m2_31_1[num][i] = get_real_part(m2_31_1_temp);
        } else if (i == 4) {
          m2_23[num_j][i] = -get_real_part(m2_23_temp);
          m2_31_1[num][i] = -get_real_part(m2_31_1_temp);
        } else {
          m2_23[num_j][i] = -get_imag_part(m2_23_temp);
          m2_31_1[num][i] = -get_imag_part(m2_31_1_temp);
        }
      }
      // B sub lattice
      num = lattice.B_sub_list[n_sub];

      // x direction
      num_j = lattice.NN_Tensor[num][2];
      norm_x[num] = Contract_two_sites_holizontal(
          C1[num], C2[num_j], C3[num_j], C4[num], eTt[num], eTt[num_j],
          eTr[num_j], eTb[num_j], eTb[num], eTl[num], Tn[num], Tn[num_j], I_123,
          I_123);

      for (int n = 0; n < num_2body; n++) {
        int i = list_2body[n];
        int i1 = i / 3;
        int i2 = i % 3;

        tensor_value_type m2_12_temp, m2_31_2_temp;
        m2_12_temp = Contract_two_sites_holizontal(
                         C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                         eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                         Tn[num], Tn[num_j], S_1[i1], S_2[i2]) /
                     norm_x[num];
        m2_31_2_temp = Contract_two_sites_holizontal(
                           C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                           eTt[num_j], eTr[num_j], eTb[num_j], eTb[num],
                           eTl[num], Tn[num], Tn[num_j], S_1[i2], S_3[i1]) /
                       norm_x[num];
        if (i == 0 || i == 2 || i == 6 || i == 8) {
          // xx, xz, zx, zz
          m2_12[num][i] = get_real_part(m2_12_temp);
          m2_31_2[num_j][i] = get_real_part(m2_31_2_temp);
        } else if (i == 4) {
          m2_12[num][i] = -get_real_part(m2_12_temp);
          m2_31_2[num_j][i] = -get_real_part(m2_31_2_temp);
        } else {
          m2_12[num][i] = -get_imag_part(m2_12_temp);
          m2_31_2[num_j][i] = -get_imag_part(m2_31_2_temp);
        }
      }
      // y direction
      num_j = lattice.NN_Tensor[num][3];

      norm_y[num] = Contract_two_sites_vertical(
          C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
          eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num], Tn[num_j],
          I_123, I_123);

      for (int n = 0; n < num_2body; n++) {
        int i = list_2body[n];
        int i1 = i / 3;
        int i2 = i % 3;

        tensor_value_type m2_12_temp, m2_31_2_temp;
        m2_12_temp = Contract_two_sites_vertical(
                         C1[num], C2[num], C3[num_j], C4[num_j], eTt[num],
                         eTr[num], eTr[num_j], eTb[num_j], eTl[num_j], eTl[num],
                         Tn[num], Tn[num_j], S_2[i2], S_1[i1]) /
                     norm_y[num];
        m2_31_2_temp = Contract_two_sites_vertical(
                           C1[num], C2[num], C3[num_j], C4[num_j], eTt[num],
                           eTr[num], eTr[num_j], eTb[num_j], eTl[num_j],
                           eTl[num], Tn[num], Tn[num_j], S_3[i1], S_1[i2]) /
                       norm_y[num];
        if (i == 0 || i == 2 || i == 6 || i == 8) {
          // xx, xz, zx, zz
          m2_12[num_j][i] = get_real_part(m2_12_temp);
          m2_31_2[num][i] = get_real_part(m2_31_2_temp);
        } else if (i == 4) {
          m2_12[num_j][i] = -get_real_part(m2_12_temp);
          m2_31_2[num][i] = -get_real_part(m2_31_2_temp);
        } else {
          m2_12[num_j][i] = -get_imag_part(m2_12_temp);
          m2_31_2[num][i] = -get_imag_part(m2_31_2_temp);
        }
      }
    }
  }
  time_obs += MPI_Wtime() - start_time;

  if (mpirank == 0) {
    std::vector<double> total_mag(3);
    if (local_parameters.Obs_calc_mag || local_parameters.Obs_calc_energy) {
      std::vector<double> sublattice_mag_A(3), sublattice_mag_B(3),
          sublattice_mag_C(3); // q=0 magnetization
      std::ofstream ofs_mag_sub, ofs_mag, ofs_mag_each;
      ofs_mag << std::setprecision(16);
      ofs_mag_sub << std::setprecision(16);
      if (local_parameters.Output_file_append) {
        ofs_mag.open("output_data/Magnetization.dat",
                     std::ios::out | std::ios::app);
        ofs_mag_sub.open("output_data/Sub_Magnetization.dat",
                         std::ios::out | std::ios::app);
        ofs_mag_each.open("output_data/Magnetization_each.dat",
                          std::ios::out | std::ios::app);
      } else {
        ofs_mag.open("output_data/Magnetization.dat", std::ios::out);
        ofs_mag_sub.open("output_data/Sub_Magnetization.dat", std::ios::out);
        ofs_mag_each.open("output_data/Magnetization_each.dat", std::ios::out);
      }

      for (int num = 0; num < N_UNIT; ++num) {
        std::cout << "## Mag 1 " << hx << " " << hz << " " << Dz << " " << Dp
                  << " " << num << " " << m_1[num][0] << " " << m_1[num][1]
                  << " " << m_1[num][2] << " "
                  << sqrt(m_1[num][0] * m_1[num][0] +
                          m_1[num][1] * m_1[num][1] + m_1[num][2] * m_1[num][2])
                  << std::endl;
        std::cout << "## Mag 2 " << hx << " " << hz << " " << Dz << " " << Dp
                  << " " << num << " " << m_2[num][0] << " " << m_2[num][1]
                  << " " << m_2[num][2] << " "
                  << sqrt(m_2[num][0] * m_2[num][0] +
                          m_2[num][1] * m_2[num][1] + m_2[num][2] * m_2[num][2])
                  << std::endl;
        std::cout << "## Mag 3 " << hx << " " << hz << " " << Dz << " " << Dp
                  << " " << num << " " << m_3[num][0] << " " << m_3[num][1]
                  << " " << m_3[num][2] << " "
                  << sqrt(m_3[num][0] * m_3[num][0] +
                          m_3[num][1] * m_3[num][1] + m_3[num][2] * m_3[num][2])
                  << std::endl;

        ofs_mag_each << hx << " " << hz << " " << Dz << " " << Dp << " " << num
                     << " " << m_1[num][0] << " " << m_1[num][1] << " "
                     << m_1[num][2] << " "
                     << sqrt(m_1[num][0] * m_1[num][0] +
                             m_1[num][1] * m_1[num][1] +
                             m_1[num][2] * m_1[num][2])
                     << " " << norm[num] << std::endl;
        ofs_mag_each << hx << " " << hz << " " << Dz << " " << Dp << " " << num
                     << " " << m_2[num][0] << " " << m_2[num][1] << " "
                     << m_2[num][2] << " "
                     << sqrt(m_2[num][0] * m_2[num][0] +
                             m_2[num][1] * m_2[num][1] +
                             m_2[num][2] * m_2[num][2])
                     << " " << norm[num] << std::endl;
        ofs_mag_each << hx << " " << hz << " " << Dz << " " << Dp << " " << num
                     << " " << m_3[num][0] << " " << m_3[num][1] << " "
                     << m_3[num][2] << " "
                     << sqrt(m_3[num][0] * m_3[num][0] +
                             m_3[num][1] * m_3[num][1] +
                             m_3[num][2] * m_3[num][2])
                     << " " << norm[num] << std::endl;
      }
      ofs_mag_each << std::endl;

      for (int n_sub = 0; n_sub < N_UNIT / 2; n_sub++) {
        // A sub lattice
        int num = lattice.A_sub_list[n_sub];
        for (int i = 0; i < 3; i++) {
          sublattice_mag_B[i] += m_1[num][i] + m_3[num][i];
          sublattice_mag_C[i] += m_2[num][i];
        }
        // B sub lattice
        num = lattice.B_sub_list[n_sub];
        for (int i = 0; i < 3; i++) {
          sublattice_mag_A[i] += m_1[num][i] + m_3[num][i];
          sublattice_mag_C[i] += m_2[num][i];
        }
      }

      for (int i = 0; i < 3; i++) {
        sublattice_mag_A[i] /= N_UNIT;
        sublattice_mag_B[i] /= N_UNIT;
        sublattice_mag_C[i] /= N_UNIT;

        total_mag[i] =
            (sublattice_mag_A[i] + sublattice_mag_B[i] + sublattice_mag_C[i]) /
            3.0;
      }

      ofs_mag << hx << " " << hz << " " << Dz << " " << Dp << " "
              << total_mag[0] << " " << total_mag[1] << " " << total_mag[2]
              << " "
              << sqrt(total_mag[0] * total_mag[0] +
                      total_mag[1] * total_mag[1] + total_mag[2] * total_mag[2])
              << std::endl;

      ofs_mag_sub << hx << " " << hz << " " << Dz << " " << Dp << " "
                  << sublattice_mag_A[0] << " " << sublattice_mag_A[1] << " "
                  << sublattice_mag_A[2] << " "
                  << sqrt(sublattice_mag_A[0] * sublattice_mag_A[0] +
                          sublattice_mag_A[1] * sublattice_mag_A[1] +
                          sublattice_mag_A[2] * sublattice_mag_A[2])
                  << std::endl;

      ofs_mag_sub << hx << " " << hz << " " << Dz << " " << Dp << " "
                  << sublattice_mag_B[0] << " " << sublattice_mag_B[1] << " "
                  << sublattice_mag_B[2] << " "
                  << sqrt(sublattice_mag_B[0] * sublattice_mag_B[0] +
                          sublattice_mag_B[1] * sublattice_mag_B[1] +
                          sublattice_mag_B[2] * sublattice_mag_B[2])
                  << std::endl;

      ofs_mag_sub << hx << " " << hz << " " << Dz << " " << Dp << " "
                  << sublattice_mag_C[0] << " " << sublattice_mag_C[1] << " "
                  << sublattice_mag_C[2] << " "
                  << sqrt(sublattice_mag_C[0] * sublattice_mag_C[0] +
                          sublattice_mag_C[1] * sublattice_mag_C[1] +
                          sublattice_mag_C[2] * sublattice_mag_C[2])
                  << std::endl;

      ofs_mag_sub << std::endl;

      std::cout << "#Total Magnetization: " << hx << " " << hz << " " << Dz
                << " " << Dp << " " << total_mag[0] << " " << total_mag[1]
                << " " << total_mag[2] << " "
                << sqrt(total_mag[0] * total_mag[0] +
                        total_mag[1] * total_mag[1] +
                        total_mag[2] * total_mag[2])
                << std::endl;

      std::cout << "#Sublattice Magnetization A: " << hx << " " << hz << " "
                << Dz << " " << Dp << " " << sublattice_mag_A[0] << " "
                << sublattice_mag_A[1] << " " << sublattice_mag_A[2] << " "
                << sqrt(sublattice_mag_A[0] * sublattice_mag_A[0] +
                        sublattice_mag_A[1] * sublattice_mag_A[1] +
                        sublattice_mag_A[2] * sublattice_mag_A[2])
                << std::endl;

      std::cout << "#Sublattice Magnetization B: " << hx << " " << hz << " "
                << Dz << " " << Dp << " " << sublattice_mag_B[0] << " "
                << sublattice_mag_B[1] << " " << sublattice_mag_B[2] << " "
                << sqrt(sublattice_mag_B[0] * sublattice_mag_B[0] +
                        sublattice_mag_B[1] * sublattice_mag_B[1] +
                        sublattice_mag_B[2] * sublattice_mag_B[2])
                << std::endl;

      std::cout << "#Sublattice Magnetization C: " << hx << " " << hz << " "
                << Dz << " " << Dp << " " << sublattice_mag_C[0] << " "
                << sublattice_mag_C[1] << " " << sublattice_mag_C[2] << " "
                << sqrt(sublattice_mag_C[0] * sublattice_mag_C[0] +
                        sublattice_mag_C[1] * sublattice_mag_C[1] +
                        sublattice_mag_C[2] * sublattice_mag_C[2])
                << std::endl;

      ofs_mag.close();
      ofs_mag_sub.close();
      ofs_mag_each.close();
    }

    if (local_parameters.Obs_calc_energy) {

      for (int num = 0; num < N_UNIT; num++) {
        std::cout << "## Dot 12 " << hx << " " << hz << " " << Dz << " " << Dp
                  << " " << num;
        for (int n = 0; n < 9; n++) {
          std::cout << " " << m_12[num][n];
        }
        std::cout << std::endl;

        std::cout << "## Dot2 12 " << hx << " " << hz << " " << Dz << " " << Dp
                  << " " << num;
        for (int n = 0; n < 9; n++) {
          std::cout << " " << m2_12[num][n];
        }
        std::cout << std::endl;

        std::cout << "## Dot 23 " << hx << " " << hz << " " << Dz << " " << Dp
                  << " " << num;
        for (int n = 0; n < 9; n++) {
          std::cout << " " << m_23[num][n];
        }
        std::cout << std::endl;

        std::cout << "## Dot2 23 " << hx << " " << hz << " " << Dz << " " << Dp
                  << " " << num;
        for (int n = 0; n < 9; n++) {
          std::cout << " " << m2_23[num][n];
        }
        std::cout << std::endl;

        std::cout << "## Dot2 31_1 " << hx << " " << hz << " " << Dz << " "
                  << Dp << " " << num;
        for (int n = 0; n < 9; n++) {
          std::cout << " " << m2_31_1[num][n];
        }
        std::cout << std::endl;

        std::cout << "## Dot2 31_2 " << hx << " " << hz << " " << Dz << " "
                  << Dp << " " << num;
        for (int n = 0; n < 9; n++) {
          std::cout << " " << m2_31_2[num][n];
        }
        std::cout << std::endl;
      }

      std::ofstream ofs_energy_bond, ofs_energy;
      std::ofstream ofs_corr_each;
      ofs_energy << std::setprecision(16);
      ofs_energy_bond << std::setprecision(16);
      ofs_corr_each << std::setprecision(16);
      if (local_parameters.Output_file_append) {
        ofs_energy.open("output_data/Energy.dat",
                        std::ios::out | std::ios::app);
        ofs_energy_bond.open("output_data/Energy_bond.dat",
                             std::ios::out | std::ios::app);
        ofs_corr_each.open("output_data/Correlation_each.dat",
                           std::ios::out | std::ios::app);
      } else {
        ofs_energy.open("output_data/Energy.dat", std::ios::out);
        ofs_energy_bond.open("output_data/Energy_bond.dat", std::ios::out);
        ofs_corr_each.open("output_data/Correlation_each.dat", std::ios::out);
      }

      for (int num = 0; num < N_UNIT; num++) {
        ofs_corr_each << hx << " " << hz << " " << Dz << " " << Dp << " "
                      << num;
        for (int n = 0; n < 9; n++) {
          ofs_corr_each << " " << m_12[num][n];
        }
        ofs_corr_each << std::endl;

        ofs_corr_each << hx << " " << hz << " " << Dz << " " << Dp << " "
                      << num;
        for (int n = 0; n < 9; n++) {
          ofs_corr_each << " " << m2_12[num][n];
        }
        ofs_corr_each << std::endl;

        ofs_corr_each << hx << " " << hz << " " << Dz << " " << Dp << " "
                      << num;
        for (int n = 0; n < 9; n++) {
          ofs_corr_each << " " << m_23[num][n];
        }
        ofs_corr_each << std::endl;

        ofs_corr_each << hx << " " << hz << " " << Dz << " " << Dp << " "
                      << num;
        for (int n = 0; n < 9; n++) {
          ofs_corr_each << " " << m2_23[num][n];
        }
        ofs_corr_each << std::endl;

        ofs_corr_each << hx << " " << hz << " " << Dz << " " << Dp << " "
                      << num;
        for (int n = 0; n < 9; n++) {
          ofs_corr_each << " " << m2_31_1[num][n];
        }
        ofs_corr_each << std::endl;

        ofs_corr_each << hx << " " << hz << " " << Dz << " " << Dp << " "
                      << num;
        for (int n = 0; n < 9; n++) {
          ofs_corr_each << " " << m2_31_2[num][n];
        }
        ofs_corr_each << std::endl;
      }
      ofs_corr_each << std::endl;

      double Energy_Heisenberg_AB = 0.0;
      double Energy_Heisenberg_BC = 0.0;
      double Energy_Heisenberg_CA = 0.0;

      double Energy_Dz_AB = 0.0;
      double Energy_Dz_BC = 0.0;
      double Energy_Dz_CA = 0.0;

      double Energy_Dp_AB = 0.0;
      double Energy_Dp_BC = 0.0;
      double Energy_Dp_CA = 0.0;

      for (int n_sub = 0; n_sub < N_UNIT / 2; n_sub++) {
        // A sub lattice
        int num = lattice.A_sub_list[n_sub];
        Energy_Heisenberg_AB +=
            (m2_31_1[num][0] + m2_31_1[num][4] + m2_31_1[num][8]) +
            (m2_31_2[num][0] + m2_31_2[num][4] + m2_31_2[num][8]);
        Energy_Heisenberg_BC += (m_12[num][0] + m_12[num][4] + m_12[num][8]) +
                                (m_23[num][0] + m_23[num][4] + m_23[num][8]) +
                                (m2_12[num][0] + m2_12[num][4] + m2_12[num][8]);
        Energy_Heisenberg_CA += (m2_23[num][0] + m2_23[num][4] + m2_23[num][8]);

        if (Dz != 0.0) {
          Energy_Dz_AB += -(m2_31_1[num][1] - m2_31_1[num][3]) -
                          (m2_31_2[num][1] - m2_31_2[num][3]);
          Energy_Dz_BC += (m_12[num][1] - m_12[num][3]) -
                          (m_23[num][1] - m_23[num][3]) +
                          (m2_12[num][1] - m2_12[num][3]);
          Energy_Dz_CA += (m2_23[num][1] - m2_23[num][3]);
        }

        if (Dp != 0.0) {
          Energy_Dp_AB +=
              std::cos(M_PI / 2.0) * (-(m2_31_1[num][5] - m2_31_1[num][7]) -
                                      (m2_31_2[num][5] - m2_31_2[num][7]));
          Energy_Dp_AB +=
              std::sin(M_PI / 2.0) * (-(m2_31_1[num][6] - m2_31_1[num][2]) -
                                      (m2_31_2[num][6] - m2_31_2[num][2]));
          Energy_Dp_BC +=
              std::cos(7.0 * M_PI / 6.0) *
              ((m_12[num][5] - m_12[num][7]) - (m_23[num][5] - m_23[num][7]) +
               (m2_23[num][5] - m2_23[num][7]));
          Energy_Dp_BC +=
              std::sin(7.0 * M_PI / 6.0) *
              ((m_12[num][6] - m_12[num][2]) - (m_23[num][6] - m_23[num][2]) +
               (m2_23[num][6] - m2_23[num][2]));
          Energy_Dp_CA +=
              std::cos(11.0 * M_PI / 6.0) * (m2_23[num][5] - m2_23[num][7]);
          Energy_Dp_CA +=
              std::sin(11.0 * M_PI / 6.0) * (m2_23[num][6] - m2_23[num][2]);
        }
        // B sub lattice
        num = lattice.B_sub_list[n_sub];
        Energy_Heisenberg_AB +=
            (m2_31_1[num][0] + m2_31_1[num][4] + m2_31_1[num][8]) +
            (m2_31_2[num][0] + m2_31_2[num][4] + m2_31_2[num][8]);
        Energy_Heisenberg_BC += (m2_23[num][0] + m2_23[num][4] + m2_23[num][8]);
        Energy_Heisenberg_CA += (m_12[num][0] + m_12[num][4] + m_12[num][8]) +
                                (m_23[num][0] + m_23[num][4] + m_23[num][8]) +
                                (m2_12[num][0] + m2_12[num][4] + m2_12[num][8]);

        if (Dz != 0.0) {
          Energy_Dz_AB += (m2_31_1[num][1] - m2_31_1[num][3]) +
                          (m2_31_2[num][1] - m2_31_2[num][3]);
          Energy_Dz_BC += -(m2_23[num][1] - m2_23[num][3]);
          Energy_Dz_CA += -(m_12[num][1] - m_12[num][3]) +
                          (m_23[num][1] - m_23[num][3]) -
                          (m2_12[num][1] - m2_12[num][3]);
        }

        if (Dp != 0.0) {
          Energy_Dp_AB +=
              std::cos(M_PI / 2.0) * ((m2_31_1[num][5] - m2_31_1[num][7]) +
                                      (m2_31_2[num][5] - m2_31_2[num][7]));
          Energy_Dp_AB +=
              std::sin(M_PI / 2.0) * ((m2_31_1[num][6] - m2_31_1[num][2]) +
                                      (m2_31_2[num][6] - m2_31_2[num][2]));
          Energy_Dp_BC +=
              std::cos(7.0 * M_PI / 6.0) * (-(m2_23[num][5] - m2_23[num][7]));
          Energy_Dp_BC +=
              std::sin(7.0 * M_PI / 6.0) * (-(m2_23[num][6] - m2_23[num][2]));
          Energy_Dp_CA +=
              std::cos(11.0 * M_PI / 6.0) *
              (-(m_12[num][5] - m_12[num][7]) + (m_23[num][5] - m_23[num][7]) -
               (m2_12[num][5] - m2_12[num][7]));
          Energy_Dp_CA +=
              std::sin(11.0 * M_PI / 6.0) *
              (-(m_12[num][6] - m_12[num][2]) + (m_23[num][6] - m_23[num][2]) -
               (m2_12[num][6] - m2_12[num][2]));
        }
      }
      // per unit
      Energy_Heisenberg_AB /= N_UNIT;
      Energy_Heisenberg_BC /= N_UNIT;
      Energy_Heisenberg_CA /= N_UNIT;

      Energy_Dz_AB *= Dz / N_UNIT;
      Energy_Dz_BC *= Dz / N_UNIT;
      Energy_Dz_CA *= Dz / N_UNIT;

      Energy_Dp_AB *= Dp / N_UNIT;
      Energy_Dp_BC *= Dp / N_UNIT;
      Energy_Dp_CA *= Dp / N_UNIT;

      double Energy_AB = Energy_Heisenberg_AB + Energy_Dz_AB + Energy_Dp_AB;
      double Energy_BC = Energy_Heisenberg_BC + Energy_Dz_BC + Energy_Dp_BC;
      double Energy_CA = Energy_Heisenberg_CA + Energy_Dz_CA + Energy_Dp_CA;

      ofs_energy_bond << hx << "  " << hz << " " << Dz << " " << Dp << " "
                      << Energy_AB << " " << Energy_BC << " " << Energy_CA
                      << " " << Energy_Heisenberg_AB << " "
                      << Energy_Heisenberg_BC << " " << Energy_Heisenberg_CA
                      << " " << Energy_Dz_AB << " " << Energy_Dz_BC << " "
                      << Energy_Dz_CA << " " << Energy_Dp_AB << " "
                      << Energy_Dp_BC << " " << Energy_Dp_CA << std::endl;
      std::cout << "## Energy AB " << hx << " " << hz << " " << Dz << " " << Dp
                << " " << Energy_Heisenberg_AB + Energy_Dz_AB + Energy_Dp_AB
                << " " << Energy_Heisenberg_AB << " " << Energy_Dz_AB << " "
                << Energy_Dp_AB << std::endl;
      std::cout << "## Energy BC " << hx << " " << hz << " " << Dz << " " << Dp
                << " " << Energy_Heisenberg_BC + Energy_Dz_BC + Energy_Dp_BC
                << " " << Energy_Heisenberg_BC << " " << Energy_Dz_BC << " "
                << Energy_Dp_BC << std::endl;
      std::cout << "## Energy CA " << hx << " " << hz << " " << Dz << " " << Dp
                << " " << Energy_Heisenberg_CA + Energy_Dz_CA + Energy_Dp_CA
                << " " << Energy_Heisenberg_CA << " " << Energy_Dz_CA << " "
                << Energy_Dp_CA << std::endl;

      // Total Energy
      double total_Energy_Heisenberg =
          (Energy_Heisenberg_AB + Energy_Heisenberg_BC + Energy_Heisenberg_CA) /
          3.0;
      double total_Energy_Dz =
          (Energy_Dz_AB + Energy_Dz_BC + Energy_Dz_CA) / 3.0;
      double total_Energy_Dp =
          (Energy_Dp_AB + Energy_Dp_BC + Energy_Dp_CA) / 3.0;
      double total_Energy_hx = -hx * total_mag[0];
      double total_Energy_hz = -hz * total_mag[2];

      double total_Energy = total_Energy_Heisenberg + total_Energy_Dz +
                            total_Energy_Dp + total_Energy_hx + total_Energy_hz;

      std::cout << "## Total Energy " << hx << " " << hz << " " << Dz << " "
                << Dp << " " << total_Energy << " " << total_Energy_Heisenberg
                << " " << total_Energy_Dz << " " << total_Energy_Dp << " "
                << total_Energy_hx << " " << total_Energy_hz << std::endl;
      ofs_energy << hx << " " << hz << " " << Dz << " " << Dp << " "
                 << total_Energy << " " << total_Energy_Heisenberg << " "
                 << total_Energy_Dz << " " << total_Energy_Dp << " "
                 << total_Energy_hx << " " << total_Energy_hz << std::endl;

      ofs_energy.close();
      ofs_energy_bond.close();
      ofs_corr_each.close();
    }

    std::ofstream ofs_timer;
    ofs_timer << std::setprecision(16);
    if (local_parameters.Output_file_append) {
      ofs_timer.open("output_data/Timer.dat", std::ios::out | std::ios::app);
    } else {
      ofs_timer.open("output_data/Timer.dat", std::ios::out);
    }
    std::cout << "##time simple update= " << time_simple_update << std::endl;
    std::cout << "##time full update= " << time_full_update << std::endl;
    std::cout << "##time environmnent= " << time_env << std::endl;
    std::cout << "##time observable= " << time_obs << std::endl;

    ofs_timer << hx << "  " << hz << " " << Dz << " " << Dp << " "
              << time_simple_update << " " << time_full_update << " "
              << time_env << " " << time_obs << std::endl;
    ofs_timer.close();
  }
  MPI_Finalize();
  return 0;
}
