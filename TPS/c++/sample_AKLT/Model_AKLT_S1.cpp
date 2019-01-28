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
typedef double tensor_value_type;
typedef Tensor<scalapack::Matrix, tensor_value_type> ptensor;
typedef Tensor<lapack::Matrix, tensor_value_type> tensor;

/* for MPI */
int mpirank;
int mpisize;

class Lattice_two_sub : public Lattice {
public:
  std::vector<int> A_sub_list, B_sub_list;
  Lattice_two_sub() : Lattice() {
    A_sub_list.resize(N_UNIT / 2);
    B_sub_list.resize(N_UNIT / 2);
  }
  void set_lattice_info() {
    // Lattice setting
    int a_num = 0;
    int b_num = 0;
    A_sub_list.resize(N_UNIT / 2);
    B_sub_list.resize(N_UNIT / 2);
    int num;
    for (int ix = 0; ix < LX; ++ix) {
      for (int iy = 0; iy < LY; ++iy) {
        num = iy * LX + ix;
        Tensor_list[ix][iy] = num;
        if ((ix + iy) % 2 == 0) {
          A_sub_list[a_num] = num;
          a_num += 1;
        } else {
          B_sub_list[b_num] = num;
          b_num += 1;
        }
      }
    };

    int ix, iy;
    for (int i = 0; i < N_UNIT; ++i) {
      ix = i % LX;
      iy = i / LX;

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

  double theta;
  //
  bool Read_Initial;
  bool Initialize_AKLT;
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

    theta = 0.0;

    Read_Initial = false;
    Initialize_AKLT = true;
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
        } else if (result[0].compare("theta") == 0) {
          std::istringstream is(result[1]);
          is >> theta;
        } else if (result[0].compare("Read_Initial") == 0) {
          std::istringstream is(result[1]);
          is >> Read_Initial;
        } else if (result[0].compare("Initialize_AKLT") == 0) {
          std::istringstream is(result[1]);
          is >> Initialize_AKLT;
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

    ofs << "theta " << theta << std::endl;

    ofs << "Read_Initial " << Read_Initial << std::endl;
    ofs << "Initialize_AKLT " << Initialize_AKLT << std::endl;
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

    std::vector<double> params_double(4);
    std::vector<int> params_int(11);

    if (irank == 0) {
      params_int[0] = random_seed_global;
      params_int[1] = Read_Initial;
      params_int[2] = Initialize_AKLT;
      params_int[3] = second_ST;
      params_int[4] = tau_step;
      params_int[5] = tau_full_step;
      params_int[6] = Env_calc_before_full;
      params_int[7] = Env_calc_before_obs;
      params_int[8] = Obs_calc_mag;
      params_int[9] = Obs_calc_energy;
      params_int[10] = Output_file_append;

      params_double[0] = theta;
      params_double[1] = Random_amp;
      params_double[2] = tau;
      params_double[3] = tau_full;

      MPI_Bcast(&params_int.front(), 11, MPI_INT, 0, comm);
      MPI_Bcast(&params_double.front(), 4, MPI_DOUBLE, 0, comm);
    } else {
      MPI_Bcast(&params_int.front(), 11, MPI_INT, 0, comm);
      MPI_Bcast(&params_double.front(), 4, MPI_DOUBLE, 0, comm);

      random_seed_global = params_int[0];
      Read_Initial = params_int[1];
      Initialize_AKLT = params_int[2];
      second_ST = params_int[3];
      tau_step = params_int[4];
      tau_full_step = params_int[5];

      theta = params_double[0];
      Random_amp = params_double[1];
      tau = params_double[2];
      tau_full = params_double[3];

      Env_calc_before_full = params_int[6];
      Env_calc_before_obs = params_int[7];
      Obs_calc_mag = params_int[8];
      Obs_calc_energy = params_int[9];

      Output_file_append = params_int[10];
    }
  };
};

ptensor Set_Hamiltonian() {
  /* メモ：mptensor の初期値はzero */
  ptensor Ham(Shape(9, 9));

  Ham.set_value(Index(0, 0), 1.0);
  Ham.set_value(Index(1, 1), 0.5);
  Ham.set_value(Index(2, 2), 1.0 / 6.0);
  Ham.set_value(Index(3, 3), 0.5);
  Ham.set_value(Index(4, 4), 2.0 / 3.0);
  Ham.set_value(Index(5, 5), 0.5);
  Ham.set_value(Index(6, 6), 1.0 / 6.0);
  Ham.set_value(Index(7, 7), 0.5);
  Ham.set_value(Index(8, 8), 1.0);

  Ham.set_value(Index(1, 3), 0.5);
  Ham.set_value(Index(3, 1), 0.5);
  Ham.set_value(Index(5, 7), 0.5);
  Ham.set_value(Index(7, 5), 0.5);

  Ham.set_value(Index(2, 6), 1.0 / 6.0);
  Ham.set_value(Index(6, 2), 1.0 / 6.0);
  Ham.set_value(Index(2, 4), 1.0 / 3.0);
  Ham.set_value(Index(4, 2), 1.0 / 3.0);
  Ham.set_value(Index(4, 6), 1.0 / 3.0);
  Ham.set_value(Index(6, 4), 1.0 / 3.0);

  return Ham;
}

void Initialize_Tensors(std::vector<Tensor<scalapack::Matrix, double> > &Tn,
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

  if (local_parameters.Initialize_AKLT) {
    // horizontal AKLT
    for (int i = 0; i < lattice.N_UNIT; ++i) {
      for (int n = 0; n < Tn[i].local_size(); ++n) {
        index = Tn[i].global_index(n);
        if (index == Index(0, 0, 1, 0, 0)) {
          Tn[i].set_value(index, sqrt(2.0 / 3.0));
        } else if (index == Index(0, 0, 0, 0, 1)) {
          Tn[i].set_value(index, -sqrt(1.0 / 3.0));
        } else if (index == Index(1, 0, 1, 0, 1)) {
          Tn[i].set_value(index, sqrt(1.0 / 3.0));
        } else if (index == Index(1, 0, 0, 0, 2)) {
          Tn[i].set_value(index, -sqrt(2.0 / 3.0));
        } else {
#ifdef DSFMT
          Tn[i].set_value(index, local_parameters.Random_amp *
                                     (dsfmt_genrand_close_open(&dsfmt) - 0.5));
#else
          Tn[i].set_value(index, local_parameters.Random_amp * dist(gen));
#endif
        }
      }
    }
  } else {
    // Neel
    for (int i = 0; i < lattice.N_UNIT / 2; ++i) {
      int num = lattice.A_sub_list[i];
      for (int n = 0; n < Tn[num].local_size(); ++n) {
        index = Tn[num].global_index(n);
        if (index == Index(0, 0, 0, 0, 0)) {
          Tn[num].set_value(index, 1.0);
        } else {
#ifdef DSFMT
          Tn[num].set_value(index,
                            local_parameters.Random_amp *
                                (dsfmt_genrand_close_open(&dsfmt) - 0.5));
#else
          Tn[num].set_value(index, local_parameters.Random_amp * dist(gen));
#endif
        }
      }
      num = lattice.B_sub_list[i];
      for (int n = 0; n < Tn[num].local_size(); ++n) {
        index = Tn[num].global_index(n);
        if (index == Index(0, 0, 0, 0, 2)) {
          Tn[num].set_value(index, 1.0);
        } else {
#ifdef DSFMT
          Tn[num].set_value(index,
                            local_parameters.Random_amp *
                                (dsfmt_genrand_close_open(&dsfmt) - 0.5));
#else
          Tn[num].set_value(index, local_parameters.Random_amp * dist(gen));
#endif
        }
      }
    }
  }
}

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
  double Jx, Jy;

  if (mpirank == 0) {
    local_parameters.read_parameters("input.dat");
    peps_parameters.read_parameters("input.dat");
    lattice.read_parameters("input.dat");
    lattice.N_UNIT = lattice.LX * lattice.LY;
  }

  local_parameters.Bcast_parameters(MPI_COMM_WORLD);
  peps_parameters.Bcast_parameters(MPI_COMM_WORLD);
  lattice.Bcast_parameters(MPI_COMM_WORLD);

  lattice.set_lattice_info();
  Jx = cos(local_parameters.theta * M_PI);
  Jy = sin(local_parameters.theta * M_PI);

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
  int N_UNIT = lattice.N_UNIT;

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
  random_tensor::set_seed(local_parameters.random_seed + mpisize);

  // Tensors
  std::vector<ptensor> Tn(N_UNIT, ptensor(Shape(D, D, D, D, 3)));
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

  ptensor Ham = Set_Hamiltonian();
  ptensor U, Ud, Us;
  std::vector<double> s;
  int info = eigh(Ham, s, U);
  std::vector<double> exp_s(9);
  Ud = conj(U);
  for (int i = 0; i < 9; ++i) {
    exp_s[i] = exp(-local_parameters.tau * s[i] * Jx);
  }
  Us = U;
  Us.multiply_vector(exp_s, 1);

  ptensor op12_Jx =
      reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(3, 3, 3, 3))
          .transpose(Axes(2, 3, 0, 1));

  for (int i = 0; i < 9; ++i) {
    exp_s[i] = exp(-0.5 * local_parameters.tau * s[i] * Jx);
  }
  Us = U;
  Us.multiply_vector(exp_s, 1);

  ptensor op12_Jx_2 =
      reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(3, 3, 3, 3))
          .transpose(Axes(2, 3, 0, 1));

  for (int i = 0; i < 9; ++i) {
    exp_s[i] = exp(-local_parameters.tau * s[i] * Jy);
  }
  Us = U;
  Us.multiply_vector(exp_s, 1);

  ptensor op12_Jy =
      reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(3, 3, 3, 3))
          .transpose(Axes(2, 3, 0, 1));
  for (int i = 0; i < 9; ++i) {
    exp_s[i] = exp(-0.5 * local_parameters.tau * s[i] * Jy);
  }
  Us = U;
  Us.multiply_vector(exp_s, 1);

  ptensor op12_Jy_2 =
      reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(3, 3, 3, 3))
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
                           lambda_tensor[num_j], op12_Jx_2, 2, peps_parameters,
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
                           lambda_tensor[num_j], op12_Jx_2, 2, peps_parameters,
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
                           lambda_tensor[num_j], op12_Jy_2, 1, peps_parameters,
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
                           lambda_tensor[num_j], op12_Jy, 1, peps_parameters,
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
                           lambda_tensor[num_j], op12_Jy_2, 1, peps_parameters,
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
                           lambda_tensor[num_j], op12_Jx_2, 2, peps_parameters,
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
                           lambda_tensor[num_j], op12_Jx_2, 2, peps_parameters,
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
                           lambda_tensor[num_j], op12_Jx, 2, peps_parameters,
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
                           lambda_tensor[num_j], op12_Jx, 2, peps_parameters,
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
                           lambda_tensor[num_j], op12_Jy, 1, peps_parameters,
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
                           lambda_tensor[num_j], op12_Jy, 1, peps_parameters,
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
    for (int i = 0; i < 9; ++i) {
      exp_s[i] = exp(-local_parameters.tau_full * s[i] * Jx);
    }
    Us = U;
    Us.multiply_vector(exp_s, 1);

    op12_Jx = reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(3, 3, 3, 3))
                  .transpose(Axes(2, 3, 0, 1));
    for (int i = 0; i < 9; ++i) {
      exp_s[i] = exp(-0.5 * local_parameters.tau_full * s[i] * Jx);
    }
    Us = U;
    Us.multiply_vector(exp_s, 1);

    op12_Jx_2 = reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(3, 3, 3, 3))
                    .transpose(Axes(2, 3, 0, 1));

    for (int i = 0; i < 9; ++i) {
      exp_s[i] = exp(-local_parameters.tau_full * s[i] * Jy);
    }
    Us = U;
    Us.multiply_vector(exp_s, 1);

    op12_Jy = reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(3, 3, 3, 3))
                  .transpose(Axes(2, 3, 0, 1));
    for (int i = 0; i < 9; ++i) {
      exp_s[i] = exp(-0.5 * local_parameters.tau_full * s[i] * Jy);
    }
    Us = U;
    Us.multiply_vector(exp_s, 1);

    op12_Jy_2 = reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(3, 3, 3, 3))
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
        ofs_env << "#Befor_full " << local_parameters.theta << " "
                << count_CTM_env << std::endl;
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
          ofs_env << "#Befor_full " << local_parameters.theta << " "
                  << count_CTM_env << std::endl;
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
                         Tn[num], Tn[num_j], op12_Jx_2, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          ix = num % LX;
          ix_j = num_j % LX;
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
                         Tn[num], Tn[num_j], op12_Jx_2, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          ix = num % LX;
          ix_j = num_j % LX;
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
                         Tn[num], Tn[num_j], op12_Jy_2, 1, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          iy = num / LX;
          iy_j = num_j / LX;
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
                         Tn[num], Tn[num_j], op12_Jy, 1, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          iy = num / LX;
          iy_j = num_j / LX;
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
                         Tn[num], Tn[num_j], op12_Jy_2, 1, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          iy = num / LX;
          iy_j = num_j / LX;
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
                         Tn[num], Tn[num_j], op12_Jx_2, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          ix = num % LX;
          ix_j = num_j % LX;
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
                         Tn[num], Tn[num_j], op12_Jx_2, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          ix = num % LX;
          ix_j = num_j % LX;
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
                         Tn[num], Tn[num_j], op12_Jx, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          ix = num % LX;
          ix_j = num_j % LX;
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
                         Tn[num], Tn[num_j], op12_Jx, 2, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          ix = num % LX;
          ix_j = num_j % LX;
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
                         Tn[num], Tn[num_j], op12_Jy, 1, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          iy = num / LX;
          iy_j = num_j / LX;
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
                         Tn[num], Tn[num_j], op12_Jy, 1, peps_parameters,
                         Tn1_new, Tn2_new);
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;

        if (peps_parameters.Full_Use_FFU) {
          iy = num / LX;
          iy_j = num_j / LX;
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
        ofs_env << "#Befor_Obs " << local_parameters.theta << " "
                << count_CTM_env << std::endl;
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
        ofs_env << "#Befor_Obs " << local_parameters.theta << " "
                << count_CTM_env << std::endl;
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
        ofs_env << "#Befor_Obs " << local_parameters.theta << " "
                << count_CTM_env << std::endl;
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

  ptensor op_identity(Shape(3, 3)), op_sz(Shape(3, 3)), op_sx(Shape(3, 3));
  ptensor op_identity_12(Shape(9, 9)), op_ene_x(Shape(3, 3, 3, 3)),
      op_ene_y(Shape(3, 3, 3, 3));

  op_identity.set_value(Index(0, 0), 1.0);
  op_identity.set_value(Index(1, 1), 1.0);
  op_identity.set_value(Index(2, 2), 1.0);

  op_sx.set_value(Index(0, 1), 1.0 / sqrt(2.0));
  op_sx.set_value(Index(1, 0), 1.0 / sqrt(2.0));
  op_sx.set_value(Index(1, 2), 1.0 / sqrt(2.0));
  op_sx.set_value(Index(2, 1), 1.0 / sqrt(2.0));

  op_sz.set_value(Index(0, 0), 1.0);
  op_sz.set_value(Index(2, 2), -1.0);

  for (int i = 0; i < 9; i++) {
    op_identity_12.set_value(Index(i, i), 1.0);
  }
  op_identity_12 = reshape(op_identity_12, Shape(3, 3, 3, 3));

  std::vector<double> mz(N_UNIT), mx(N_UNIT);
  std::vector<std::vector<double> > ene(N_UNIT, std::vector<double>(2));
  for (int i = 0; i < N_UNIT; ++i) {
    mx[i] = 0.0;
    mz[i] = 0.0;
    ene[i][0] = 0.0;
    ene[i][1] = 0.0;
  }

  op_ene_x = Jx * reshape(Ham, Shape(3, 3, 3, 3));
  op_ene_y = Jy * reshape(Ham, Shape(3, 3, 3, 3));

  std::vector<double> norm(N_UNIT), norm_x(N_UNIT), norm_y(N_UNIT);
  int num_j;
  start_time = MPI_Wtime();
  if (local_parameters.Obs_calc_mag) {
    for (int i = 0; i < N_UNIT; ++i) {
      norm[i] = Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                                  eTb[i], eTl[i], Tn[i], op_identity);
      mz[i] = Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                                eTb[i], eTl[i], Tn[i], op_sz) /
              norm[i];
      mx[i] = Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                                eTb[i], eTl[i], Tn[i], op_sx) /
              norm[i];
      if (mpirank == 0) {
        std::cout << "## Mag " << local_parameters.theta << " " << i << " "
                  << " " << norm[i] << " " << mx[i] << " " << mz[i] << " "
                  << sqrt(mx[i] * mx[i] + mz[i] * mz[i]) << std::endl;
      }
    }
  }
  if (local_parameters.Obs_calc_energy) {
    for (int num = 0; num < N_UNIT; ++num) {
      num_j = lattice.NN_Tensor[num][2];

      // x direction
      norm_x[num] = Contract_two_sites_holizontal_op12(
          C1[num], C2[num_j], C3[num_j], C4[num], eTt[num], eTt[num_j],
          eTr[num_j], eTb[num_j], eTb[num], eTl[num], Tn[num], Tn[num_j],
          op_identity_12);
      ene[num][0] = Contract_two_sites_holizontal_op12(
                        C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                        eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                        Tn[num], Tn[num_j], op_ene_x) /
                    norm_x[num];

      // y direction
      num_j = lattice.NN_Tensor[num][3];

      norm_y[num] = Contract_two_sites_vertical_op12(
          C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
          eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num], Tn[num_j],
          op_identity_12);
      ene[num][1] = Contract_two_sites_vertical_op12(
                        C1[num], C2[num], C3[num_j], C4[num_j], eTt[num],
                        eTr[num], eTr[num_j], eTb[num_j], eTl[num_j], eTl[num],
                        Tn[num], Tn[num_j], op_ene_y) /
                    norm_y[num];

      if (mpirank == 0) {
        std::cout << "## Ene " << local_parameters.theta << " " << num << " "
                  << norm_x[num] << " " << norm_y[num] << " " << ene[num][0]
                  << " " << ene[num][1] << std::endl;
      }
    }
  }
  time_obs += MPI_Wtime() - start_time;

  if (mpirank == 0) {
    if (local_parameters.Obs_calc_energy) {
      std::ofstream ofs_energy_sub, ofs_energy;
      ofs_energy << std::setprecision(16);
      ofs_energy_sub << std::setprecision(16);
      if (local_parameters.Output_file_append) {
        ofs_energy.open("output_data/Energy.dat",
                        std::ios::out | std::ios::app);
        ofs_energy_sub.open("output_data/Energy_sub.dat",
                            std::ios::out | std::ios::app);
      } else {
        ofs_energy.open("output_data/Energy.dat", std::ios::out);
        ofs_energy_sub.open("output_data/Energy_sub.dat", std::ios::out);
      }

      double Energy = 0.0;
      for (int i = 0; i < N_UNIT; i++) {
        Energy += ene[i][0] + ene[i][1];
        ofs_energy_sub << local_parameters.theta << " " << i << " " << ene[i][0]
                       << " " << ene[i][1] << " " << norm_x[i] << " "
                       << norm_y[i] << std::endl;
      }
      ofs_energy_sub << std::endl;
      Energy /= N_UNIT;
      ofs_energy << local_parameters.theta << " " << Energy << std::endl;

      std::cout << "Energy per site: " << local_parameters.theta << " "
                << Energy << std::endl;
      ofs_energy.close();
      ofs_energy_sub.close();
    }
    if (local_parameters.Obs_calc_mag) {
      std::vector<double> sublatice_mag(3);
      std::ofstream ofs_mag_sub, ofs_mag;
      ofs_mag << std::setprecision(16);
      ofs_mag_sub << std::setprecision(16);
      if (local_parameters.Output_file_append) {
        ofs_mag.open("output_data/Magnetization.dat",
                     std::ios::out | std::ios::app);
        ofs_mag_sub.open("output_data/Magnetization_sub.dat",
                         std::ios::out | std::ios::app);
      } else {
        ofs_mag.open("output_data/Magnetization.dat", std::ios::out);
        ofs_mag_sub.open("output_data/Magnetization_sub.dat", std::ios::out);
      }

      for (int i = 0; i < N_UNIT / 2; i++) {
        int num = lattice.A_sub_list[i];
        sublatice_mag[0] += mx[num];
        sublatice_mag[1] += 0.0;
        sublatice_mag[2] += mz[num];

        num = lattice.B_sub_list[i];
        sublatice_mag[0] -= mx[num];
        sublatice_mag[1] -= 0.0;
        sublatice_mag[2] -= mz[num];
      }
      for (int num = 0; num < N_UNIT; ++num) {
        ofs_mag_sub << local_parameters.theta << " " << num << " " << mx[num]
                    << " " << mz[num] << " "
                    << sqrt(mx[num] * mx[num] + mz[num] * mz[num]) << " "
                    << norm[num] << std::endl;
      }
      ofs_mag_sub << std::endl;

      sublatice_mag[0] /= N_UNIT;
      sublatice_mag[1] /= N_UNIT;
      sublatice_mag[2] /= N_UNIT;

      ofs_mag << local_parameters.theta << " " << sublatice_mag[0] << " "
              << sublatice_mag[1] << " " << sublatice_mag[2] << " "
              << sqrt(sublatice_mag[0] * sublatice_mag[0] +
                      sublatice_mag[1] * sublatice_mag[1] +
                      sublatice_mag[2] * sublatice_mag[2])
              << std::endl;

      std::cout << "Sublatice Magnetization: " << local_parameters.theta << " "
                << sublatice_mag[0] << " " << sublatice_mag[1] << " "
                << sublatice_mag[2] << " "
                << sqrt(sublatice_mag[0] * sublatice_mag[0] +
                        sublatice_mag[1] * sublatice_mag[1] +
                        sublatice_mag[2] * sublatice_mag[2])
                << std::endl;

      ofs_mag.close();
      ofs_mag_sub.close();
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

    ofs_timer << local_parameters.theta << time_simple_update << " "
              << time_full_update << " " << time_env << " " << time_obs
              << std::endl;
    ofs_timer.close();
  }
  MPI_Finalize();
  return 0;
}
