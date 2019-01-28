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
#endif

using namespace mptensor;
/* for finite hxyz, dtype must be complex */
typedef complex tensor_value_type;
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

  double Kx;
  double Ky;
  double Kz;
  double J;
  double hxyz;
  //
  bool Read_Initial;
  bool Read_Only_Tn;
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

    Kx = -1.0;
    Ky = -1.0;
    Kz = -1.0;
    J = 0.0;
    hxyz = 0.0;

    Read_Initial = false;
    Read_Only_Tn = false;
    Initial_type = 0; // random
    Random_amp = 0.001;

    second_ST = false;
    tau = 0.01;
    tau_step = 1000;

    tau_full = 0.01;
    tau_full_step = 0;

    // environment
    Env_calc_before_full = true;
    Env_calc_before_obs = true;

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
        } else if (result[0].compare("Kx") == 0) {
          std::istringstream is(result[1]);
          is >> Kx;
        } else if (result[0].compare("Ky") == 0) {
          std::istringstream is(result[1]);
          is >> Ky;
        } else if (result[0].compare("Kz") == 0) {
          std::istringstream is(result[1]);
          is >> Kz;
        } else if (result[0].compare("J") == 0) {
          std::istringstream is(result[1]);
          is >> J;
        } else if (result[0].compare("hxyz") == 0) {
          std::istringstream is(result[1]);
          is >> hxyz;
        } else if (result[0].compare("Read_Initial") == 0) {
          std::istringstream is(result[1]);
          is >> Read_Initial;
        } else if (result[0].compare("Read_Only_Tn") == 0) {
          std::istringstream is(result[1]);
          is >> Read_Only_Tn;
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

    ofs << "Kx " << Kx << std::endl;
    ofs << "Ky " << Ky << std::endl;
    ofs << "Kz " << Kz << std::endl;
    ofs << "J " << J << std::endl;
    ofs << "hxyz " << hxyz << std::endl;

    ofs << "Read_Initial " << Read_Initial << std::endl;
    ofs << "Read_Only_Tn " << Read_Only_Tn << std::endl;
    ofs << "Initial_type " << Initial_type << std::endl;
    ofs << "Random_amp " << Random_amp << std::endl;

    ofs << "second_ST " << second_ST << std::endl;
    ofs << "tau " << tau << std::endl;
    ofs << "tau_step " << tau_step << std::endl;
    ofs << "tau_full " << tau_full << std::endl;
    ofs << "tau_full_step " << tau_full_step << std::endl;

    ofs << "Env_calc_before_full " << Env_calc_before_full << std::endl;
    ofs << "Env_calc_before_obs " << Env_calc_before_obs << std::endl;

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

    std::vector<double> params_double(8);
    std::vector<int> params_int(10);

    if (irank == 0) {
      params_int[0] = random_seed_global;
      params_int[1] = Read_Initial;
      params_int[2] = Read_Only_Tn;
      params_int[3] = Initial_type;
      params_int[4] = second_ST;
      params_int[5] = tau_step;
      params_int[6] = tau_full_step;
      params_int[7] = Env_calc_before_full;
      params_int[8] = Env_calc_before_obs;
      params_int[9] = Output_file_append;

      params_double[0] = Kx;
      params_double[1] = Ky;
      params_double[2] = Kz;
      params_double[3] = J;
      params_double[4] = hxyz;
      params_double[5] = Random_amp;
      params_double[6] = tau;
      params_double[7] = tau_full;

      MPI_Bcast(&params_int.front(), 10, MPI_INT, 0, comm);
      MPI_Bcast(&params_double.front(), 8, MPI_DOUBLE, 0, comm);
    } else {
      MPI_Bcast(&params_int.front(), 10, MPI_INT, 0, comm);
      MPI_Bcast(&params_double.front(), 8, MPI_DOUBLE, 0, comm);

      random_seed_global = params_int[0];
      Read_Initial = params_int[1];
      Read_Only_Tn = params_int[2];
      Initial_type = params_int[3];
      second_ST = params_int[4];
      tau_step = params_int[5];
      tau_full_step = params_int[6];

      Kx = params_double[0];
      Ky = params_double[1];
      Kz = params_double[2];
      J = params_double[3];
      hxyz = params_double[4];
      Random_amp = params_double[5];
      tau = params_double[6];
      tau_full = params_double[7];

      Env_calc_before_full = params_int[7];
      Env_calc_before_obs = params_int[8];

      Output_file_append = params_int[9];
    }
  };
};

ptensor Set_Hamiltonian(const double hxyz, const double J, const double K,
                        const int direction) {
  ptensor Ham(Shape(4, 4));
  double factor = 0.5 / sqrt(3.0) / 3.0;

  if (direction == 0) {
    Ham.set_value(Index(0, 0), 0.25 * J - 2.0 * factor * hxyz);
    Ham.set_value(Index(0, 1), std::complex<double>(-1.0, 1.0) * factor * hxyz);
    Ham.set_value(Index(0, 2), std::complex<double>(-1.0, 1.0) * factor * hxyz);
    Ham.set_value(Index(0, 3), 0.25 * K);

    Ham.set_value(Index(1, 0),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(1, 1), -0.25 * J);
    Ham.set_value(Index(1, 2), 0.5 * J + 0.25 * K);
    Ham.set_value(Index(1, 3), std::complex<double>(-1.0, 1.0) * factor * hxyz);

    Ham.set_value(Index(2, 0),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(2, 1), 0.5 * J + 0.25 * K);
    Ham.set_value(Index(2, 2), -0.25 * J);
    Ham.set_value(Index(2, 3), std::complex<double>(-1.0, 1.0) * factor * hxyz);

    Ham.set_value(Index(3, 0), 0.25 * K);
    Ham.set_value(Index(3, 1),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(3, 2),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(3, 3), 0.25 * J + 2.0 * factor * hxyz);
  } else if (direction == 1) {
    Ham.set_value(Index(0, 0), 0.25 * J - 2.0 * factor * hxyz);
    Ham.set_value(Index(0, 1), std::complex<double>(-1.0, 1.0) * factor * hxyz);
    Ham.set_value(Index(0, 2), std::complex<double>(-1.0, 1.0) * factor * hxyz);
    Ham.set_value(Index(0, 3), -0.25 * K);

    Ham.set_value(Index(1, 0),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(1, 1), -0.25 * J);
    Ham.set_value(Index(1, 2), 0.5 * J + 0.25 * K);
    Ham.set_value(Index(1, 3), std::complex<double>(-1.0, 1.0) * factor * hxyz);

    Ham.set_value(Index(2, 0),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(2, 1), 0.5 * J + 0.25 * K);
    Ham.set_value(Index(2, 2), -0.25 * J);
    Ham.set_value(Index(2, 3), std::complex<double>(-1.0, 1.0) * factor * hxyz);

    Ham.set_value(Index(3, 0), -0.25 * K);
    Ham.set_value(Index(3, 1),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(3, 2),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(3, 3), 0.25 * J + 2.0 * factor * hxyz);
  } else {
    Ham.set_value(Index(0, 0), 0.25 * J + 0.25 * K - 2.0 * factor * hxyz);
    Ham.set_value(Index(0, 1), std::complex<double>(-1.0, 1.0) * factor * hxyz);
    Ham.set_value(Index(0, 2), std::complex<double>(-1.0, 1.0) * factor * hxyz);
    Ham.set_value(Index(0, 3), 0.0);

    Ham.set_value(Index(1, 0),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(1, 1), -0.25 * J - 0.25 * K);
    Ham.set_value(Index(1, 2), 0.5 * J);
    Ham.set_value(Index(1, 3), std::complex<double>(-1.0, 1.0) * factor * hxyz);

    Ham.set_value(Index(2, 0),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(2, 1), 0.5 * J);
    Ham.set_value(Index(2, 2), -0.25 * J - 0.25 * K);
    Ham.set_value(Index(2, 3), std::complex<double>(-1.0, 1.0) * factor * hxyz);

    Ham.set_value(Index(3, 0), 0.0);
    Ham.set_value(Index(3, 1),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(3, 2),
                  std::complex<double>(-1.0, -1.0) * factor * hxyz);
    Ham.set_value(Index(3, 3), 0.25 * J + 0.25 * K + 2.0 * factor * hxyz);
  }
  return Ham;
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
  int ix, iy;
  double ran_real, ran_imag;

  // Random intial state
  for (int i = 0; i < lattice.N_UNIT; ++i) {
    for (int n = 0; n < Tn[i].local_size(); ++n) {
      index = Tn[i].global_index(n);

#ifdef DSFMT
      ran_real = dsfmt_genrand_close_open(&dsfmt) - 0.5;
      ran_imag = dsfmt_genrand_close_open(&dsfmt) - 0.5;
#else
      ran_real = dist(gen);
      ran_imag = dist(gen);
#endif
      Tn[i].set_value(index, local_parameters.Random_amp *
                                 std::complex<double>(ran_real, ran_imag));
    }
  }
  if (local_parameters.Initial_type == 1) {
    // ferro
    for (int i = 0; i < lattice.N_UNIT; ++i) {
      for (int n = 0; n < Tn[i].local_size(); ++n) {
        index = Tn[i].global_index(n);
        if (index == Index(0, 0, 0, 0, 0)) {
          Tn[i].set_value(index, 1.0);
        } else if (index == Index(0, 0, 0, 0, 1)) {
          Tn[i].set_value(index, 0.0);
        }
      }
    }
  } else if (local_parameters.Initial_type == 2) {
    // AF
    for (int i = 0; i < lattice.N_UNIT / 2; ++i) {
      int num = lattice.A_sub_list[i];
      for (int n = 0; n < Tn[num].local_size(); ++n) {
        index = Tn[num].global_index(n);
        if (index == Index(0, 0, 0, 0, 0)) {
          Tn[num].set_value(index, 1.0);
        } else if (index == Index(0, 0, 0, 0, 1)) {
          Tn[num].set_value(index, 0.0);
        }
      }
      num = lattice.B_sub_list[i];
      for (int n = 0; n < Tn[num].local_size(); ++n) {
        index = Tn[num].global_index(n);
        if (index == Index(0, 0, 0, 0, 1)) {
          Tn[num].set_value(index, 1.0);
        } else if (index == Index(0, 0, 0, 0, 0)) {
          Tn[num].set_value(index, 0.0);
        }
      }
    }
  } else if (local_parameters.Initial_type == 3) {
    // zigzag
    for (int i = 0; i < lattice.N_UNIT; ++i) {
      ix = i % lattice.LX_ori;
      for (int n = 0; n < Tn[i].local_size(); ++n) {
        index = Tn[i].global_index(n);
        if (ix % 2 == 0) {
          if (index == Index(0, 0, 0, 0, 0)) {
            Tn[i].set_value(index, 1.0);
          } else if (index == Index(0, 0, 0, 0, 1)) {
            Tn[i].set_value(index, 0.0);
          };
        } else {
          if (index == Index(0, 0, 0, 0, 0)) {
            Tn[i].set_value(index, 0.0);
          } else if (index == Index(0, 0, 0, 0, 1)) {
            Tn[i].set_value(index, 1.0);
          };
        }
      }
    }
  } else if (local_parameters.Initial_type == 4) {
    // stripy
    for (int i = 0; i < lattice.N_UNIT; ++i) {
      iy = i / lattice.LX_ori;
      for (int n = 0; n < Tn[i].local_size(); ++n) {
        index = Tn[i].global_index(n);
        if (iy % 2 == 0) {
          if (index == Index(0, 0, 0, 0, 0)) {
            Tn[i].set_value(index, 1.0);
          } else if (index == Index(0, 0, 0, 0, 1)) {
            Tn[i].set_value(index, 0.0);
          }
        } else {
          if (index == Index(0, 0, 0, 0, 0)) {
            Tn[i].set_value(index, 0.0);
          } else if (index == Index(0, 0, 0, 0, 1)) {
            Tn[i].set_value(index, 1.0);
          }
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
  double J, Kx, Ky, Kz, hxyz;

  if (mpirank == 0) {
    local_parameters.read_parameters("input.dat");
    peps_parameters.read_parameters("input.dat");
    lattice.read_parameters("input.dat");
    lattice.N_UNIT = lattice.LX_ori * lattice.LY_ori;
  }

  local_parameters.Bcast_parameters(MPI_COMM_WORLD);
  peps_parameters.Bcast_parameters(MPI_COMM_WORLD);
  lattice.Bcast_parameters(MPI_COMM_WORLD);

  lattice.set_lattice_info();
  Kx = local_parameters.Kx;
  Ky = local_parameters.Ky;
  Kz = local_parameters.Kz;

  J = local_parameters.J;
  hxyz = local_parameters.hxyz;

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
  int LX_ori = lattice.LX_ori;
  int LY_ori = lattice.LY_ori;

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
  std::vector<ptensor> C1(N_UNIT, ptensor(Shape(CHI, CHI))),
      C2(N_UNIT, ptensor(Shape(CHI, CHI))),
      C3(N_UNIT, ptensor(Shape(CHI, CHI))),
      C4(N_UNIT, ptensor(Shape(CHI, CHI)));

  std::vector<ptensor> Tn(N_UNIT);
  std::vector<ptensor> eTt(N_UNIT), eTr(N_UNIT), eTb(N_UNIT), eTl(N_UNIT);
  std::vector<std::vector<std::vector<double> > > lambda_tensor(
      N_UNIT, std::vector<std::vector<double> >(4));

  for (int i = 0; i < N_UNIT; ++i) {
    int ix = i % LX;
    int iy = i / LX;
    if ((ix + iy) % 2 == 0) {
      Tn[i] = ptensor(Shape(1, D, D, D, 2));
      eTt[i] = ptensor(Shape(CHI, CHI, D, D));
      eTr[i] = ptensor(Shape(CHI, CHI, D, D));
      eTb[i] = ptensor(Shape(CHI, CHI, D, D));
      eTl[i] = ptensor(Shape(CHI, CHI, 1, 1));
      lambda_tensor[i][0] = std::vector<double>(1);
      lambda_tensor[i][1] = std::vector<double>(D);
      lambda_tensor[i][2] = std::vector<double>(D);
      lambda_tensor[i][3] = std::vector<double>(D);
    } else {
      Tn[i] = ptensor(Shape(D, D, 1, D, 2));
      eTt[i] = ptensor(Shape(CHI, CHI, D, D));
      eTr[i] = ptensor(Shape(CHI, CHI, 1, 1));
      eTb[i] = ptensor(Shape(CHI, CHI, D, D));
      eTl[i] = ptensor(Shape(CHI, CHI, D, D));
      lambda_tensor[i][0] = std::vector<double>(D);
      lambda_tensor[i][1] = std::vector<double>(D);
      lambda_tensor[i][2] = std::vector<double>(1);
      lambda_tensor[i][3] = std::vector<double>(D);
    }
  }

  if (!local_parameters.Read_Initial) {
    Initialize_Tensors(Tn, local_parameters, lattice);

    for (int i1 = 0; i1 < N_UNIT; ++i1) {
      for (int i2 = 0; i2 < 4; ++i2) {
        for (int i3 = 0; i3 < lambda_tensor[i1][i2].size(); ++i3) {
          lambda_tensor[i1][i2][i3] = 1.0;
        }
      }
    }
  } else if (local_parameters.Read_Only_Tn) {
    for (int i = 0; i < N_UNIT; i++) {
      std::stringstream ss;
      ss << i;
      std::string filename;

      filename = "input_tensors/Tn" + ss.str();
      Tn[i].load(filename.c_str());
    }
    // extention
    if (Tn[0].shape()[1] < D) {
      for (int i = 0; i < N_UNIT; i++) {
        int ix = i % LX_ori;
        int iy = i / LX_ori;
        if ((ix + iy) % 2 == 0) {
          Tn[i] = extend(Tn[i], Shape(1, D, D, D, 2));
        } else {
          Tn[i] = extend(Tn[i], Shape(D, D, 1, D, 2));
        }
      }
    }
    for (int i1 = 0; i1 < N_UNIT; ++i1) {
      for (int i2 = 0; i2 < 4; ++i2) {
        for (int i3 = 0; i3 < lambda_tensor[i1][i2].size(); ++i3) {
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
    int D_read = Tn[0].shape()[1];
    int CHI_read = C1[0].shape()[0];
    bool D_extend_flag = false;
    bool CHI_extend_flag = false;
    if (D_read < D)
      D_extend_flag = true;
    if (CHI_read < CHI)
      CHI_extend_flag = true;

    if (D_extend_flag) {
      for (int i = 0; i < N_UNIT; i++) {
        int ix = i % LX_ori;
        int iy = i / LX_ori;
        if ((ix + iy) % 2 == 0) {
          Tn[i] = extend(Tn[i], Shape(1, D, D, D, 2));
        } else {
          Tn[i] = extend(Tn[i], Shape(D, D, 1, D, 2));
        }
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
        int ix = i % LX_ori;
        int iy = i / LX_ori;
        eTt[i] = extend(eTt[i], Shape(CHI, CHI, D, D));
        eTb[i] = extend(eTb[i], Shape(CHI, CHI, D, D));
        if ((ix + iy) % 2 == 0) {
          eTr[i] = extend(eTr[i], Shape(CHI, CHI, D, D));
          eTl[i] = extend(eTl[i], Shape(CHI, CHI, 1, 1));
        } else {
          eTr[i] = extend(eTr[i], Shape(CHI, CHI, 1, 1));
          eTl[i] = extend(eTl[i], Shape(CHI, CHI, D, D));
        }
      }
    }

    std::vector<double> lambda_load(N_UNIT * 3 * D_read);
    if (mpirank == 0) {
      std::string filename = "input_tensors/lambdas";
      std::ifstream fin(filename.c_str(), std::ifstream::binary);
      fin.read(reinterpret_cast<char *>(&lambda_load[0]),
               sizeof(double) * N_UNIT * 3 * D_read);
      fin.close();

      MPI_Bcast(&lambda_load.front(), N_UNIT * 3 * D_read, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    } else {
      MPI_Bcast(&lambda_load.front(), N_UNIT * 3 * D_read, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    }
    int num;
    for (int i1 = 0; i1 < N_UNIT; ++i1) {
      int ix = i1 % LX_ori;
      int iy = i1 / LX_ori;
      if ((ix + iy) % 2 == 0) {
        for (int i2 = 0; i2 < 3; ++i2) {
          for (int i3 = 0; i3 < D_read; ++i3) {
            num = i1 * 3 * D_read + i2 * D_read + i3;
            lambda_tensor[i1][i2 + 1][i3] = lambda_load[num];
          }
        }
      } else {
        for (int i2 = 0; i2 < 3; ++i2) {
          for (int i3 = 0; i3 < D_read; ++i3) {
            num = i1 * 3 * D_read + i2 * D_read + i3;
            if (i2 == 2) {
              lambda_tensor[i1][i2 + 1][i3] = lambda_load[num];
            } else {
              lambda_tensor[i1][i2][i3] = lambda_load[num];
            }
          }
        }
      }
    }
  }

  ptensor Ham_x = Set_Hamiltonian(hxyz, J, Kx, 0);
  ptensor Ham_y = Set_Hamiltonian(hxyz, J, Ky, 1);
  ptensor Ham_z = Set_Hamiltonian(hxyz, J, Kz, 2);

  ptensor U, Ud, Us;
  std::vector<double> s;
  int info = eigh(Ham_x, s, U);
  std::vector<double> exp_s(4);
  Ud = conj(U);
  for (int i = 0; i < 4; ++i) {
    exp_s[i] = exp(-local_parameters.tau * s[i]);
  }
  Us = U;
  Us.multiply_vector(exp_s, 1);

  ptensor op12_x =
      reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(2, 2, 2, 2))
          .transpose(Axes(2, 3, 0, 1));

  for (int i = 0; i < 4; ++i) {
    exp_s[i] = exp(-0.5 * local_parameters.tau * s[i]);
  }
  Us = U;
  Us.multiply_vector(exp_s, 1);

  ptensor op12_x_2 =
      reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(2, 2, 2, 2))
          .transpose(Axes(2, 3, 0, 1));

  info = eigh(Ham_y, s, U);
  Ud = conj(U);

  for (int i = 0; i < 4; ++i) {
    exp_s[i] = exp(-local_parameters.tau * s[i]);
  }
  Us = U;
  Us.multiply_vector(exp_s, 1);

  ptensor op12_y =
      reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(2, 2, 2, 2))
          .transpose(Axes(2, 3, 0, 1));

  for (int i = 0; i < 4; ++i) {
    exp_s[i] = exp(-0.5 * local_parameters.tau * s[i]);
  }
  Us = U;
  Us.multiply_vector(exp_s, 1);

  ptensor op12_y_2 =
      reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(2, 2, 2, 2))
          .transpose(Axes(2, 3, 0, 1));

  info = eigh(Ham_z, s, U);
  Ud = conj(U);

  for (int i = 0; i < 4; ++i) {
    exp_s[i] = exp(-local_parameters.tau * s[i]);
  }
  Us = U;
  Us.multiply_vector(exp_s, 1);

  ptensor op12_z =
      reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(2, 2, 2, 2))
          .transpose(Axes(2, 3, 0, 1));

  ptensor Tn1_new, Tn2_new;
  std::vector<double> lambda_c;

  start_time = MPI_Wtime();
  for (int int_tau = 0; int_tau < local_parameters.tau_step; ++int_tau) {
    int num, num_j;
    // simple update
    if (local_parameters.second_ST) {

      // x-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], op12_x_2, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // y-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], op12_y_2, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // z-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], op12_z, 2, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][2] = lambda_c;
        lambda_tensor[num_j][0] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // y-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], op12_y_2, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // x-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], op12_x_2, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

    } else {
      // x-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], op12_x, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // y-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], op12_y, 1, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][1] = lambda_c;
        lambda_tensor[num_j][3] = lambda_c;
        Tn[num] = Tn1_new;
        Tn[num_j] = Tn2_new;
      }

      // z-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Simple_update_bond(Tn[num], Tn[num_j], lambda_tensor[num],
                           lambda_tensor[num_j], op12_z, 2, peps_parameters,
                           Tn1_new, Tn2_new, lambda_c);
        lambda_tensor[num][2] = lambda_c;
        lambda_tensor[num_j][0] = lambda_c;
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

    info = eigh(Ham_x, s, U);
    Ud = conj(U);
    for (int i = 0; i < 4; ++i) {
      exp_s[i] = exp(-local_parameters.tau_full * s[i]);
    }
    Us = U;
    Us.multiply_vector(exp_s, 1);

    op12_x = reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(2, 2, 2, 2))
                 .transpose(Axes(2, 3, 0, 1));

    for (int i = 0; i < 4; ++i) {
      exp_s[i] = exp(-0.5 * local_parameters.tau_full * s[i]);
    }
    Us = U;
    Us.multiply_vector(exp_s, 1);

    op12_x_2 = reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(2, 2, 2, 2))
                   .transpose(Axes(2, 3, 0, 1));

    info = eigh(Ham_y, s, U);
    Ud = conj(U);

    for (int i = 0; i < 4; ++i) {
      exp_s[i] = exp(-local_parameters.tau_full * s[i]);
    }
    Us = U;
    Us.multiply_vector(exp_s, 1);

    op12_y = reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(2, 2, 2, 2))
                 .transpose(Axes(2, 3, 0, 1));

    for (int i = 0; i < 4; ++i) {
      exp_s[i] = exp(-0.5 * local_parameters.tau_full * s[i]);
    }
    Us = U;
    Us.multiply_vector(exp_s, 1);

    op12_y_2 = reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(2, 2, 2, 2))
                   .transpose(Axes(2, 3, 0, 1));

    info = eigh(Ham_z, s, U);
    Ud = conj(U);

    for (int i = 0; i < 4; ++i) {
      exp_s[i] = exp(-local_parameters.tau_full * s[i]);
    }
    Us = U;
    Us.multiply_vector(exp_s, 1);

    op12_z = reshape(tensordot(Us, Ud, Axes(1), Axes(1)), Shape(2, 2, 2, 2))
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
        ofs_env << "#Befor_full " << local_parameters.J << " "
                << local_parameters.hxyz << " " << count_CTM_env << std::endl;
        file_count_CTM_exist = true;
        ofs_env.close();
      }

    } else if (local_parameters.Env_calc_before_full) {
      {
        start_time = MPI_Wtime();
        count_CTM_env = Calc_CTM_Environment(
            C1, C2, C3, C4, eTt, eTr, eTb, eTl, Tn, peps_parameters, lattice,
            !local_parameters.Read_Initial || local_parameters.Read_Only_Tn);
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
          ofs_env << "#Befor_full " << local_parameters.J << " "
                  << local_parameters.hxyz << " " << count_CTM_env << std::endl;
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

      // x-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], op12_x_2, 1, peps_parameters,
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

      // y-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], op12_y_2, 1, peps_parameters,
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

      // z-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Full_update_bond(C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                         eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                         Tn[num], Tn[num_j], op12_z, 2, peps_parameters,
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
      // y-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], op12_y_2, 1, peps_parameters,
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

      // x-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], op12_x_2, 1, peps_parameters,
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
    } else {
      // x-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], op12_x, 1, peps_parameters,
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

      // y-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.B_sub_list[i];
        num_j = lattice.NN_Tensor[num][1];

        Full_update_bond(C4[num], C1[num_j], C2[num_j], C3[num], eTl[num],
                         eTl[num_j], eTt[num_j], eTr[num_j], eTr[num], eTb[num],
                         Tn[num], Tn[num_j], op12_y, 1, peps_parameters,
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

      // z-bond
      for (int i = 0; i < N_UNIT / 2; ++i) {
        num = lattice.A_sub_list[i];
        num_j = lattice.NN_Tensor[num][2];

        Full_update_bond(C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                         eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                         Tn[num], Tn[num_j], op12_z, 2, peps_parameters,
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
    std::vector<double> lambda_save(N_UNIT * 3 * D);
    int num;
    for (int i1 = 0; i1 < N_UNIT; ++i1) {
      int ix = i1 % LX_ori;
      int iy = i1 / LX_ori;
      if ((ix + iy) % 2 == 0) {
        for (int i2 = 0; i2 < 3; ++i2) {
          for (int i3 = 0; i3 < D; ++i3) {
            num = i1 * 3 * D + i2 * D + i3;
            lambda_save[num] = lambda_tensor[i1][i2 + 1][i3];
          }
        }
      } else {
        for (int i2 = 0; i2 < 3; ++i2) {
          for (int i3 = 0; i3 < D; ++i3) {
            num = i1 * 3 * D + i2 * D + i3;
            if (i2 == 2) {
              lambda_save[num] = lambda_tensor[i1][i2 + 1][i3];
            } else {
              lambda_save[num] = lambda_tensor[i1][i2][i3];
            }
          }
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
    if (((local_parameters.Read_Initial && !local_parameters.Read_Only_Tn) &&
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
        ofs_env << "#Befor_Obs " << local_parameters.J << " "
                << local_parameters.hxyz << " " << count_CTM_env << std::endl;
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
        ofs_env << "#Befor_Obs " << local_parameters.J << " "
                << local_parameters.hxyz << " " << count_CTM_env << std::endl;
        ofs_env.close();
      }
    }
  } else {
    if (local_parameters.tau_full_step == 0 &&
        (local_parameters.tau_step > 0 ||
         (!local_parameters.Read_Initial || local_parameters.Read_Only_Tn))) {
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
        ofs_env << "#Befor_Obs " << local_parameters.J << " "
                << local_parameters.hxyz << " " << count_CTM_env << std::endl;
        ofs_env.close();
      }
    };
  }

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

  ptensor op_identity(Shape(2, 2)), op_mz(Shape(2, 2)), op_mx(Shape(2, 2)),
      op_my(Shape(2, 2));

  op_identity.set_value(Index(0, 0), 1.0);
  op_identity.set_value(Index(1, 1), 1.0);

  op_mx.set_value(Index(0, 1), 0.5);
  op_mx.set_value(Index(1, 0), 0.5);

  op_my.set_value(Index(0, 1), std::complex<double>(0.0, 0.5));
  op_my.set_value(Index(1, 0), std::complex<double>(0.0, -0.5));

  op_mz.set_value(Index(0, 0), 0.5);
  op_mz.set_value(Index(1, 1), -0.5);

  std::vector<double> mz(N_UNIT), mx(N_UNIT), my(N_UNIT);
  std::vector<std::vector<double> > zx(N_UNIT / 2, std::vector<double>(3));
  std::vector<std::vector<double> > zy(N_UNIT / 2, std::vector<double>(3));
  std::vector<std::vector<double> > zz(N_UNIT / 2, std::vector<double>(3));
  std::vector<std::vector<double> > xx(N_UNIT / 2, std::vector<double>(3));
  std::vector<std::vector<double> > xy(N_UNIT / 2, std::vector<double>(3));
  std::vector<std::vector<double> > xz(N_UNIT / 2, std::vector<double>(3));
  std::vector<std::vector<double> > yx(N_UNIT / 2, std::vector<double>(3));
  std::vector<std::vector<double> > yy(N_UNIT / 2, std::vector<double>(3));
  std::vector<std::vector<double> > yz(N_UNIT / 2, std::vector<double>(3));

  std::vector<tensor_value_type> norm(N_UNIT), norm_x(N_UNIT / 2),
      norm_y(N_UNIT / 2), norm_z(N_UNIT / 2);
  int num_j;
  start_time = MPI_Wtime();
  for (int i = 0; i < N_UNIT; ++i) {
    norm[i] = Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                                eTb[i], eTl[i], Tn[i], op_identity);
    mz[i] = (Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                               eTb[i], eTl[i], Tn[i], op_mz) /
             norm[i])
                .real();
    mx[i] = (Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                               eTb[i], eTl[i], Tn[i], op_mx) /
             norm[i])
                .real();
    my[i] = (Contract_one_site(C1[i], C2[i], C3[i], C4[i], eTt[i], eTr[i],
                               eTb[i], eTl[i], Tn[i], op_my) /
             norm[i])
                .real();
    if (mpirank == 0) {
      std::cout << "## Mag " << local_parameters.J << " "
                << local_parameters.hxyz << " " << i << " "
                << " " << norm[i] << " " << mx[i] << " " << my[i] << " "
                << mz[i] << " "
                << sqrt(mx[i] * mx[i] + my[i] * my[i] + mz[i] * mz[i])
                << std::endl;
    }
  }
  for (int i = 0; i < N_UNIT / 2; ++i) {
    int num = lattice.A_sub_list[i];
    num_j = lattice.NN_Tensor[num][1];
    // x bond
    norm_x[i] = Contract_two_sites_vertical(
        C1[num_j], C2[num_j], C3[num], C4[num], eTt[num_j], eTr[num_j],
        eTr[num], eTb[num], eTl[num], eTl[num_j], Tn[num_j], Tn[num],
        op_identity, op_identity);

    zx[i][0] = (Contract_two_sites_vertical(
                    C1[num_j], C2[num_j], C3[num], C4[num], eTt[num_j],
                    eTr[num_j], eTr[num], eTb[num], eTl[num], eTl[num_j],
                    Tn[num_j], Tn[num], op_mz, op_mx) /
                norm_x[i])
                   .real();
    zy[i][0] = (Contract_two_sites_vertical(
                    C1[num_j], C2[num_j], C3[num], C4[num], eTt[num_j],
                    eTr[num_j], eTr[num], eTb[num], eTl[num], eTl[num_j],
                    Tn[num_j], Tn[num], op_mz, op_my) /
                norm_x[i])
                   .real();
    zz[i][0] = (Contract_two_sites_vertical(
                    C1[num_j], C2[num_j], C3[num], C4[num], eTt[num_j],
                    eTr[num_j], eTr[num], eTb[num], eTl[num], eTl[num_j],
                    Tn[num_j], Tn[num], op_mz, op_mz) /
                norm_x[i])
                   .real();
    xx[i][0] = (Contract_two_sites_vertical(
                    C1[num_j], C2[num_j], C3[num], C4[num], eTt[num_j],
                    eTr[num_j], eTr[num], eTb[num], eTl[num], eTl[num_j],
                    Tn[num_j], Tn[num], op_mx, op_mx) /
                norm_x[i])
                   .real();
    xy[i][0] = (Contract_two_sites_vertical(
                    C1[num_j], C2[num_j], C3[num], C4[num], eTt[num_j],
                    eTr[num_j], eTr[num], eTb[num], eTl[num], eTl[num_j],
                    Tn[num_j], Tn[num], op_mx, op_my) /
                norm_x[i])
                   .real();
    xz[i][0] = (Contract_two_sites_vertical(
                    C1[num_j], C2[num_j], C3[num], C4[num], eTt[num_j],
                    eTr[num_j], eTr[num], eTb[num], eTl[num], eTl[num_j],
                    Tn[num_j], Tn[num], op_mx, op_mz) /
                norm_x[i])
                   .real();
    yx[i][0] = (Contract_two_sites_vertical(
                    C1[num_j], C2[num_j], C3[num], C4[num], eTt[num_j],
                    eTr[num_j], eTr[num], eTb[num], eTl[num], eTl[num_j],
                    Tn[num_j], Tn[num], op_my, op_mx) /
                norm_x[i])
                   .real();
    yy[i][0] = (Contract_two_sites_vertical(
                    C1[num_j], C2[num_j], C3[num], C4[num], eTt[num_j],
                    eTr[num_j], eTr[num], eTb[num], eTl[num], eTl[num_j],
                    Tn[num_j], Tn[num], op_my, op_my) /
                norm_x[i])
                   .real();
    yz[i][0] = (Contract_two_sites_vertical(
                    C1[num_j], C2[num_j], C3[num], C4[num], eTt[num_j],
                    eTr[num_j], eTr[num], eTb[num], eTl[num], eTl[num_j],
                    Tn[num_j], Tn[num], op_my, op_mz) /
                norm_x[i])
                   .real();

    // y bond
    num_j = lattice.NN_Tensor[num][3];
    norm_y[i] = Contract_two_sites_vertical(
        C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num], eTr[num_j],
        eTb[num_j], eTl[num_j], eTl[num], Tn[num], Tn[num_j], op_identity,
        op_identity);

    zx[i][1] = (Contract_two_sites_vertical(
                    C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
                    eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num],
                    Tn[num_j], op_mz, op_mx) /
                norm_y[i])
                   .real();
    zy[i][1] = (Contract_two_sites_vertical(
                    C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
                    eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num],
                    Tn[num_j], op_mz, op_my) /
                norm_y[i])
                   .real();
    zz[i][1] = (Contract_two_sites_vertical(
                    C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
                    eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num],
                    Tn[num_j], op_mz, op_mz) /
                norm_y[i])
                   .real();
    xx[i][1] = (Contract_two_sites_vertical(
                    C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
                    eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num],
                    Tn[num_j], op_mx, op_mx) /
                norm_y[i])
                   .real();
    xy[i][1] = (Contract_two_sites_vertical(
                    C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
                    eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num],
                    Tn[num_j], op_mx, op_my) /
                norm_y[i])
                   .real();
    xz[i][1] = (Contract_two_sites_vertical(
                    C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
                    eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num],
                    Tn[num_j], op_mx, op_mz) /
                norm_y[i])
                   .real();
    yx[i][1] = (Contract_two_sites_vertical(
                    C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
                    eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num],
                    Tn[num_j], op_my, op_mx) /
                norm_y[i])
                   .real();
    yy[i][1] = (Contract_two_sites_vertical(
                    C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
                    eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num],
                    Tn[num_j], op_my, op_my) /
                norm_y[i])
                   .real();
    yz[i][1] = (Contract_two_sites_vertical(
                    C1[num], C2[num], C3[num_j], C4[num_j], eTt[num], eTr[num],
                    eTr[num_j], eTb[num_j], eTl[num_j], eTl[num], Tn[num],
                    Tn[num_j], op_my, op_mz) /
                norm_y[i])
                   .real();

    // z bond
    num_j = lattice.NN_Tensor[num][2];
    norm_z[i] = Contract_two_sites_holizontal(
        C1[num], C2[num_j], C3[num_j], C4[num], eTt[num], eTt[num_j],
        eTr[num_j], eTb[num_j], eTb[num], eTl[num], Tn[num], Tn[num_j],
        op_identity, op_identity);

    zx[i][2] = (Contract_two_sites_holizontal(
                    C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                    eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                    Tn[num], Tn[num_j], op_mz, op_mx) /
                norm_z[i])
                   .real();
    zy[i][2] = (Contract_two_sites_holizontal(
                    C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                    eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                    Tn[num], Tn[num_j], op_mz, op_my) /
                norm_z[i])
                   .real();
    zz[i][2] = (Contract_two_sites_holizontal(
                    C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                    eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                    Tn[num], Tn[num_j], op_mz, op_mz) /
                norm_z[i])
                   .real();
    xx[i][2] = (Contract_two_sites_holizontal(
                    C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                    eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                    Tn[num], Tn[num_j], op_mx, op_mx) /
                norm_z[i])
                   .real();
    xy[i][2] = (Contract_two_sites_holizontal(
                    C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                    eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                    Tn[num], Tn[num_j], op_mx, op_my) /
                norm_z[i])
                   .real();
    xz[i][2] = (Contract_two_sites_holizontal(
                    C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                    eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                    Tn[num], Tn[num_j], op_mx, op_mz) /
                norm_z[i])
                   .real();
    yx[i][2] = (Contract_two_sites_holizontal(
                    C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                    eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                    Tn[num], Tn[num_j], op_my, op_mx) /
                norm_z[i])
                   .real();
    yy[i][2] = (Contract_two_sites_holizontal(
                    C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                    eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                    Tn[num], Tn[num_j], op_my, op_my) /
                norm_z[i])
                   .real();
    yz[i][2] = (Contract_two_sites_holizontal(
                    C1[num], C2[num_j], C3[num_j], C4[num], eTt[num],
                    eTt[num_j], eTr[num_j], eTb[num_j], eTb[num], eTl[num],
                    Tn[num], Tn[num_j], op_my, op_mz) /
                norm_z[i])
                   .real();

    if (mpirank == 0) {
      std::cout << "## Dot " << local_parameters.J << " "
                << local_parameters.hxyz << " " << i << " " << norm_x[i] << " "
                << norm_y[i] << " " << norm_z[i] << " " << xx[i][0] << " "
                << yy[i][0] << " " << zz[i][0] << " " << xx[i][1] << " "
                << yy[i][1] << " " << zz[i][1] << " " << xx[i][2] << " "
                << yy[i][2] << " " << zz[i][2] << std::endl;
      std::cout << "## Dot Off " << local_parameters.J << " "
                << local_parameters.hxyz << " " << i << " " << norm_x[i] << " "
                << norm_y[i] << " " << norm_z[i] << " " << xy[i][0] << " "
                << xz[i][0] << " " << yx[i][0] << " " << yz[i][0] << " "
                << zx[i][0] << " " << zy[i][0] << " " << xy[i][1] << " "
                << xz[i][1] << " " << yx[i][1] << " " << yz[i][1] << " "
                << zx[i][1] << " " << zy[i][1] << " " << xy[i][2] << " "
                << xz[i][2] << " " << yx[i][2] << " " << yz[i][2] << " "
                << zx[i][2] << " " << zy[i][2] << std::endl;
    }
  }
  time_obs += MPI_Wtime() - start_time;

  if (mpirank == 0) {
    std::ofstream ofs_energy_sub, ofs_energy;
    ofs_energy << std::setprecision(16);
    ofs_energy_sub << std::setprecision(16);
    if (local_parameters.Output_file_append) {
      ofs_energy.open("output_data/Energy.dat", std::ios::out | std::ios::app);
      ofs_energy_sub.open("output_data/Energy_sub.dat",
                          std::ios::out | std::ios::app);
    } else {
      ofs_energy.open("output_data/Energy.dat", std::ios::out);
      ofs_energy_sub.open("output_data/Energy_sub.dat", std::ios::out);
    }
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

    double Energy_x = 0.0;
    double Energy_y = 0.0;
    double Energy_z = 0.0;
    double Energy_h = 0.0;
    double Energy = 0.0;
    double Mag_x = 0.0;
    double Mag_y = 0.0;
    double Mag_z = 0.0;
    double Mag_abs = 0.0;

    for (int num = 0; num < N_UNIT; num++) {
      Mag_x += mx[num];
      Mag_y += my[num];
      Mag_z += mz[num];
      Mag_abs +=
          sqrt(mx[num] * mx[num] + my[num] * my[num] + mz[num] * mz[num]);

      ofs_mag_sub << local_parameters.J << " " << local_parameters.hxyz << " "
                  << num << " " << mx[num] << " " << my[num] << " " << mz[num]
                  << " " << (mx[num] + my[num] + mz[num]) / sqrt(3.0) << " "
                  << sqrt(mx[num] * mx[num] + my[num] * my[num] +
                          mz[num] * mz[num])
                  << " " << norm[num] << std::endl;

      Energy_h += -hxyz * (mx[num] + my[num] + mz[num]) / sqrt(3.0);
    }
    ofs_mag_sub << std::endl;

    for (int i = 0; i < N_UNIT / 2; i++) {
      double Ex = Kx * xx[i][0] + J * (xx[i][0] + yy[i][0] + zz[i][0]);
      double Ey = Ky * yy[i][1] + J * (xx[i][1] + yy[i][1] + zz[i][1]);
      double Ez = Kz * zz[i][2] + J * (xx[i][2] + yy[i][2] + zz[i][2]);

      Energy_x += Ex;
      Energy_y += Ey;
      Energy_z += Ez;
      ofs_energy_sub << local_parameters.J << " " << local_parameters.hxyz
                     << " " << i << " " << Ex << " " << Ey << " " << Ez << " "
                     << norm_x[i] << " " << norm_y[i] << " " << norm_z[i]
                     << std::endl;
    }
    ofs_energy_sub << std::endl;
    Energy_x /= N_UNIT;
    Energy_y /= N_UNIT;
    Energy_z /= N_UNIT;
    Energy_h /= N_UNIT;
    Energy = Energy_x + Energy_y + Energy_z + Energy_h;

    Mag_x /= N_UNIT;
    Mag_y /= N_UNIT;
    Mag_z /= N_UNIT;
    Mag_abs /= N_UNIT;

    ofs_energy << local_parameters.J << " " << local_parameters.hxyz << " "
               << Energy << " " << Energy_x << " " << Energy_y << " "
               << Energy_z << " " << Energy_h << std::endl;

    std::cout << "Energy per site: " << local_parameters.J << " "
              << local_parameters.hxyz << " " << Energy << " " << Energy_x
              << " " << Energy_y << " " << Energy_z << " " << Energy_h
              << std::endl;

    ofs_energy.close();
    ofs_energy_sub.close();

    ofs_mag << local_parameters.J << " " << local_parameters.hxyz << " "
            << (Mag_x + Mag_y + Mag_z) / sqrt(3.0) << " " << Mag_abs << " "
            << Mag_x << " " << Mag_y << " " << Mag_z << " " << std::endl;

    std::cout << "Magnetization: " << local_parameters.J << " "
              << local_parameters.hxyz << " "
              << (Mag_x + Mag_y + Mag_z) / sqrt(3.0) << " " << Mag_abs << " "
              << Mag_x << " " << Mag_y << " " << Mag_z << " " << std::endl;

    ofs_mag.close();
    ofs_mag_sub.close();

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

    ofs_timer << local_parameters.J << " " << local_parameters.hxyz << " "
              << time_simple_update << " " << time_full_update << " "
              << time_env << " " << time_obs << std::endl;
    ofs_timer.close();
  }
  MPI_Finalize();
  return 0;
}
