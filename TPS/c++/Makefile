PTNS_DIR = $(HOME)/pTNS_dev
PTNS_INCLUDE_DIR = $(PTNS_DIR)/c++

MPTENSOR_DIR = $(HOME)/mptensor
MPTENSOR_INCLUDE_DIR = $(MPTENSOR_DIR)/src
MPTENSOR_LIB_DIR = $(MPTENSOR_DIR)/src

# Psi
#CXX = mpicxx
#Random=BOOST
#Boost_PATH=/opt/MateriApps/boost/boost-1.58.0-1/
#CXXFLAGS= -O3 -xHost -openmp -parallel
#LDFLAGS = -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -mkl=parallel -I${Boost_PATH}/include -L${Boost_PATH}/lib

# sekirei
Random=CPP11
CXX = mpicxx
CXXFLAGS= -O3 -xCORE-AVX2 -qopenmp -parallel -std=c++11
LDFLAGS= -lmkl_scalapack_lp64 -lmkl_blacs_sgimpt_lp64 -mkl=parallel -lmpi 
#Random=BOOST
#Boost_PATH=/home/issp/materiapps/tool/boost/boost-1.58.0-1/
#CXXFLAGS= -O3 -xCORE-AVX2 -qopenmp -parallel 
#LDFLAGS= -lmkl_scalapack_lp64 -lmkl_blacs_sgimpt_lp64 -mkl=parallel -lmpi -I${Boost_PATH}/include -L${Boost_PATH}/lib

# Mac OS X (Macports, gcc5)
#Random=CPP11
#CXX = /opt/local/bin/mpicxx
#CXXFLAGS= -fopenmp -std=c++11
#LDFLAGS= -L/opt/local/lib -lscalapack

.SUFFIXES: .cpp .o

Transverse_Ising:Model_Transverse_Ising_Square.cpp
	${CXX} -D${Random} -o Transverse_Ising ${CXXFLAGS} -I${PTNS_INCLUDE_DIR} -I ${MPTENSOR_INCLUDE_DIR} Model_Transverse_Ising_Square.cpp -L ${MPTENSOR_LIB_DIR} -lmptensor ${LDFLAGS}
clean:
	rm Transverse_Ising

