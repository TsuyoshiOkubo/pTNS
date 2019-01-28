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

# kei
# CXX = mpiFCCpx
# Random=DSFMT
# DSFMT_PATH= /home/hp120283/k00380/dSFMT
# RANDOM_OBJ = ${DSFMT_PATH}/dSFMT.o
# CXXFLAGS= -Kfast,parallel,ocl,openmp -I${DSFMT_PATH}
# LDFLAGS= -SCALAPACK -SSL2BLAMP -I${DSFMT_PATH} 



.SUFFIXES: .cpp .o

all:Kagome_Real Kagome_Complex
Kagome_Real:Model_Kagome.cpp
	${CXX} -D${Random} -DREAL_TENSOR -o Kagome_Real ${CXXFLAGS} -I${PTNS_INCLUDE_DIR} -I ${MPTENSOR_INCLUDE_DIR} Model_Kagome.cpp -L ${MPTENSOR_LIB_DIR} -lmptensor ${LDFLAGS}

Kagome_Complex:Model_Kagome.cpp
	${CXX} -D${Random}  -o Kagome_Complex ${CXXFLAGS} -I${PTNS_INCLUDE_DIR} -I ${MPTENSOR_INCLUDE_DIR} Model_Kagome.cpp -L ${MPTENSOR_LIB_DIR} -lmptensor ${LDFLAGS}

clean:
	rm Kagome_Real Kagome_Complex

