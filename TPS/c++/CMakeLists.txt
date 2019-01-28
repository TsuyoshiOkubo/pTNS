cmake_minimum_required(VERSION 2.8.12)
project(Transverse_Ising CXX)

file(GLOB SRC_FILES RELATIVE ${PROJECT_SOURCE_DIR} "*.cpp")
file(GLOB HEADER_FILES RELATIVE ${PROJECT_SOURCE_DIR} "*.hpp")

find_package(SCALAPACK REQUIRED)
find_package(TENSOR REQUIRED)

add_executable(Transverse_Ising ${SRC_FILES})
target_link_libraries(Transverse_Ising tensor)
