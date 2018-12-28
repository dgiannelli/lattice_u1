cmake_minimum_required(VERSION 3.0)
set(CMAKE_VERBOSE_MAKEFILE on)

find_package(HDF5 REQUIRED)
find_package(pybind11 REQUIRED)

set(PYBIND11_CPP_STANDARD -std=c++17)

pybind11_add_module(lattice lattice.cpp)
target_include_directories(lattice PUBLIC ${CMAKE_PREFIX_PATH}/include)
target_link_libraries(lattice PUBLIC ${HDF5_C_LIBRARIES})
target_compile_options(lattice PUBLIC ${HIGHFIVE_CPP_STD_FLAG})


