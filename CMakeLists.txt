cmake_minimum_required(VERSION 3.0)
set(CMAKE_VERBOSE_MAKEFILE on)

find_package(HDF5 REQUIRED)
find_package(pybind11 REQUIRED)

set(PYBIND11_CPP_STANDARD -std=c++17)

pybind11_add_module(simulation simulation.cpp)
target_include_directories(simulation PUBLIC ${CMAKE_PREFIX_PATH}/include)
target_compile_options(simulation PUBLIC ${HIGHFIVE_CPP_STD_FLAG})
target_link_libraries(simulation PUBLIC ${HDF5_C_LIBRARIES} -lstdc++fs)


