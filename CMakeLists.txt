#cmake_minimum_required(VERSION 2.8.12)
#project(example)

set(PYBIND11_CPP_STANDARD -std=c++17)
find_package(pybind11 REQUIRED)
pybind11_add_module(prova prova.cpp)

