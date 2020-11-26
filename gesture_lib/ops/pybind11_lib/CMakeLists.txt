cmake_minimum_required(VERSION 2.8.12)
project(pybind11_lib)
find_package(pybind11 REQUIRED)
pybind11_add_module(lib_math lib_math.cpp)
