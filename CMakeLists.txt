cmake_minimum_required(VERSION 3.10.2)

set(This Mandelbrot)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

project(${This} LANGUAGES CUDA C CXX)

set(Sources
    Image.cpp
    Mandelbrot.cu
    ImageGPU.cu
)

find_package(OpenMP)

add_library(${This} ${Sources})
set_target_properties(${This} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${This} INTERFACE ./)

if (OpenMP_CXX_FOUND)
    target_link_libraries(${This} OpenMP::OpenMP_CXX)
endif()

add_subdirectory(Demo)