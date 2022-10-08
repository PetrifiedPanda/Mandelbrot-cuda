cmake_minimum_required(VERSION 3.10.2)

set(This mandelbrot)

set(CMAKE_CXX_STANDARD 17)

project(${This} LANGUAGES CXX CUDA)

set(Sources
    Image.cpp
    mandelbrot.cu
    ImageGPU.cu
)

find_package(OpenMP)

add_library(${This} ${Sources})
set_target_properties(${This} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${This} INTERFACE ./)

if (OpenMP_CXX_FOUND)
    target_link_libraries(${This} OpenMP::OpenMP_CXX)
endif()

add_subdirectory(demo)
