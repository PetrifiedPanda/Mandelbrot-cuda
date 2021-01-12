cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

set(This Demo)

project(${This})

set(Sources
    Main.cpp
)

add_executable(${This} ${Sources})
target_link_libraries(${This} Mandelbrot)