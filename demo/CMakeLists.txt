cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

set(This mandelbrot-demo)

project(${This})

set(Sources
    main.cpp
)

add_executable(${This} ${Sources})
target_link_libraries(${This} mandelbrot)
