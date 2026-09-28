cmake_minimum_required(VERSION 3.24)

project(
    gpu_particle_simulator
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(
    gpu_info
    src/main.cu
)

add_executable(
    cuda_smoke_test
    src/cuda_smoke_test.cu
)

add_executable(
    particle_sim
    src/particle_sim.cu
)