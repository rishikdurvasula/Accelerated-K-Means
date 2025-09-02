
cmake_minimum_required(VERSION 3.18)
project(cuda_kmeans LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(cuda_kmeans src/main.cu)
target_include_directories(cuda_kmeans PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compiler flags
target_compile_options(cuda_kmeans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas=-v>)
target_compile_options(cuda_kmeans PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)

# Set a default architecture if not provided
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
