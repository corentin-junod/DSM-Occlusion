cmake_minimum_required(VERSION 3.20)
enable_language(CUDA)

project(
    TerrainOcclusion
    VERSION 0.1
    DESCRIPTION "Fast, ray-traced, occlusion for raster models"
    LANGUAGES CXX CUDA
)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O3 -fopenmp)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -use_fast_math>)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "native")

option(TIMING_DEBUG "Print debug times during computation" OFF)
option(DEBUG_MESSAGES "Print debug messages during computation" OFF)

find_library(LIBRARY_GDAL gdal)
if (LIBRARY_GDAL)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARY_GDAL})
else()
    message(FATAL_ERROR "Unable to find GDAL library")
endif()


if(APPLE)
  set_property(TARGET particle_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()