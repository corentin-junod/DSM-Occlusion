cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_SUPPRESS_REGENERATION true)

if (UNIX)
    find_file(CMAKE_CUDA_COMPILER nvcc /usr/local/cuda/bin/ /opt/cuda/bin/)
    set(CMAKE_CXX_COMPILER g++)
endif (UNIX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(
    DSM_Occlusion
    VERSION 0.1
    DESCRIPTION "Fast, ray-traced, occlusion for raster models"
    LANGUAGES CXX CUDA
)

enable_language(CUDA)

option(TIMING_DEBUG "Print debug times during computation" OFF)
option(DEBUG_MESSAGES "Print debug messages during computation" OFF)

# General configuration
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")
add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release")


if (MSVC) # If windows
    #target_compile_options(${PROJECT_NAME} PRIVATE -W4 -O2)

    if(NOT DEFINED GDAL_INCLUDE)
        set(GDAL_INCLUDE "C:/OSGeo4W/include")
    endif()

    if(NOT DEFINED GDAL_LIB)
        find_library(GDAL_LIB gdal.lib HINTS "C:/OSGeo4W/lib" REQUIRED)
    endif()

else() # If Linux
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O3 -fopenmp)

    if(NOT DEFINED GDAL_INCLUDE)
        set(GDAL_INCLUDE "/usr/include/gdal/")
    endif()

    if(NOT DEFINED GDAL_LIB)
        find_library(GDAL_LIB NAMES gdal libgdal.so HINTS /usr/lib /usr/local/lib REQUIRED)
    endif()
endif()


if(NOT GDAL_INCLUDE)
    message(FATAL_ERROR "Unable to find GDAL headers directory")
endif()

if(NOT GDAL_LIB)
    message(FATAL_ERROR "GDAL library not found!")
endif()

message(GDAL_INCLUDE=${GDAL_INCLUDE})
message(GDAL_LIB=${GDAL_LIB})

target_include_directories(${PROJECT_NAME} PRIVATE ${GDAL_INCLUDE})
target_link_libraries(${PROJECT_NAME} PRIVATE ${GDAL_LIB})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
                                                           