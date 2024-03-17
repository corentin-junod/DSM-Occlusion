cmake_minimum_required(VERSION 3.20)

set(CMAKE_SUPPRESS_REGENERATION true)

if (UNIX)
    find_file(CMAKE_CUDA_COMPILER nvcc /usr/local/cuda/bin/ /opt/cuda/bin/)
endif (UNIX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()
enable_language(CUDA)

project(
    DSM_Occlusion
    VERSION 0.1
    DESCRIPTION "Fast, ray-traced, occlusion for raster models"
    LANGUAGES CXX CUDA
)

option(TIMING_DEBUG "Print debug times during computation" OFF)
option(DEBUG_MESSAGES "Print debug messages during computation" OFF)


# General configuration
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")
add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release")


if (MSVC) # If windows
	add_compile_options(/W4 /O2)

    if(NOT DEFINED GDAL_INCLUDE_DIR)
        set(GDAL_INCLUDE_DIR "C:/OSGeo4W/include")
    endif()

    if(NOT DEFINED GDAL_LIB_DIR)
        find_library(GDAL_LIB_DIR gdal.lib HINTS "C:/OSGeo4W/lib")
    endif()


else() # If Linux
	add_compile_options(-Wall -Wextra -O3 -fopenmp)

    if(NOT DEFINED GDAL_INCLUDE_DIR)
        set(GDAL_INCLUDE_DIR "/usr/include/gdal/")
    endif()

    if(NOT DEFINED GDAL_LIB_DIR)
        find_library(GDAL_LIB_DIR gdal)
    endif()
endif()

if (GDAL_INCLUDE_DIR)
    target_include_directories(${PROJECT_NAME} PRIVATE ${GDAL_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Unable to find GDAL headers directory")
endif()

if (GDAL_LIB_DIR)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${GDAL_LIB_DIR})
else()
    message(FATAL_ERROR "Unable to find GDAL library")
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler"-use_fast_math -Wall -Wextra -O3">)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
                                                           