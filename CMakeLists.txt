cmake_minimum_required(VERSION 3.16)
project(CRAPH VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/inc)

# Source files
set(CPU_SOURCES
    src/sparse_structs.cpp
)

set(CUDA_SOURCES
    src/kernels/vertex_centric.cu
    src/kernels/edge_centric.cu
)

# Create static library
add_library(craph STATIC ${CPU_SOURCES} ${CUDA_SOURCES})

# Set library properties
set_target_properties(craph PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link CUDA libraries
target_link_libraries(craph ${CUDA_LIBRARIES})

# Create tests
enable_testing()

# Simple test
add_executable(simple_test tests/simple_test.cpp)
target_link_libraries(simple_test craph)

# Comprehensive test
add_executable(comprehensive_test tests/test_sparse_formats.cpp)
target_link_libraries(comprehensive_test craph)

# CUDA test
add_executable(cuda_test tests/cuda_test.cu)
target_link_libraries(cuda_test craph)

# Add tests
add_test(NAME SimpleTest COMMAND simple_test)
add_test(NAME ComprehensiveTest COMMAND comprehensive_test)
add_test(NAME CudaTest COMMAND cuda_test)

# Installation
install(TARGETS craph
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY inc/
    DESTINATION include/craph
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh"
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CRAPHConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CRAPHConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CRAPHConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CRAPHConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CRAPHConfigVersion.cmake"
    DESTINATION lib/cmake/CRAPH
) 