cmake_minimum_required(VERSION 3.18)
project(DeepLearningEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(PYBIND11_FINDPYTHON ON)
add_subdirectory(extern/pybind11)

file(GLOB_RECURSE SOURCES
        "source/*.cpp"
        "source/*.cu"
)

list(APPEND SOURCES "bindings.cpp")

pybind11_add_module(PureAttention ${SOURCES})

target_include_directories(PureAttention PRIVATE include)

target_link_libraries(PureAttention PRIVATE CUDA::cudart)


set_target_properties(PureAttention PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        PREFIX ""
)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set_property(TARGET PureAttention PROPERTY CUDA_ARCHITECTURES native)
endif()