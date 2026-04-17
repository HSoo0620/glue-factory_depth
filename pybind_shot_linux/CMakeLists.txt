cmake_minimum_required(VERSION 3.16)
project(shot_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Find dependencies ----
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(PCL 1.10 REQUIRED COMPONENTS common features filters kdtree search octree)
find_package(OpenMP)

# ---- Build pybind11 module ----
pybind11_add_module(shot_module NO_EXTRAS shot_module.cpp)

target_include_directories(shot_module PRIVATE ${PCL_INCLUDE_DIRS})
target_link_directories(shot_module PRIVATE ${PCL_LIBRARY_DIRS})
target_link_libraries(shot_module PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(shot_module PRIVATE
    NOMINMAX
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(shot_module PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- Install (optional) ----
install(TARGETS shot_module LIBRARY DESTINATION .)
