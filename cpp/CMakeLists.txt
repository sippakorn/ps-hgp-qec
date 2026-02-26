cmake_minimum_required(VERSION 3.12)
project(pym4ri)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Find m4ri library
find_library(M4RI_LIBRARY NAMES m4ri PATHS ../m4ri/.libs REQUIRED)
find_path(M4RI_INCLUDE_DIR m4ri/m4ri.h PATHS ../m4ri REQUIRED)

# Include directories
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${M4RI_INCLUDE_DIR})

# Create Python extension module
Python3_add_library(pym4ri MODULE src/pym4ri.cpp)

# Link libraries
target_link_libraries(pym4ri PRIVATE ${M4RI_LIBRARY} m)

# Set output directory
set_target_properties(pym4ri PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build
)

# Set RPATH to find m4ri library at runtime
set_target_properties(pym4ri PROPERTIES
    BUILD_RPATH "${M4RI_LIBRARY_DIR}"
    INSTALL_RPATH "${M4RI_LIBRARY_DIR}"
)

# Create standalone test executable (for debugging with gdb)
add_executable(pym4ri_test src/pym4ri.cpp)
target_link_libraries(pym4ri_test PRIVATE ${M4RI_LIBRARY} ${Python3_LIBRARIES} m)
target_include_directories(pym4ri_test PRIVATE 
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${M4RI_INCLUDE_DIR}
)
set_target_properties(pym4ri_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build
    BUILD_RPATH "${M4RI_LIBRARY_DIR}"
    INSTALL_RPATH "${M4RI_LIBRARY_DIR}"
)
