cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Eigen include directory explicitly
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen-3.4.0")

# Add source files
set(SOURCES
    dense.cpp
    activations.cpp
    losses.cpp
    network.cpp
    convolutional.cpp
)

# Add header files
set(HEADERS
    layer.hpp
    dense.hpp
    activations.hpp
    losses.hpp
    network.hpp
    convolutional.hpp
)

# Create library
add_library(neural_network ${SOURCES} ${HEADERS})

# Add include directories
target_include_directories(neural_network PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
        `${EIGEN3_INCLUDE_DIR}
)

# Add example program
add_executable(xor_example xor_example.cpp)
target_link_libraries(xor_example neural_network) 