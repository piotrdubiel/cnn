project(cnn)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set (PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set (PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(HEADERS
    ${PROJECT_INCLUDE_DIR}/network.h
    ${PROJECT_INCLUDE_DIR}/layer.h
    ${PROJECT_INCLUDE_DIR}/convolution_layer.h
    ${PROJECT_INCLUDE_DIR}/subsampling_layer.h
    ${PROJECT_INCLUDE_DIR}/hidden_layer.h
    ${PROJECT_INCLUDE_DIR}/softmax_layer.h
    ${PROJECT_INCLUDE_DIR}/weight.h
    ${PROJECT_INCLUDE_DIR}/link.h
    ${PROJECT_INCLUDE_DIR}/neuron.h
    ${PROJECT_INCLUDE_DIR}/matrix.h
    ${PROJECT_INCLUDE_DIR}/size.h
    ${PROJECT_INCLUDE_DIR}/utils.h
)

set(SOURCES
    ${HEADERS}
    ${PROJECT_SOURCE_DIR}/convolution_layer.cpp
    ${PROJECT_SOURCE_DIR}/subsampling_layer.cpp
    ${PROJECT_SOURCE_DIR}/softmax_layer.cpp
    ${PROJECT_SOURCE_DIR}/hidden_layer.cpp
    ${PROJECT_SOURCE_DIR}/weight.cpp
    ${PROJECT_SOURCE_DIR}/link.cpp
    ${PROJECT_SOURCE_DIR}/matrix.cpp
    ${PROJECT_SOURCE_DIR}/utils.cpp
    ${PROJECT_SOURCE_DIR}/functions.hpp
)

include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_INCLUDE_DIR}")

add_library(${PROJECT_NAME} SHARED ${SOURCES})
add_executable(cnn_test ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cnn_test.cpp)
add_executable(simple_test ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/tests/simple_test.cpp)
add_executable(simpler_test ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/tests/simpler_test.cpp)
