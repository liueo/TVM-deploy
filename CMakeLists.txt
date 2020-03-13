cmake_minimum_required(VERSION 3.6)
project(tvm_detection)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl -pthread -lX11")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR})
SET(HOME_TVM /home/ubuntu/tvm)
SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
  message(STATUS "OpenCV imgcodecs missing")
  find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${HOME_TVM}/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dlpack/include)
add_executable(tvm_detection  tvm_runtime_pack.cc src/detection/detection.cpp)
target_link_libraries(tvm_detection    ${OpenCV_LIBS}  ${CMAKE_DL_LIBS})
