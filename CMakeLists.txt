cmake_minimum_required(VERSION 3.6)
project(tvm_detection)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl -pthread")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR})
SET(HOME_TVM /home/ubuntu/tvm)

INCLUDE_DIRECTORIES(${HOME_TVM}/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dlpack/include)

FIND_PACKAGE(JPEG)
get_filename_component(JPEG_LIB_DIRS ${JPEG_LIBRARIES} PATH)
link_directories(${JPEG_LIB_DIRS})
include_directories(${JPEG_INCLUDE_DIR})

FIND_PACKAGE(X11)
link_directories(${X11_LIB_DIRS})
include_directories(${X11_INCLUDE_DIR})  

FIND_PACKAGE(ZLIB)
FIND_PACKAGE(PNG)
link_directories(${ZLIB_LIB_DIRS})
include_directories(${ZLIB_INCLUDE_DIR})
link_directories(${PNG_LIB_DIRS})
include_directories(${PNG_INCLUDE_DIR} )
  
add_executable(tvm_detection  tvm_runtime_pack.cc src/detection/detection.cpp)
target_link_libraries(tvm_detection    ${CMAKE_DL_LIBS}  ${JPEG_LIBRARIES} ${X11_LIBRARIES} ${PNG_LIBRARIES} ${ZLIB_LIBRARIES})
