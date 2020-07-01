cmake_minimum_required(VERSION 2.4)
project(multiview)

message("msg CMAKE_PREFIX_PATH:::${CMAKE_PREFIX_PATH}")

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/OpenCVConfig.cmake")
find_package( OpenCV 4 REQUIRED )

message(STATUS "opencv version: ${OpenCV_VERSION}")
message(STATUS "opencv include：${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv libs：${OpenCV_LIBRARIES}")

add_compile_options(-std=c++11 )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -pthread -fopenmp")

add_executable(multiview main.cpp)
target_link_libraries( multiview ${OpenCV_LIBS} )
