cmake_minimum_required(VERSION 2.4)
project(multiview)

message("msg CMAKE_PREFIX_PATH:::${CMAKE_PREFIX_PATH}")

set(OpenCV_DIR "/usr/local/share/OpenCV/OpenCVConfig.cmake")
find_package( OpenCV REQUIRED )

message(STATUS "opencv version: ${OpenCV_VERSION}")
message(STATUS "opencv include：${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv libs：${OpenCV_LIBRARIES}")

add_compile_options(-std=c++11 )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -pthread -fopenmp")

add_executable(multiview main.cpp)
add_executable(fisheyecalib fisheye_calib.cpp)
add_executable(fisheyecor fisheye_cor.cpp)
add_executable(fisheyeshow fisheye_show.cpp)

target_link_libraries( multiview ${OpenCV_LIBS} )
target_link_libraries( fisheyecalib ${OpenCV_LIBS} )
target_link_libraries( fisheyecor ${OpenCV_LIBS} )
target_link_libraries( fisheyeshow ${OpenCV_LIBS} )
