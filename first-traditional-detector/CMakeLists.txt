cmake_minimum_required(VERSION 3.16)
project(first_traditional_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(DaedalusSimSdk 1.1 REQUIRED CONFIG)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui)

add_executable(first_traditional_detector src/main.cpp)
target_link_libraries(first_traditional_detector PRIVATE
  DaedalusSimSdk::DaedalusSimSdk
  ${OpenCV_LIBS}
)
target_include_directories(first_traditional_detector PRIVATE ${OpenCV_INCLUDE_DIRS})

if(MSVC)
  target_compile_options(first_traditional_detector PRIVATE /utf-8)
endif()
