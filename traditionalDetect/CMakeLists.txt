cmake_minimum_required(VERSION 3.16)
project(traditional_detect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(DaedalusSimSdk 1.1 REQUIRED CONFIG)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui videoio)

add_executable(traditional_detect src/main.cpp)
target_link_libraries(traditional_detect PRIVATE
  DaedalusSimSdk::DaedalusSimSdk
  ${OpenCV_LIBS}
)
target_include_directories(traditional_detect PRIVATE ${OpenCV_INCLUDE_DIRS})

if(MSVC)
  target_compile_options(traditional_detect PRIVATE /utf-8)
endif()
