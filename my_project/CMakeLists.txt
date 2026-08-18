cmake_minimum_required(VERSION 3.20)

project(my_project VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(DaedalusSimSdk 1.1 REQUIRED CONFIG)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)

add_library(my_project_sdk STATIC
    src/sdk/daedalus_frame_source.cpp
)
target_include_directories(my_project_sdk PUBLIC src)
target_link_libraries(my_project_sdk
    PUBLIC DaedalusSimSdk::DaedalusSimSdk
)
target_compile_definitions(my_project_sdk PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

add_executable(daedalus_image_viewer
    src/test/daedalus_image_viewer.cpp
)
target_link_libraries(daedalus_image_viewer
    PRIVATE my_project_sdk ${OpenCV_LIBS}
)
target_include_directories(daedalus_image_viewer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(daedalus_image_viewer PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

add_executable(daedalus_receive_benchmark
    src/test/daedalus_receive_benchmark.cpp
)
target_link_libraries(daedalus_receive_benchmark
    PRIVATE my_project_sdk
)
target_compile_definitions(daedalus_receive_benchmark PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

if(MSVC)
    target_compile_options(my_project_sdk PRIVATE /W4 /permissive- /EHsc)
    target_compile_options(daedalus_image_viewer PRIVATE /W4 /permissive- /EHsc)
    target_compile_options(daedalus_receive_benchmark PRIVATE /W4 /permissive- /EHsc)
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME image_viewer_help COMMAND daedalus_image_viewer --help)
    add_test(NAME receive_benchmark_help COMMAND daedalus_receive_benchmark --help)
endif()
