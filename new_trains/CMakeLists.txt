cmake_minimum_required(VERSION 3.20)
project(new_trains_capture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(DaedalusSimSdk 1.1 REQUIRED CONFIG)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(new_trains_capture src/capture_dataset.cpp)
target_link_libraries(new_trains_capture PRIVATE
  DaedalusSimSdk::DaedalusSimSdk
  ${OpenCV_LIBS}
)
target_include_directories(new_trains_capture PRIVATE ${OpenCV_INCLUDE_DIRS})

if(WIN32)
  add_custom_command(TARGET new_trains_capture POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_RUNTIME_DLLS:new_trains_capture>
      $<TARGET_FILE_DIR:new_trains_capture>
    COMMAND_EXPAND_LISTS
  )

  set(NEW_TRAINS_ZLIB_DLL
      "${CMAKE_CURRENT_SOURCE_DIR}/../my_project/vcpkg_installed/x64-windows/bin/z.dll")
  if(EXISTS "${NEW_TRAINS_ZLIB_DLL}")
    add_custom_command(TARGET new_trains_capture POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NEW_TRAINS_ZLIB_DLL}"
        $<TARGET_FILE_DIR:new_trains_capture>
    )
  endif()
endif()

if(MSVC)
  target_compile_options(new_trains_capture PRIVATE /utf-8 /W4)
endif()
