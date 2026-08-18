cmake_minimum_required(VERSION 3.20)

project(yolo_detect VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(DaedalusSimSdk 1.1 REQUIRED CONFIG)
find_package(OpenCV 4 REQUIRED COMPONENTS core highgui imgcodecs imgproc videoio)

include(FetchContent)
set(ONNXRUNTIME_VERSION "1.23.2")
FetchContent_Declare(onnxruntime_archive
  URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime_archive)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_IMPLIB "${onnxruntime_archive_SOURCE_DIR}/lib/onnxruntime.lib"
  IMPORTED_LOCATION "${onnxruntime_archive_SOURCE_DIR}/lib/onnxruntime.dll"
  INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_archive_SOURCE_DIR}/include"
)

set(YOLO_DETECT_CUDA_LIBRARY_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../trains/.venv/Lib/site-packages/torch/lib")
if(NOT EXISTS "${YOLO_DETECT_CUDA_LIBRARY_DIR}/cudart64_12.dll" OR
   NOT EXISTS "${YOLO_DETECT_CUDA_LIBRARY_DIR}/cudnn64_9.dll")
  message(FATAL_ERROR
      "CUDA 12/cuDNN 9 runtime DLLs were not found in ${YOLO_DETECT_CUDA_LIBRARY_DIR}")
endif()

add_executable(yolo_detect
    src/app/main.cpp
    src/detection/yolo_pose_detector.cpp
)
target_include_directories(yolo_detect PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_detect PRIVATE
    DaedalusSimSdk::DaedalusSimSdk
    onnxruntime
    ${OpenCV_LIBS}
)
target_compile_definitions(yolo_detect PRIVATE
    YOLO_DETECT_DEFAULT_CUDA_LIB_DIR="${YOLO_DETECT_CUDA_LIBRARY_DIR}"
)

if(WIN32)
  add_custom_command(TARGET yolo_detect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_RUNTIME_DLLS:yolo_detect>
      $<TARGET_FILE_DIR:yolo_detect>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/models/armor_pose.onnx"
      "$<TARGET_FILE_DIR:yolo_detect>/armor_pose.onnx"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/../RobotDetectionModel/Model/0526.onnx"
      "$<TARGET_FILE_DIR:yolo_detect>/robot_armor_0526.onnx"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:onnxruntime>"
      "$<TARGET_FILE_DIR:yolo_detect>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${onnxruntime_archive_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll"
      "${onnxruntime_archive_SOURCE_DIR}/lib/onnxruntime_providers_cuda.dll"
      "$<TARGET_FILE_DIR:yolo_detect>"
    COMMAND_EXPAND_LISTS
  )

  set(YOLO_DETECT_ZLIB_DLL
      "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/z.dll")
  if(EXISTS "${YOLO_DETECT_ZLIB_DLL}")
    add_custom_command(TARGET yolo_detect POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${YOLO_DETECT_ZLIB_DLL}"
        $<TARGET_FILE_DIR:yolo_detect>
    )
  endif()
endif()

if(MSVC)
  target_compile_options(yolo_detect PRIVATE /utf-8 /W4 /permissive- /EHsc)
  target_compile_definitions(yolo_detect PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(yolo_pose_smoke_test
      src/test/detector_smoke_test.cpp
      src/detection/yolo_pose_detector.cpp
  )
  target_include_directories(yolo_pose_smoke_test PRIVATE
      src ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(yolo_pose_smoke_test PRIVATE
      onnxruntime ${OpenCV_LIBS}
  )
  target_compile_definitions(yolo_pose_smoke_test PRIVATE
      YOLO_DETECT_DEFAULT_CUDA_LIB_DIR="${YOLO_DETECT_CUDA_LIBRARY_DIR}"
  )
  if(MSVC)
    target_compile_options(yolo_pose_smoke_test PRIVATE
        /utf-8 /W4 /permissive- /EHsc
    )
    target_compile_definitions(yolo_pose_smoke_test PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX
    )
  endif()

  add_test(NAME yolo_detect_help COMMAND yolo_detect --help)
  add_test(NAME yolo_pose_inference
      COMMAND yolo_pose_smoke_test
        "${CMAKE_CURRENT_SOURCE_DIR}/models/armor_pose.onnx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../sampledata/origindata/20260814_043627_frame_312511.bmp"
  )
  add_test(NAME robot_armor_0526_inference
      COMMAND yolo_pose_smoke_test
        "${CMAKE_CURRENT_SOURCE_DIR}/../RobotDetectionModel/Model/0526.onnx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../sampledata/origindata/20260814_043627_frame_312511.bmp"
  )
endif()
