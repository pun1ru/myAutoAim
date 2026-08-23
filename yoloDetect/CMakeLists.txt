cmake_minimum_required(VERSION 3.20)

project(yolo_detect VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The Windows simulator package bundled with this workspace provides SDK 1.1.
# The consumer only uses the stable TCP, Talos metadata, scene-control, and
# UDP gimbal APIs shared by the 1.1 and 1.3 simulator releases.
find_package(DaedalusSimSdk 1.1 REQUIRED CONFIG)
find_package(Eigen3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  find_path(EIGEN3_INCLUDE_DIR Eigen/Core)
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen 3.3 or later is required for whole_vehicle_tracking")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
  )
elseif(DEFINED Eigen3_VERSION AND Eigen3_VERSION VERSION_LESS 3.3)
  message(FATAL_ERROR "Eigen 3.3 or later is required for whole_vehicle_tracking")
endif()
find_package(OpenCV 4 REQUIRED COMPONENTS calib3d core highgui imgcodecs imgproc videoio)
find_package(Threads REQUIRED)

set(ONNXRUNTIME_VERSION "1.23.2")
if(WIN32)
  set(ONNXRUNTIME_PACKAGE_NAME
      "onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}")
  set(ONNXRUNTIME_ARCHIVE
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip")
else()
  set(ONNXRUNTIME_PACKAGE_NAME
      "onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
  set(ONNXRUNTIME_ARCHIVE
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz")
endif()
set(ONNXRUNTIME_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_deps")
set(ONNXRUNTIME_ARCHIVE_PATH
    "${ONNXRUNTIME_DOWNLOAD_DIR}/${ONNXRUNTIME_PACKAGE_NAME}.archive")
set(onnxruntime_archive_SOURCE_DIR
    "${ONNXRUNTIME_DOWNLOAD_DIR}/${ONNXRUNTIME_PACKAGE_NAME}")
if(NOT EXISTS "${onnxruntime_archive_SOURCE_DIR}/include/onnxruntime_cxx_api.h")
  file(MAKE_DIRECTORY "${ONNXRUNTIME_DOWNLOAD_DIR}")
  message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION}")
  file(DOWNLOAD "${ONNXRUNTIME_ARCHIVE}" "${ONNXRUNTIME_ARCHIVE_PATH}"
      TLS_VERIFY ON SHOW_PROGRESS STATUS ONNXRUNTIME_DOWNLOAD_STATUS)
  list(GET ONNXRUNTIME_DOWNLOAD_STATUS 0 ONNXRUNTIME_DOWNLOAD_RESULT)
  list(GET ONNXRUNTIME_DOWNLOAD_STATUS 1 ONNXRUNTIME_DOWNLOAD_MESSAGE)
  if(NOT ONNXRUNTIME_DOWNLOAD_RESULT EQUAL 0)
    message(FATAL_ERROR "ONNX Runtime download failed: ${ONNXRUNTIME_DOWNLOAD_MESSAGE}")
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar xf "${ONNXRUNTIME_ARCHIVE_PATH}"
    WORKING_DIRECTORY "${ONNXRUNTIME_DOWNLOAD_DIR}"
    RESULT_VARIABLE ONNXRUNTIME_EXTRACT_RESULT
  )
  if(NOT ONNXRUNTIME_EXTRACT_RESULT EQUAL 0)
    message(FATAL_ERROR "ONNX Runtime archive extraction failed")
  endif()
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${onnxruntime_archive_SOURCE_DIR}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${onnxruntime_archive_SOURCE_DIR}/lib/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_archive_SOURCE_DIR}/include"
  )
else()
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_archive_SOURCE_DIR}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_archive_SOURCE_DIR}/include"
  )
endif()

if(WIN32)
  set(YOLO_DETECT_CUDA_LIBRARY_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/../../trains/.venv/Lib/site-packages/torch/lib")
  if(NOT EXISTS "${YOLO_DETECT_CUDA_LIBRARY_DIR}/cudart64_12.dll" OR
     NOT EXISTS "${YOLO_DETECT_CUDA_LIBRARY_DIR}/cudnn64_9.dll")
    message(WARNING
        "CUDA 12/cuDNN 9 runtime DLLs were not found in ${YOLO_DETECT_CUDA_LIBRARY_DIR}; "
        "yolo_detect will default to ONNX Runtime CPU. Pass --cuda-libs after installing them to enable CUDA.")
    set(YOLO_DETECT_CUDA_LIBRARY_DIR "")
  endif()
else()
  set(YOLO_DETECT_CUDA_LIBRARY_DIR "")
endif()

add_library(ballistics STATIC
    src/ballistics/vacuum_ballistic_solver.cpp
)
target_include_directories(ballistics PUBLIC src)

add_library(armor_pose STATIC
    src/pose/armor_pose_estimator.cpp
)
target_include_directories(armor_pose PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(armor_pose PUBLIC ${OpenCV_LIBS})

add_library(coordinate_system STATIC
    src/coordinates/coordinate_frames.cpp
)
target_include_directories(coordinate_system PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(coordinate_system PUBLIC ${OpenCV_LIBS})

add_library(simulator_pose_adapter STATIC
    src/coordinates/simulator_pose_adapter.cpp
)
target_include_directories(simulator_pose_adapter PUBLIC src)
target_link_libraries(simulator_pose_adapter PUBLIC
    coordinate_system DaedalusSimSdk::DaedalusSimSdk
)

add_library(gimbal_aim STATIC
    src/control/gimbal_aim_solver.cpp
    src/control/static_target_controller.cpp
)
target_include_directories(gimbal_aim PUBLIC src)
target_link_libraries(gimbal_aim PUBLIC ballistics coordinate_system)

add_library(whole_vehicle_tracking STATIC
    src/tracking/constrained_yaw_solver.cpp
    src/tracking/whole_vehicle_ekf.cpp
    src/tracking/tracker_measurement_adapter.cpp
)
target_include_directories(whole_vehicle_tracking PUBLIC src)
target_link_libraries(whole_vehicle_tracking PUBLIC
    Eigen3::Eigen coordinate_system armor_pose
)

add_executable(yolo_detect
    src/app/main.cpp
    src/detection/yolo_pose_detector.cpp
    src/web/mjpeg_server.cpp
)
if(UNIX AND NOT APPLE)
  target_sources(yolo_detect PRIVATE src/sdk/glibc_compat.cpp)
endif()
target_include_directories(yolo_detect PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_detect PRIVATE
    DaedalusSimSdk::DaedalusSimSdk
    onnxruntime
    armor_pose
    simulator_pose_adapter
    gimbal_aim
    whole_vehicle_tracking
    ${OpenCV_LIBS}
    Threads::Threads
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
      "${CMAKE_CURRENT_SOURCE_DIR}/models/szu_best2_sim_416.onnx"
      "$<TARGET_FILE_DIR:yolo_detect>/szu_best2_sim_416.onnx"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:onnxruntime>"
      "$<TARGET_FILE_DIR:yolo_detect>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${onnxruntime_archive_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll"
      "${onnxruntime_archive_SOURCE_DIR}/lib/onnxruntime_providers_cuda.dll"
      "$<TARGET_FILE_DIR:yolo_detect>"
    COMMAND_EXPAND_LISTS
  )

  set(YOLO_DETECT_ROBOT_MODEL
      "${CMAKE_CURRENT_SOURCE_DIR}/../../trains/models/legacy_robot_detection/0526.onnx")
  if(EXISTS "${YOLO_DETECT_ROBOT_MODEL}")
    add_custom_command(TARGET yolo_detect POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${YOLO_DETECT_ROBOT_MODEL}"
        "$<TARGET_FILE_DIR:yolo_detect>/robot_armor_0526.onnx"
    )
  endif()

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

if(UNIX AND NOT APPLE)
  file(GLOB YOLO_DETECT_ONNXRUNTIME_LIBRARIES
      "${onnxruntime_archive_SOURCE_DIR}/lib/libonnxruntime*.so*")
  add_custom_command(TARGET yolo_detect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${YOLO_DETECT_ONNXRUNTIME_LIBRARIES}
      "$<TARGET_FILE_DIR:yolo_detect>"
    COMMAND_EXPAND_LISTS
  )
  foreach(YOLO_DETECT_MODEL
      "${CMAKE_CURRENT_SOURCE_DIR}/models/szu_best2_sim_416.onnx"
      "${CMAKE_CURRENT_SOURCE_DIR}/../RobotDetectionModel/Model/0526.onnx")
    if(EXISTS "${YOLO_DETECT_MODEL}")
      get_filename_component(YOLO_DETECT_MODEL_NAME
          "${YOLO_DETECT_MODEL}" NAME)
      if(YOLO_DETECT_MODEL_NAME STREQUAL "0526.onnx")
        set(YOLO_DETECT_MODEL_NAME "robot_armor_0526.onnx")
      endif()
      add_custom_command(TARGET yolo_detect POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${YOLO_DETECT_MODEL}"
          "$<TARGET_FILE_DIR:yolo_detect>/${YOLO_DETECT_MODEL_NAME}"
      )
    endif()
  endforeach()
  set_target_properties(yolo_detect PROPERTIES BUILD_RPATH "$ORIGIN")
endif()

if(MSVC)
  # Headers and tests contain UTF-8 comments and string literals. Apply the
  # encoding consistently instead of relying on the active Windows code page.
  add_compile_options(/utf-8)
  target_compile_options(ballistics PRIVATE /utf-8 /W4 /permissive- /EHsc)
  target_compile_definitions(ballistics PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(armor_pose PRIVATE /utf-8 /W4 /permissive- /EHsc)
  target_compile_definitions(armor_pose PRIVATE
      WIN32_LEAN_AND_MEAN NOMINMAX
  )
  target_compile_options(coordinate_system PRIVATE /utf-8 /W4 /permissive- /EHsc)
  target_compile_definitions(coordinate_system PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(simulator_pose_adapter PRIVATE /utf-8 /W4 /permissive- /EHsc)
  target_compile_definitions(simulator_pose_adapter PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(gimbal_aim PRIVATE /utf-8 /W4 /permissive- /EHsc)
  target_compile_definitions(gimbal_aim PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
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

  add_executable(vacuum_ballistic_test
      src/test/vacuum_ballistic_test.cpp
  )
  target_link_libraries(vacuum_ballistic_test PRIVATE ballistics)
  if(MSVC)
    target_compile_options(vacuum_ballistic_test PRIVATE
        /utf-8 /W4 /permissive- /EHsc
    )
    target_compile_definitions(vacuum_ballistic_test PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX
    )
  endif()

  add_executable(pnp_pose_test
      src/test/pnp_pose_test.cpp
  )
  target_link_libraries(pnp_pose_test PRIVATE armor_pose)

  add_executable(coordinate_aim_test
      src/test/coordinate_aim_test.cpp
  )
  target_link_libraries(coordinate_aim_test PRIVATE gimbal_aim)

  add_executable(whole_vehicle_ekf_test
      src/test/whole_vehicle_ekf_test.cpp
  )
  target_link_libraries(whole_vehicle_ekf_test PRIVATE whole_vehicle_tracking)

  add_executable(tracker_measurement_adapter_test
      src/test/tracker_measurement_adapter_test.cpp
  )

  add_executable(constrained_yaw_solver_test
      src/test/constrained_yaw_solver_test.cpp
  )
  target_link_libraries(constrained_yaw_solver_test PRIVATE
      whole_vehicle_tracking
  )
  target_link_libraries(tracker_measurement_adapter_test PRIVATE
      whole_vehicle_tracking
  )

  if(MSVC)
    target_compile_options(whole_vehicle_tracking PRIVATE
        /utf-8 /W4 /permissive- /EHsc
    )
    target_compile_definitions(whole_vehicle_tracking PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX
    )
    target_compile_options(whole_vehicle_ekf_test PRIVATE
        /utf-8 /W4 /permissive- /EHsc
    )
    target_compile_definitions(whole_vehicle_ekf_test PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX
    )
    target_compile_options(tracker_measurement_adapter_test PRIVATE
        /utf-8 /W4 /permissive- /EHsc
    )
    target_compile_definitions(tracker_measurement_adapter_test PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX
    )
  endif()
  if(MSVC)
    target_compile_options(coordinate_aim_test PRIVATE
        /utf-8 /W4 /permissive- /EHsc
    )
    target_compile_definitions(coordinate_aim_test PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX
    )
  endif()

  if(MSVC)
    target_compile_options(pnp_pose_test PRIVATE
        /utf-8 /W4 /permissive- /EHsc
    )
    target_compile_definitions(pnp_pose_test PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX
    )
  endif()

  add_test(NAME yolo_detect_help COMMAND yolo_detect --help)
  add_test(NAME ballistics_unit COMMAND vacuum_ballistic_test)
  add_test(NAME pnp_pose_unit COMMAND pnp_pose_test)
  add_test(NAME coordinate_aim_unit COMMAND coordinate_aim_test)
  add_test(NAME whole_vehicle_ekf_unit COMMAND whole_vehicle_ekf_test)
  add_test(NAME tracker_measurement_adapter_unit
      COMMAND tracker_measurement_adapter_test)
  add_test(NAME constrained_yaw_solver_unit COMMAND constrained_yaw_solver_test)
  add_test(NAME szu_best2_sim_inference
      COMMAND yolo_pose_smoke_test
        "${CMAKE_CURRENT_SOURCE_DIR}/models/szu_best2_sim_416.onnx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../sampledata/origindata/20260814_043627_frame_312511.bmp"
  )
  add_test(NAME robot_armor_0526_inference
      COMMAND yolo_pose_smoke_test
        "${CMAKE_CURRENT_SOURCE_DIR}/../RobotDetectionModel/Model/0526.onnx"
        "${CMAKE_CURRENT_SOURCE_DIR}/../sampledata/origindata/20260814_043627_frame_312511.bmp"
  )
endif()
