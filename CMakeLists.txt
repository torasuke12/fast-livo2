cmake_minimum_required(VERSION 3.8)
project(fast_livo2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -pthread -fexceptions)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  add_compile_options(-O3 -mtune=native -ffast-math)
  add_compile_definitions(ARM_ARCH)
else()
  add_compile_options(-O3 -mtune=native -funroll-loops)
  add_compile_definitions(X86_ARCH)
endif()
# Use the distribution's baseline instruction set when exchanging Eigen/PCL
# containers with its prebuilt libraries. Native AVX changes Eigen's allocator
# ABI: PCL VoxelGrid allocates a cloud using malloc, but a native caller may use
# handmade_aligned_free on it. Tuning is safe; changing SIMD alignment is not.

add_compile_definitions(ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

include(ProcessorCount)
ProcessorCount(PROC_COUNT)
if(PROC_COUNT GREATER 4)
  add_compile_definitions(MP_EN MP_PROC_NUM=4)
elseif(PROC_COUNT GREATER 1)
  add_compile_definitions(MP_EN MP_PROC_NUM=${PROC_COUNT})
else()
  add_compile_definitions(MP_PROC_NUM=1)
endif()

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP QUIET)
find_package(PCL REQUIRED COMPONENTS common filters io kdtree)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Sophus REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

set(FAST_LIVO2_DEPS
  cv_bridge
  geometry_msgs
  livox_ros_driver2
  nav_msgs
  pcl_conversions
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

if(TARGET Sophus::Sophus)
  set(FAST_LIVO2_SOPHUS_TARGET Sophus::Sophus)
endif()

add_library(fast_livo2_vio
  src/vio.cpp src/vio_temporal.cpp
  src/frame.cpp
  src/visual_point.cpp
)
# Image bounds and rejection checks must retain NaN/Inf semantics on ARM too.
target_compile_options(fast_livo2_vio PRIVATE -fno-fast-math)
ament_target_dependencies(fast_livo2_vio ${FAST_LIVO2_DEPS})
target_link_libraries(fast_livo2_vio ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(fast_livo2_vio ${FAST_LIVO2_SOPHUS_TARGET})

add_library(fast_livo2_xicp src/xicp_geometry.cpp src/xicp_update.cpp)
# Finite checks must also work on ARM, where the legacy targets use fast-math.
target_compile_options(fast_livo2_xicp PRIVATE -fno-fast-math)

add_library(fast_livo2_normals src/normal_consistency.cpp)
target_compile_options(fast_livo2_normals PRIVATE -fno-fast-math)
target_link_libraries(fast_livo2_normals ${PCL_LIBRARIES})

add_library(fast_livo2_diagnostics src/lio_diagnostics.cpp)
target_compile_options(fast_livo2_diagnostics PRIVATE -fno-fast-math)

add_library(fast_livo2_lio
  src/voxel_map.cpp
  src/voxel_map_initialization.cpp
  src/voxel_map_xicp.cpp
  include/ikd-Tree/ikd_Tree.cpp
)
ament_target_dependencies(fast_livo2_lio ${FAST_LIVO2_DEPS})
target_link_libraries(fast_livo2_lio fast_livo2_xicp fast_livo2_diagnostics fast_livo2_normals ${PCL_LIBRARIES})
target_link_libraries(fast_livo2_lio ${FAST_LIVO2_SOPHUS_TARGET})

add_library(fast_livo2_preprocess src/preprocess.cpp)
ament_target_dependencies(fast_livo2_preprocess ${FAST_LIVO2_DEPS})
target_link_libraries(fast_livo2_preprocess ${PCL_LIBRARIES})
target_link_libraries(fast_livo2_preprocess ${FAST_LIVO2_SOPHUS_TARGET})

add_library(fast_livo2_imu_proc src/IMU_Processing.cpp)
ament_target_dependencies(fast_livo2_imu_proc ${FAST_LIVO2_DEPS})
target_link_libraries(fast_livo2_imu_proc ${PCL_LIBRARIES})
target_link_libraries(fast_livo2_imu_proc ${FAST_LIVO2_SOPHUS_TARGET})

add_library(fast_livo2_temporal
  src/temporal_imu.cpp src/temporal_planes.cpp
  src/temporal_factors.cpp src/temporal_window.cpp src/temporal_visual.cpp src/temporal_structure.cpp src/temporal_map.cpp src/temporal_landmarks.cpp)
target_compile_options(fast_livo2_temporal PRIVATE -fno-fast-math)
target_link_libraries(fast_livo2_vio fast_livo2_temporal)

add_library(fast_livo2_reprojection src/temporal_reprojection.cpp)
target_compile_options(fast_livo2_reprojection PRIVATE -fno-fast-math)
target_link_libraries(fast_livo2_reprojection fast_livo2_temporal ${OpenCV_LIBRARIES})

add_library(fast_livo2_mapper src/LIVMapper.cpp src/LIVMapper_temporal.cpp src/LIVMapper_temporal_visual.cpp)
ament_target_dependencies(fast_livo2_mapper ${FAST_LIVO2_DEPS})
target_link_libraries(fast_livo2_mapper
  fast_livo2_vio
  fast_livo2_lio
  fast_livo2_preprocess
  fast_livo2_imu_proc
  fast_livo2_temporal
  fast_livo2_reprojection
  ${FAST_LIVO2_SOPHUS_TARGET}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fast_livo2_vio OpenMP::OpenMP_CXX)
  target_link_libraries(fast_livo2_lio OpenMP::OpenMP_CXX)
  target_link_libraries(fast_livo2_preprocess OpenMP::OpenMP_CXX)
  target_link_libraries(fast_livo2_imu_proc OpenMP::OpenMP_CXX)
  target_link_libraries(fast_livo2_mapper OpenMP::OpenMP_CXX)
endif()

add_executable(fastlivo2_mapping src/main.cpp)
ament_target_dependencies(fastlivo2_mapping ${FAST_LIVO2_DEPS})
target_link_libraries(fastlivo2_mapping fast_livo2_mapper)

# Offline reconstruction uses the same callbacks and estimator, without DDS
# delivery or playback timing affecting which measurements reach the estimator.
add_executable(fastlivo2_offline src/offline.cpp)
ament_target_dependencies(fastlivo2_offline ${FAST_LIVO2_DEPS} rosbag2_cpp)
target_link_libraries(fastlivo2_offline fast_livo2_mapper)

if(BUILD_TESTING)
  enable_testing()
  add_executable(temporal_reprojection_test tests/temporal_reprojection_test.cpp)
  target_compile_options(temporal_reprojection_test PRIVATE -fno-fast-math)
  target_link_libraries(temporal_reprojection_test fast_livo2_reprojection)
  add_test(NAME temporal_tracked_reprojection COMMAND temporal_reprojection_test)
  add_executable(pcl_binary_compatibility_test tests/pcl_binary_compatibility_test.cpp)
  target_link_libraries(pcl_binary_compatibility_test ${PCL_LIBRARIES})
  add_test(NAME pcl_binary_allocation_compatibility COMMAND pcl_binary_compatibility_test)
  add_executable(visual_patch_bounds_test tests/visual_patch_bounds_test.cpp)
  target_compile_options(visual_patch_bounds_test PRIVATE -fno-fast-math)
  ament_target_dependencies(visual_patch_bounds_test ${FAST_LIVO2_DEPS})
  target_link_libraries(visual_patch_bounds_test fast_livo2_vio fast_livo2_lio)
  add_test(NAME visual_patch_footprint COMMAND visual_patch_bounds_test)
  add_executable(point_uncertainty_test tests/point_uncertainty_test.cpp)
  target_compile_options(point_uncertainty_test PRIVATE -fno-fast-math)
  add_test(NAME point_covariance_frames COMMAND point_uncertainty_test)
  add_executable(temporal_deskew_test tests/temporal_deskew_test.cpp)
  target_compile_options(temporal_deskew_test PRIVATE -fno-fast-math)
  target_link_libraries(temporal_deskew_test fast_livo2_temporal)
  add_test(NAME temporal_raw_return_derivatives COMMAND temporal_deskew_test)
  add_executable(temporal_landmarks_test tests/temporal_landmarks_test.cpp)
  target_compile_options(temporal_landmarks_test PRIVATE -fno-fast-math)
  target_link_libraries(temporal_landmarks_test fast_livo2_temporal)
  add_test(NAME temporal_joint_planes COMMAND temporal_landmarks_test)
  add_executable(temporal_map_test tests/temporal_map_test.cpp)
  target_compile_options(temporal_map_test PRIVATE -fno-fast-math)
  target_link_libraries(temporal_map_test fast_livo2_temporal)
  add_test(NAME temporal_causal_map COMMAND temporal_map_test)
  add_executable(temporal_lio_test tests/temporal_lio_test.cpp)
  target_compile_options(temporal_lio_test PRIVATE -fno-fast-math)
  target_link_libraries(temporal_lio_test fast_livo2_temporal)
  add_test(NAME temporal_joint_estimation COMMAND temporal_lio_test)
  add_executable(temporal_visual_test tests/temporal_visual_test.cpp)
  target_compile_options(temporal_visual_test PRIVATE -fno-fast-math)
  target_link_libraries(temporal_visual_test fast_livo2_temporal)
  add_test(NAME temporal_visual_fusion COMMAND temporal_visual_test)
  add_executable(temporal_visual_map_test tests/temporal_visual_map_test.cpp)
  target_compile_options(temporal_visual_map_test PRIVATE -fno-fast-math)
  ament_target_dependencies(temporal_visual_map_test ${FAST_LIVO2_DEPS})
  target_link_libraries(temporal_visual_map_test fast_livo2_vio fast_livo2_lio)
  add_test(NAME temporal_visual_map_anchors COMMAND temporal_visual_map_test)
  add_executable(voxel_diagnostics_test tests/voxel_diagnostics_test.cpp)
  target_compile_options(voxel_diagnostics_test PRIVATE -fno-fast-math)
  ament_target_dependencies(voxel_diagnostics_test ${FAST_LIVO2_DEPS})
  target_link_libraries(voxel_diagnostics_test fast_livo2_lio)
  add_test(NAME voxel_plane_diagnostics COMMAND voxel_diagnostics_test)
  add_executable(normal_consistency_test tests/normal_consistency_test.cpp)
  target_compile_options(normal_consistency_test PRIVATE -fno-fast-math)
  ament_target_dependencies(normal_consistency_test ${FAST_LIVO2_DEPS})
  target_link_libraries(normal_consistency_test fast_livo2_lio)
  add_test(NAME source_map_normal_consistency COMMAND normal_consistency_test)
  add_executable(lio_diagnostics_test tests/lio_diagnostics_test.cpp)
  target_compile_options(lio_diagnostics_test PRIVATE -fno-fast-math)
  target_link_libraries(lio_diagnostics_test fast_livo2_diagnostics fast_livo2_xicp)
  add_test(NAME lio_diagnostics_recording COMMAND lio_diagnostics_test)
  add_executable(xicp_test tests/xicp_test.cpp)
  target_compile_options(xicp_test PRIVATE -fno-fast-math)
  target_link_libraries(xicp_test fast_livo2_xicp)
  add_test(NAME xicp_geometry_and_constrained_filter COMMAND xicp_test)
  # Compile the IMU implementation with NaN initialization to make accidental
  # reads of uninitialized Eigen vectors deterministic in this regression test.
  add_executable(imu_propagation_test
    tests/imu_propagation_test.cpp
    src/IMU_Processing.cpp
  )
  target_compile_definitions(imu_propagation_test PRIVATE EIGEN_INITIALIZE_MATRICES_BY_NAN)
  ament_target_dependencies(imu_propagation_test ${FAST_LIVO2_DEPS})
  target_link_libraries(imu_propagation_test ${PCL_LIBRARIES} ${FAST_LIVO2_SOPHUS_TARGET})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(imu_propagation_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME imu_propagation_no_new_samples COMMAND imu_propagation_test)

  add_executable(voxel_initialization_test tests/voxel_initialization_test.cpp)
  ament_target_dependencies(voxel_initialization_test ${FAST_LIVO2_DEPS})
  target_link_libraries(voxel_initialization_test fast_livo2_lio)
  add_test(NAME voxel_map_initialization COMMAND voxel_initialization_test)
endif()

install(TARGETS
  fast_livo2_reprojection
  fast_livo2_temporal
  fast_livo2_normals
  fast_livo2_diagnostics
  fast_livo2_xicp
  fast_livo2_vio
  fast_livo2_lio
  fast_livo2_preprocess
  fast_livo2_imu_proc
  fast_livo2_mapper
  fastlivo2_mapping
  fastlivo2_offline
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch rviz_cfg scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
