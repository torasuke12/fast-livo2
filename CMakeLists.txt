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
  add_compile_options(-O3 -mcpu=native -mtune=native -ffast-math)
  add_compile_definitions(ARM_ARCH)
else()
  add_compile_options(-O3 -march=native -mtune=native -funroll-loops)
  add_compile_definitions(X86_ARCH)
endif()

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
find_package(PCL REQUIRED COMPONENTS common filters io)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
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
  src/vio.cpp
  src/frame.cpp
  src/visual_point.cpp
)
ament_target_dependencies(fast_livo2_vio ${FAST_LIVO2_DEPS})
target_link_libraries(fast_livo2_vio ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(fast_livo2_vio ${FAST_LIVO2_SOPHUS_TARGET})

add_library(fast_livo2_lio src/voxel_map.cpp)
ament_target_dependencies(fast_livo2_lio ${FAST_LIVO2_DEPS})
target_link_libraries(fast_livo2_lio ${PCL_LIBRARIES})
target_link_libraries(fast_livo2_lio ${FAST_LIVO2_SOPHUS_TARGET})

add_library(fast_livo2_preprocess src/preprocess.cpp)
ament_target_dependencies(fast_livo2_preprocess ${FAST_LIVO2_DEPS})
target_link_libraries(fast_livo2_preprocess ${PCL_LIBRARIES})
target_link_libraries(fast_livo2_preprocess ${FAST_LIVO2_SOPHUS_TARGET})

add_library(fast_livo2_imu_proc src/IMU_Processing.cpp)
ament_target_dependencies(fast_livo2_imu_proc ${FAST_LIVO2_DEPS})
target_link_libraries(fast_livo2_imu_proc ${PCL_LIBRARIES})
target_link_libraries(fast_livo2_imu_proc ${FAST_LIVO2_SOPHUS_TARGET})

add_library(fast_livo2_mapper src/LIVMapper.cpp)
ament_target_dependencies(fast_livo2_mapper ${FAST_LIVO2_DEPS})
target_link_libraries(fast_livo2_mapper
  fast_livo2_vio
  fast_livo2_lio
  fast_livo2_preprocess
  fast_livo2_imu_proc
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

install(TARGETS
  fast_livo2_vio
  fast_livo2_lio
  fast_livo2_preprocess
  fast_livo2_imu_proc
  fast_livo2_mapper
  fastlivo2_mapping
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch rviz_cfg scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
