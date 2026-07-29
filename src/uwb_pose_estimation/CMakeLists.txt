cmake_minimum_required(VERSION 3.10)
project(uwb_pose_estimation)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找ROS依赖
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  nlink_parser
  message_filters
)
find_package(imu_mahony REQUIRED)

# 查找外部依赖
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs visualization_msgs nlink_parser message_filters imu_mahony
  DEPENDS Eigen3 Ceres
)

include_directories(
  include/uwb_pose_estimation
  ${catkin_INCLUDE_DIRS}
  ${imu_mahony_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

# Original UWB optimization node.
add_executable(${PROJECT_NAME}_node
  src/uwb_localization_node.cpp
  src/node_localization.cpp
  src/pose_estimation.cpp
  src/utils.cpp
  src/config_loader.cpp
)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${CERES_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

add_dependencies(${PROJECT_NAME}_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

# Simulation version of the optimization node.
add_executable(uwb_localization_sim_node
  src/uwb_localization_sim_node.cpp
  src/node_localization.cpp
  src/pose_estimation.cpp
  src/utils.cpp
  src/config_loader.cpp
)
target_link_libraries(uwb_localization_sim_node
  ${catkin_LIBRARIES}
  ${imu_mahony_LIBRARIES}
  ${CERES_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)
add_dependencies(uwb_localization_sim_node 
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS})

# Pose comparison logger.
add_executable(pose_compare_logger src/pose_compare_logger.cpp)
target_link_libraries(pose_compare_logger ${catkin_LIBRARIES})
add_dependencies(pose_compare_logger ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Original ESKF node.
add_executable(uwb_imu_tof_eskf_node src/uwb_imu_tof_eskf_node.cpp)
target_link_libraries(uwb_imu_tof_eskf_node ${catkin_LIBRARIES})
add_dependencies(uwb_imu_tof_eskf_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Simulation version of the ESKF node.
add_executable(uwb_imu_tof_eskf_sim_node src/uwb_imu_tof_eskf_sim_node.cpp)
target_link_libraries(uwb_imu_tof_eskf_sim_node ${catkin_LIBRARIES})
add_dependencies(uwb_imu_tof_eskf_sim_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Unified deterministic simulation data generator.
add_executable(sim_data_generator_node src/sim_data_generator_node.cpp)
target_link_libraries(sim_data_generator_node ${catkin_LIBRARIES})
add_dependencies(sim_data_generator_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(uwb_imu_tof_eskf_fake_height_node
  src/uwb_imu_tof_eskf_fake_height.cpp)
target_link_libraries(uwb_imu_tof_eskf_fake_height_node ${catkin_LIBRARIES})
add_dependencies(uwb_imu_tof_eskf_fake_height_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)


# Existing fake IMU node, if retained in the package.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/fake_imu_node.cpp")
  add_executable(fake_imu_node src/fake_imu_node.cpp)
  target_link_libraries(fake_imu_node ${catkin_LIBRARIES})
  add_dependencies(fake_imu_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
endif()

install(TARGETS
  ${PROJECT_NAME}_node
  uwb_localization_sim_node
  pose_compare_logger
  uwb_imu_tof_eskf_node
  uwb_imu_tof_eskf_sim_node
  sim_data_generator_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(TARGET fake_imu_node)
  install(TARGETS fake_imu_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endif()

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
