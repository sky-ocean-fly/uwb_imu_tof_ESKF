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

# 查找外部依赖
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs visualization_msgs nlink_parser message_filters
  DEPENDS Eigen3 Ceres
)

include_directories(
  include/uwb_pose_estimation
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

# ---- UWB 定位节点 ----
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

# ---- 保存动捕与估计算法数据节点 ---- 
add_executable(pose_compare_logger
  src/pose_compare_logger.cpp
)

target_link_libraries(pose_compare_logger
  ${catkin_LIBRARIES}
)

# ---- uwb_imu_tof的ESKF算法节点 ---- 
add_executable(uwb_imu_tof_eskf_node
  src/uwb_imu_tof_eskf_node.cpp
)

target_link_libraries(uwb_imu_tof_eskf_node
  ${catkin_LIBRARIES}
)

add_dependencies(uwb_imu_tof_eskf_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

# ---- Fake IMU publisher for platform A ----
add_executable(fake_imu_node
  src/fake_imu_node.cpp
)

target_link_libraries(fake_imu_node
  ${catkin_LIBRARIES}
)

add_dependencies(fake_imu_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

# 安装
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS fake_imu_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
