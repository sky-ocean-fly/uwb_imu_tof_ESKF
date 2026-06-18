project(imu_mahony)
#自动生成PROJECT_NAME

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  nlink_parser
)
#自动生成catkin_INCLUDE_DIRS、catkin_LIBRARIES

message(STATUS "catkin_INCLUDE_DIRS = ${catkin_INCLUDE_DIRS}")
message(STATUS "catkin_LIBRARIES = ${catkin_LIBRARIES}")
message(STATUS "catkin_EXPORTED_TARGETS = ${catkin_EXPORTED_TARGETS}")

catkin_package(
  CATKIN_DEPENDS nlink_parser
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# ---- IMU Mahony 节点 ----
add_executable(imu_node
  src/imu_node.cpp
  src/imu_mahony.cpp
)

target_link_libraries(imu_node
  ${catkin_LIBRARIES}
)

add_dependencies(imu_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

# ---- 地面距离解算节点 ----
add_executable(ground_distance_node
  src/ground_distance_node.cpp
)

target_link_libraries(ground_distance_node
  ${catkin_LIBRARIES}
)

add_dependencies(ground_distance_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

# ---- UWB 同步记录节点 ----
add_executable(uwb_sync_recorder
  src/uwb_sync_recorder.cpp
)

target_link_libraries(uwb_sync_recorder
  ${catkin_LIBRARIES}
)

add_dependencies(uwb_sync_recorder
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

install(TARGETS imu_node ground_distance_node uwb_sync_recorder
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
