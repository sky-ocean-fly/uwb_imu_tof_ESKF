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
  INCLUDE_DIRS include
  LIBRARIES imu_mahony_filter
  CATKIN_DEPENDS nlink_parser
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# ---- 可复用的 IMU Mahony 滤波库 ----
add_library(imu_mahony_filter
  src/imu_mahony.cpp
)

target_link_libraries(imu_mahony_filter
  ${catkin_LIBRARIES}
)

# ---- IMU Mahony 节点 ----
add_executable(imu_node
  src/imu_node.cpp
)

target_link_libraries(imu_node
  imu_mahony_filter
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

install(TARGETS imu_mahony_filter imu_node ground_distance_node uwb_sync_recorder
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
