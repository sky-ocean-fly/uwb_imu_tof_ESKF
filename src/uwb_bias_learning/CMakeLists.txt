cmake_minimum_required(VERSION 3.0.2)
project(uwb_bias_learning)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nlink_parser
  rosbag
  rosgraph
  rospy
  rostopic
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS geometry_msgs nlink_parser rosbag rosgraph rospy rostopic std_msgs
)

catkin_install_python(PROGRAMS
  scripts/inspect_ros_interfaces.py
  scripts/generate_pose_plan.py
  scripts/list_pose_status.py
  scripts/capture_static_pose.py
  scripts/bag_to_raw_dataset.py
  scripts/validate_raw_dataset.py
  scripts/build_pose_dataset.py
  scripts/make_splits.py
  scripts/train_baselines.py
  scripts/train_mlp.py
  scripts/evaluate.py
  scripts/export_model.py
  scripts/predict_bias.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS scripts/record_session.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY configs docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
