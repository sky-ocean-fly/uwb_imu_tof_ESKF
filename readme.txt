uwb_pose_estimation内部
uwb_imu_tof_eskf_fake_height.cpp是最新版的仿真代码
能在假高度和没有假高度之间切换，仿真没有重力观测
内部是加偏置，外部是没偏置

uwb_imu_tof_eskf_sim_node.cpp是次版
没有假高度模式

uwb_imu_tof_eskf_real_gravity_node.cpp是最新的实物代码
能在假高度和没有假高度之间切换，实物有重力观测


uwb_pose_estimation外部
uwb_imu_tof_eskf_real_node.cpp是次版实物代码
能在假高度和没有假高度之间切换，没有重力观测

运行
终端1
roscore
终端2
rosbag play --clock test_5.bag
终端3
roslaunch imu_mahony ground_distance.launch 
终端4
rosparam set /use_sim_time true
roslaunch uwb_pose_estimation uwb_imu_tof_eskf_real.launch
终端5
docker cp uwb_data:/home/yu/catkin_workspace/eskf_pose_compare_real.csv /home/yu/UWB/true_experiment