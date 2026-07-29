 时间顺序：
1. rosbag_to_bias_csv 一个个bag采集
run_001 run_002 run_003

2. rosbag_to_bias_csv_continuous uwb_collection_marker添加手机控制功能，一个bag能采集多个位姿，对应的转csv
run_004

3. rosbag_to_bias_samples 单帧UWB+动捕->一个数据
run_004

4. rosbag_to_bias_murp_bias_csv 使用六次多项式拟合的数据代码
run_004