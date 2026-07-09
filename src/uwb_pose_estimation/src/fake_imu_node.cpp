#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

#include <string>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fake_imu_node");

    ros::NodeHandle pnh("~");

    std::string topic;
    std::string frame_id;
    double rate_hz;
    double angular_velocity_x;
    double angular_velocity_y;
    double angular_velocity_z;
    double linear_acceleration_x;
    double linear_acceleration_y;
    double linear_acceleration_z;

    pnh.param<std::string>("topic", topic, "/imu_A/data_raw");
    pnh.param<std::string>("frame_id", frame_id, "imu_A");
    pnh.param<double>("rate_hz", rate_hz, 500.0);
    pnh.param<double>("angular_velocity_x", angular_velocity_x, 0.0);
    pnh.param<double>("angular_velocity_y", angular_velocity_y, 0.0);
    pnh.param<double>("angular_velocity_z", angular_velocity_z, 0.0);
    pnh.param<double>("linear_acceleration_x", linear_acceleration_x, 0.0);
    pnh.param<double>("linear_acceleration_y", linear_acceleration_y, 0.0);
    pnh.param<double>("linear_acceleration_z", linear_acceleration_z, 9.80665);

    if (rate_hz <= 0.0)
    {
        ROS_WARN("fake_imu_node: rate_hz <= 0, reset to 500 Hz");
        rate_hz = 500.0;
    }

    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<sensor_msgs::Imu>(topic, 1000);

    sensor_msgs::Imu msg;
    msg.header.frame_id = frame_id;

    msg.orientation_covariance[0] = -1.0;

    msg.angular_velocity.x = angular_velocity_x;
    msg.angular_velocity.y = angular_velocity_y;
    msg.angular_velocity.z = angular_velocity_z;

    msg.linear_acceleration.x = linear_acceleration_x;
    msg.linear_acceleration.y = linear_acceleration_y;
    msg.linear_acceleration.z = linear_acceleration_z;

    ROS_INFO("fake_imu_node publishing %s at %.3f Hz", topic.c_str(), rate_hz);
    ROS_INFO("  frame_id: %s", frame_id.c_str());
    ROS_INFO("  angular_velocity: [%.6f, %.6f, %.6f] rad/s",
             angular_velocity_x, angular_velocity_y, angular_velocity_z);
    ROS_INFO("  linear_acceleration: [%.6f, %.6f, %.6f] m/s^2",
             linear_acceleration_x, linear_acceleration_y, linear_acceleration_z);

    ros::Rate rate(rate_hz);
    while (ros::ok())
    {
        msg.header.stamp = ros::Time::now();
        pub.publish(msg);
        rate.sleep();
    }

    return 0;
}
