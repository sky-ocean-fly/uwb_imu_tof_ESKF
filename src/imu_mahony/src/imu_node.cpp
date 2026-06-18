#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Vector3.h>

#include <array>
#include <string>

#include "imu_mahony/imu_mahony.h"

class IMUNode {
public:
    IMUNode()
        : nh_("~"),
          filter_(1.0f, 0.01f),
          first_msg_(true) {
        nh_.param<std::string>("imu_topic", imu_topic_, std::string("/mynteye/imu/data_raw"));
        nh_.param<std::string>("rpy_topic", rpy_topic_, std::string("/imu/rpy"));
        nh_.param("kp", kp_, 1.0f);
        nh_.param("ki", ki_, 0.01f);

        filter_ = IMU_Mahony(kp_, ki_);

        sub_ = nh_.subscribe(imu_topic_, 1000, &IMUNode::imuCallback, this);
        pub_ = nh_.advertise<geometry_msgs::Vector3>(rpy_topic_, 1000);

        ROS_INFO("IMU Mahony node started.");
        ROS_INFO("Subscribing: %s", imu_topic_.c_str());
        ROS_INFO("Publishing : %s", rpy_topic_.c_str());
    }

private:
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
        const ros::Time now = msg->header.stamp;

        if (first_msg_) {
            last_time_ = now;
            first_msg_ = false;
            return;
        }

        const float dt = (now - last_time_).toSec();
        last_time_ = now;

        if (dt <= 0.0f || dt > 0.5f) {
            return;
        }

        std::array<float, 3> gyro = {
            static_cast<float>(msg->angular_velocity.x),
            static_cast<float>(msg->angular_velocity.y),
            static_cast<float>(msg->angular_velocity.z)
        };

        std::array<float, 3> acc = {
            static_cast<float>(msg->linear_acceleration.x),
            static_cast<float>(msg->linear_acceleration.y),
            static_cast<float>(msg->linear_acceleration.z)
        };

        // 如果你的 IMU 输出是“角度每秒”，这里要转成“弧度每秒”
        // for (auto &g : gyro) g *= static_cast<float>(M_PI / 180.0);

        filter_.update(gyro, acc, dt);

        geometry_msgs::Vector3 rpy;
        rpy.x = filter_.getRoll();
        rpy.y = filter_.getPitch();
        rpy.z = filter_.getYaw();

        pub_.publish(rpy);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;

    IMU_Mahony filter_;

    ros::Time last_time_;
    bool first_msg_;

    std::string imu_topic_;
    std::string rpy_topic_;
    float kp_;
    float ki_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "imu_mahony_node");
    IMUNode node;
    ros::spin();
    return 0;
}