/**
 * @file sim_data_generator_node.cpp
 * @brief Deterministic ROS1 simulation data generator for two relative-pose nodes.
 *
 * Frames and truth:
 *   R_A_B = R_W_A^T R_W_B
 *   p_A_AB = R_W_A^T (p_W_B - p_W_A)
 *
 * Published topics:
 *   /clock                         rosgraph_msgs/Clock
 *   /imu_A/data_raw                sensor_msgs/Imu, 100 Hz
 *   /imu_B/data_raw                sensor_msgs/Imu, 100 Hz
 *   /ground_distance               geometry_msgs/PointStamped, 30 Hz
 *   /nlink_linktrack_nodeframe2    nlink_parser/LinktrackNodeframe2
 *                                   six sequential tag messages per 20 ms cycle
 *   /sim/world_pose_A              geometry_msgs/PoseStamped, 50 Hz
 *   /sim/world_pose_B              geometry_msgs/PoseStamped, 50 Hz
 *   /sim/relative_truth_pose       geometry_msgs/PoseStamped, 50 Hz
 *   /sim/relative_truth_twist      geometry_msgs/TwistStamped, 50 Hz
 *   /sim/initial_pose              geometry_msgs/PoseStamped, latched
 */

#include <ros/ros.h>
#include <rosgraph_msgs/Clock.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nlink_parser/LinktrackNodeframe2.h>
#include <nlink_parser/LinktrackNode2.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>

#include "simulation_geometry.h"

namespace
{

constexpr double kGravity = 9.80665;
constexpr double kDegToRad = M_PI / 180.0;
constexpr double kRadToDeg = 180.0 / M_PI;
constexpr double kBaseRateHz = 300.0;
constexpr double kBaseDt = 1.0 / kBaseRateHz;
constexpr int kImuTicks = 3;   // 100 Hz
constexpr int kTruthTicks = 6; // 50 Hz
constexpr int kTofTicks = 10;  // 30 Hz
constexpr double kOmegaDifferenceDt = 1.0 / 500.0;

Eigen::Matrix3d skew(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return S;
}

Eigen::Matrix3d expSO3(const Eigen::Vector3d &phi)
{
    const double theta = phi.norm();
    const Eigen::Matrix3d W = skew(phi);
    if (theta < 1e-10)
        return Eigen::Matrix3d::Identity() + W + 0.5 * W * W;

    const double a = std::sin(theta) / theta;
    const double b = (1.0 - std::cos(theta)) / (theta * theta);
    return Eigen::Matrix3d::Identity() + a * W + b * W * W;
}

Eigen::Vector3d logSO3(const Eigen::Matrix3d &R)
{
    Eigen::AngleAxisd aa(R);
    double angle = aa.angle();
    if (!std::isfinite(angle) || angle < 1e-10)
    {
        return Eigen::Vector3d(
            0.5 * (R(2, 1) - R(1, 2)),
            0.5 * (R(0, 2) - R(2, 0)),
            0.5 * (R(1, 0) - R(0, 1)));
    }
    return angle * aa.axis();
}

Eigen::Matrix3d eulerZYX(double yaw, double pitch, double roll)
{
    return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
        .toRotationMatrix();
}

// 运动学
struct Kinematics
{
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
};

struct TruthState
{
    Kinematics A;
    Kinematics B;
    Eigen::Matrix3d R_A_B = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p_A_AB = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_A_AB = Eigen::Vector3d::Zero();
};

//无人车的轨迹，此时高度无变化，无pitch和roll
Kinematics platformA(double t)
{
    Kinematics s;
    constexpr double ax = 0.55;
    constexpr double ay = 0.40;
    constexpr double wx = 0.10;
    constexpr double wy = 0.13;

    s.p << ax * std::sin(wx * t),
           ay * (std::sin(wy * t + 0.25) - std::sin(0.25)),
           0.0;
    s.v << ax * wx * std::cos(wx * t),
           ay * wy * std::cos(wy * t + 0.25),
           0.0;
    s.a << -ax * wx * wx * std::sin(wx * t),
           -ay * wy * wy * std::sin(wy * t + 0.25),
           0.0;

    const double yaw = 8.0 * kDegToRad * std::sin(0.12 * t);
    s.R = eulerZYX(yaw, 0.0, 0.0);
    return s;
}
// ？
void relativeTranslationA(double t,
                          Eigen::Vector3d &r,
                          Eigen::Vector3d &rdot,
                          Eigen::Vector3d &rddot)
{
    r << 0.80 + 0.25 * (1.0 - std::cos(0.11 * t)),
         0.20 * std::sin(0.09 * t),
         1.70 + 0.12 * std::sin(0.14 * t);

    rdot << 0.25 * 0.11 * std::sin(0.11 * t),
            0.20 * 0.09 * std::cos(0.09 * t),
            0.12 * 0.14 * std::cos(0.14 * t);

    rddot << 0.25 * 0.11 * 0.11 * std::cos(0.11 * t),
             -0.20 * 0.09 * 0.09 * std::sin(0.09 * t),
             -0.12 * 0.14 * 0.14 * std::sin(0.14 * t);
}

//返回无人车A在时刻t的世界系姿态
Eigen::Matrix3d attitudeA(double t)
{
    return eulerZYX(8.0 * kDegToRad * std::sin(0.12 * t), 0.0, 0.0);
}

//返回无人机B在时刻t的世界系姿态
Eigen::Matrix3d attitudeB(double t)
{
    const double yaw_A = 8.0 * kDegToRad * std::sin(0.12 * t);
    const double relative_yaw = 12.0 * kDegToRad * std::sin(0.18 * t);
    const double pitch = 7.0 * kDegToRad * std::sin(0.27 * t);
    const double roll = 6.0 * kDegToRad * std::sin(0.31 * t);
    return eulerZYX(yaw_A + relative_yaw, pitch, roll);
}

TruthState evaluateTruth(double t)
{
    TruthState out;
    out.A = platformA(t);

    Eigen::Vector3d r_A, rdot_A, rddot_A;
    relativeTranslationA(t, r_A, rdot_A, rddot_A);

    const double yaw_rate = 8.0 * kDegToRad * 0.12 * std::cos(0.12 * t);
    const double yaw_acc = -8.0 * kDegToRad * 0.12 * 0.12 * std::sin(0.12 * t);
    const Eigen::Vector3d omega_A(0.0, 0.0, yaw_rate);
    const Eigen::Vector3d alpha_A(0.0, 0.0, yaw_acc);

    //计算B的世界位置
    out.B.p = out.A.p + out.A.R * r_A;
    //计算B的世界速度
    out.B.v = out.A.v + out.A.R * (rdot_A + omega_A.cross(r_A));
    //计算B的世界加速度
    out.B.a = out.A.a + out.A.R * (
        rddot_A +                           //B 相对 A 的主动加速度
        2.0 * omega_A.cross(rdot_A) +       //科里奥利加速度
        alpha_A.cross(r_A) +                //A 角加速度产生的切向加速度
        omega_A.cross(omega_A.cross(r_A))); //向心加速度
    out.B.R = attitudeB(t);

    out.R_A_B = out.A.R.transpose() * out.B.R;
    out.p_A_AB = out.A.R.transpose() * (out.B.p - out.A.p);
    out.v_A_AB = out.A.R.transpose() * (out.B.v - out.A.v);
    return out;
}

//有限差分计算本体角速度
Eigen::Vector3d bodyAngularVelocity(double t,
                                    Eigen::Matrix3d (*attitude)(double))
{
    const double t0 = std::max(0.0, t - 0.5 * kOmegaDifferenceDt);
    const double t1 = t0 + kOmegaDifferenceDt;
    const Eigen::Matrix3d R0 = attitude(t0);
    const Eigen::Matrix3d R1 = attitude(t1);
    return logSO3(R0.transpose() * R1) / kOmegaDifferenceDt;
}

//打包位姿和时间戳
void fillPose(const Eigen::Vector3d &p,
              const Eigen::Matrix3d &R,
              const ros::Time &stamp,
              const std::string &frame_id,
              geometry_msgs::PoseStamped &msg)
{
    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id;
    msg.pose.position.x = p.x();
    msg.pose.position.y = p.y();
    msg.pose.position.z = p.z();
    Eigen::Quaterniond q(R);
    q.normalize();
    msg.pose.orientation.x = q.x();
    msg.pose.orientation.y = q.y();
    msg.pose.orientation.z = q.z();
    msg.pose.orientation.w = q.w();
}

} // namespace

class SimDataGenerator
{
public:
    SimDataGenerator()
        : nh_(), pnh_("~"), rng_(20260714U)
    {
        loadParams();
        rng_.seed(random_seed_);

        clock_pub_ = nh_.advertise<rosgraph_msgs::Clock>("/clock", 10);
        imu_a_pub_ = nh_.advertise<sensor_msgs::Imu>(imu_a_topic_, 50);
        imu_b_pub_ = nh_.advertise<sensor_msgs::Imu>(imu_b_topic_, 50);
        tof_pub_ = nh_.advertise<geometry_msgs::PointStamped>(tof_topic_, 20);
        uwb_pub_ = nh_.advertise<nlink_parser::LinktrackNodeframe2>(uwb_topic_, 100);
        world_a_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(world_a_topic_, 10);
        world_b_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(world_b_topic_, 10);
        relative_truth_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(relative_truth_topic_, 10);
        relative_twist_pub_ = nh_.advertise<geometry_msgs::TwistStamped>(relative_twist_topic_, 10);
        initial_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(initial_pose_topic_, 1, true);

        wall_start_ = ros::WallTime::now();
        wall_timer_ = nh_.createWallTimer(ros::WallDuration(kBaseDt),
                                          &SimDataGenerator::wallTimerCallback,
                                          this);

        ROS_INFO("Simulation generator ready: duration=%.1f s, seed=%u", duration_sec_, random_seed_);
        ROS_INFO("Rates: IMU=100 Hz, truth=50 Hz, ToF=30 Hz, each UWB tag=50 Hz, total UWB=300 Hz");
        ROS_INFO("IMU B gyro bias: [%.6f %.6f %.6f] rad/s",
                 imu_b_gyro_bias_.x(), imu_b_gyro_bias_.y(), imu_b_gyro_bias_.z());
        ROS_INFO("IMU B accelerometer bias: [%.6f %.6f %.6f] m/s^2",
                 imu_b_acc_bias_.x(), imu_b_acc_bias_.y(), imu_b_acc_bias_.z());
        ROS_INFO("Geometry: 6 anchors (IDs 0-5), 6 tags (IDs 6-11)");
    }

private:
    void loadParams()
    {
        pnh_.param<double>("duration_sec", duration_sec_, 60.0);
        pnh_.param<double>("start_delay_sec", start_delay_sec_, 1.0);
        pnh_.param<bool>("loop", loop_, false);
        pnh_.param<bool>("shutdown_on_finish", shutdown_on_finish_, false);
        int seed = 20260714;
        pnh_.param<int>("random_seed", seed, 20260714);
        random_seed_ = static_cast<std::uint32_t>(std::max(seed, 0));

        pnh_.param<double>("sigma_gyro_density", sigma_gyro_density_, 0.002);
        pnh_.param<double>("sigma_acc_density", sigma_acc_density_, 0.020);

        double imu_b_gyro_bias_x = 0.0;
        double imu_b_gyro_bias_y = 0.0;
        double imu_b_gyro_bias_z = 0.0;
        pnh_.param<double>("imu_b_gyro_bias_x", imu_b_gyro_bias_x, 0.0);
        pnh_.param<double>("imu_b_gyro_bias_y", imu_b_gyro_bias_y, 0.0);
        pnh_.param<double>("imu_b_gyro_bias_z", imu_b_gyro_bias_z, 0.0);
        imu_b_gyro_bias_ << imu_b_gyro_bias_x,
                            imu_b_gyro_bias_y,
                            imu_b_gyro_bias_z;

        double imu_b_acc_bias_x = 0.0;
        double imu_b_acc_bias_y = 0.0;
        double imu_b_acc_bias_z = 0.0;
        pnh_.param<double>("imu_b_acc_bias_x", imu_b_acc_bias_x, 0.0);
        pnh_.param<double>("imu_b_acc_bias_y", imu_b_acc_bias_y, 0.0);
        pnh_.param<double>("imu_b_acc_bias_z", imu_b_acc_bias_z, 0.0);
        imu_b_acc_bias_ << imu_b_acc_bias_x,
                           imu_b_acc_bias_y,
                           imu_b_acc_bias_z;

        pnh_.param<double>("sigma_uwb", sigma_uwb_, 0.033);
        pnh_.param<double>("uwb_bias_min", uwb_bias_min_, 0.1);
        pnh_.param<double>("uwb_bias_max", uwb_bias_max_, 0.3);
        pnh_.param<double>("sigma_tof", sigma_tof_, 0.020);

        double init_dp_x = 0.25;
        double init_dp_y = -0.20;
        double init_dp_z = 0.15;
        pnh_.param<double>("init_dp_x", init_dp_x, 0.25);
        pnh_.param<double>("init_dp_y", init_dp_y, -0.20);
        pnh_.param<double>("init_dp_z", init_dp_z, 0.15);
        init_dp_ << init_dp_x, init_dp_y, init_dp_z;
        double droll_deg = 5.0;
        double dpitch_deg = -4.0;
        double dyaw_deg = 8.0;
        pnh_.param<double>("init_droll_deg", droll_deg, 5.0);
        pnh_.param<double>("init_dpitch_deg", dpitch_deg, -4.0);
        pnh_.param<double>("init_dyaw_deg", dyaw_deg, 8.0);
        init_dtheta_ << droll_deg * kDegToRad,
                        dpitch_deg * kDegToRad,
                        dyaw_deg * kDegToRad;

        pnh_.param<std::string>("imu_a_topic", imu_a_topic_, "/imu_A/data_raw");
        pnh_.param<std::string>("imu_b_topic", imu_b_topic_, "/imu_B/data_raw");
        pnh_.param<std::string>("tof_topic", tof_topic_, "/ground_distance");
        pnh_.param<std::string>("uwb_topic", uwb_topic_, "/nlink_linktrack_nodeframe2");
        pnh_.param<std::string>("world_a_topic", world_a_topic_, "/sim/world_pose_A");
        pnh_.param<std::string>("world_b_topic", world_b_topic_, "/sim/world_pose_B");
        pnh_.param<std::string>("relative_truth_topic", relative_truth_topic_, "/sim/relative_truth_pose");
        pnh_.param<std::string>("relative_twist_topic", relative_twist_topic_, "/sim/relative_truth_twist");
        pnh_.param<std::string>("initial_pose_topic", initial_pose_topic_, "/sim/initial_pose");
    }

    //生成 [uwb_bias_min_, uwb_bias_max_] 范围内的均匀分布偏置
    double uniformBias()
    {
        if (!(uwb_bias_max_ > uwb_bias_min_))
            return uwb_bias_min_;
        std::uniform_real_distribution<double> dist(uwb_bias_min_, uwb_bias_max_);
        return dist(rng_);
    }

    //生成高斯噪声
    double gaussian(double sigma)
    {
        if (!(sigma > 0.0))
            return 0.0;
        std::normal_distribution<double> dist(0.0, sigma);
        return dist(rng_);
    }

    //生成三维高斯噪声
    Eigen::Vector3d gaussian3(double sigma)
    {
        return Eigen::Vector3d(gaussian(sigma), gaussian(sigma), gaussian(sigma));
    }

    //发布/clock话题
    void publishClock(const ros::Time &stamp)
    {
        rosgraph_msgs::Clock msg;
        msg.clock = stamp;
        clock_pub_.publish(msg);
    }

    //发布仅一次的初始位姿估计
    void publishInitialPose(const ros::Time &stamp)
    {
        if (initial_pose_published_)
            return;

        const TruthState truth0 = evaluateTruth(0.0);
        const Eigen::Matrix3d R_init = expSO3(init_dtheta_) * truth0.R_A_B;
        const Eigen::Vector3d p_init = truth0.p_A_AB + init_dp_;

        geometry_msgs::PoseStamped msg;
        fillPose(p_init, R_init, stamp, "anchor_frame_A", msg);
        initial_pose_pub_.publish(msg);
        initial_pose_published_ = true;

        ROS_INFO("Published common initial pose: p error=[%.3f %.3f %.3f] m, dtheta=[%.2f %.2f %.2f] deg",
                 init_dp_.x(), init_dp_.y(), init_dp_.z(),
                 init_dtheta_.x() * kRadToDeg,
                 init_dtheta_.y() * kRadToDeg,
                 init_dtheta_.z() * kRadToDeg);
    }

    //发布50Hz的真值数据
    void publishTruth(double t, const ros::Time &stamp, const TruthState &truth)
    {
        geometry_msgs::PoseStamped pose_a;
        geometry_msgs::PoseStamped pose_b;
        geometry_msgs::PoseStamped pose_rel;
        fillPose(truth.A.p, truth.A.R, stamp, "world", pose_a);
        fillPose(truth.B.p, truth.B.R, stamp, "world", pose_b);
        fillPose(truth.p_A_AB, truth.R_A_B, stamp, "anchor_frame_A", pose_rel);
        world_a_pub_.publish(pose_a);
        world_b_pub_.publish(pose_b);
        relative_truth_pub_.publish(pose_rel);

        geometry_msgs::TwistStamped twist;
        twist.header.stamp = stamp;
        twist.header.frame_id = "anchor_frame_A";
        twist.twist.linear.x = truth.v_A_AB.x();
        twist.twist.linear.y = truth.v_A_AB.y();
        twist.twist.linear.z = truth.v_A_AB.z();
        const Eigen::Vector3d omega_rel_B = bodyAngularVelocity(t, attitudeB) - truth.R_A_B.transpose() * bodyAngularVelocity(t, attitudeA);
        twist.twist.angular.x = omega_rel_B.x();
        twist.twist.angular.y = omega_rel_B.y();
        twist.twist.angular.z = omega_rel_B.z();
        relative_twist_pub_.publish(twist);
    }

    //发布100Hz的IMU数据
    void publishImu(double t, const ros::Time &stamp, const TruthState &truth)
    {
        constexpr double imu_dt = 0.01;
        const double gyro_sample_sigma = sigma_gyro_density_ / std::sqrt(imu_dt);
        const double acc_sample_sigma = sigma_acc_density_ / std::sqrt(imu_dt);
        const Eigen::Vector3d gravity_W(0.0, 0.0, -kGravity);

        const Eigen::Vector3d omega_A = bodyAngularVelocity(t, attitudeA) + gaussian3(gyro_sample_sigma);
        const Eigen::Vector3d omega_B = bodyAngularVelocity(t, attitudeB) + gaussian3(gyro_sample_sigma) + imu_b_gyro_bias_;
        const Eigen::Vector3d f_A = truth.A.R.transpose() * (truth.A.a - gravity_W) + gaussian3(acc_sample_sigma);
        const Eigen::Vector3d f_B = truth.B.R.transpose() * (truth.B.a - gravity_W) + gaussian3(acc_sample_sigma) + imu_b_acc_bias_;

        sensor_msgs::Imu msg_a;
        sensor_msgs::Imu msg_b;
        fillImuMessage(stamp, "imu_A", omega_A, f_A, msg_a);
        fillImuMessage(stamp, "imu_B", omega_B, f_B, msg_b);
        imu_a_pub_.publish(msg_a);
        imu_b_pub_.publish(msg_b);
    }

    //将角速度和比力填入sensor_msgs/Imu
    void fillImuMessage(const ros::Time &stamp,
                        const std::string &frame_id,
                        const Eigen::Vector3d &omega,
                        const Eigen::Vector3d &specific_force,
                        sensor_msgs::Imu &msg)
    {
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id;
        msg.orientation.w = 1.0;
        msg.orientation_covariance[0] = -1.0;
        msg.angular_velocity.x = omega.x();
        msg.angular_velocity.y = omega.y();
        msg.angular_velocity.z = omega.z();
        msg.linear_acceleration.x = specific_force.x();
        msg.linear_acceleration.y = specific_force.y();
        msg.linear_acceleration.z = specific_force.z();
    }

    //发布30Hz的tof测距数据到/ground_distance
    void publishTof(const ros::Time &stamp, const TruthState &truth)
    {
        const auto &c_T = uwb_pose_estimation::simulation_geometry::tofPositionB();
        const auto &s_T = uwb_pose_estimation::simulation_geometry::tofRayB();
        const Eigen::Vector3d e3(0.0, 0.0, 1.0);
        const double alpha = e3.dot(truth.p_A_AB + truth.R_A_B * c_T);
        const double beta = e3.dot(truth.R_A_B * s_T);

        if (std::fabs(beta) < 1e-6 || alpha <= 0.0)
        {
            ROS_ERROR_THROTTLE(1.0, "Invalid ToF geometry: alpha=%.6f beta=%.6f", alpha, beta);
            return;
        }

        const double slant = -alpha / beta;
        geometry_msgs::PointStamped msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = "tof_B";
        msg.point.x = slant + gaussian(sigma_tof_);
        msg.point.y = alpha + gaussian(sigma_tof_);
        msg.point.z = 0.0;
        tof_pub_.publish(msg);
    }

    //每tick发布一个UWB tag的测距数据到/nlink_linktrack_nodeframe2
    void publishOneUwbTag(std::uint64_t tick, double t, 
                            const ros::Time &stamp, const TruthState &truth)
    {
        const auto &tag_ids = uwb_pose_estimation::simulation_geometry::tagIds();
        const auto &anchor_ids = uwb_pose_estimation::simulation_geometry::anchorIds();
        const auto &tag_positions = uwb_pose_estimation::simulation_geometry::tagPositionsB();
        const auto &anchor_positions = uwb_pose_estimation::simulation_geometry::anchorPositionsA();

        const std::size_t tag_index = static_cast<std::size_t>(tick % tag_ids.size());
        nlink_parser::LinktrackNodeframe2 frame;
        frame.role = 1;
        frame.id = static_cast<std::uint8_t>(tag_ids[tag_index]);
        const std::uint32_t elapsed_ms = static_cast<std::uint32_t>(std::llround((stamp.toSec() - sim_epoch_sec_) * 1000.0));
        frame.local_time = elapsed_ms;
        frame.system_time = elapsed_ms;
        frame.voltage = 12.0F;
        frame.pos_3d[0] = frame.pos_3d[1] = frame.pos_3d[2] = 0.0F;
        frame.eop_3d[0] = frame.eop_3d[1] = frame.eop_3d[2] = 0.0F;
        frame.vel_3d[0] = frame.vel_3d[1] = frame.vel_3d[2] = 0.0F;
        frame.angle_3d[0] = frame.angle_3d[1] = frame.angle_3d[2] = 0.0F;
        frame.quaternion[0] = 0.0F;
        frame.quaternion[1] = 0.0F;
        frame.quaternion[2] = 0.0F;
        frame.quaternion[3] = 1.0F;
        frame.imu_gyro_3d[0] = frame.imu_gyro_3d[1] = frame.imu_gyro_3d[2] = 0.0F;
        frame.imu_acc_3d[0] = frame.imu_acc_3d[1] = frame.imu_acc_3d[2] = 0.0F;
        frame.nodes.reserve(anchor_ids.size());

        const Eigen::Vector3d tag_A = truth.p_A_AB + truth.R_A_B * tag_positions[tag_index];
        for (std::size_t i = 0; i < anchor_ids.size(); ++i)
        {
            nlink_parser::LinktrackNode2 node;
            node.role = 0;
            node.id = static_cast<std::uint8_t>(anchor_ids[i]);
            const double true_distance = (tag_A - anchor_positions[i]).norm();
            double uwb_bias_change = uniformBias();//测试使用
            uwb_bias_change = 0.0;//UWB偏置
            node.dis = static_cast<float>(true_distance + uwb_bias_change + gaussian(sigma_uwb_));
            node.fp_rssi = -80.0F;
            node.rx_rssi = -77.0F;
            frame.nodes.push_back(node);
        }
        uwb_pub_.publish(frame);
    }

    void wallTimerCallback(const ros::WallTimerEvent &)
    {
        const double wall_elapsed = (ros::WallTime::now() - wall_start_).toSec();
        if (wall_elapsed < start_delay_sec_)
            return;

        if (finished_)
            return;

        double t = static_cast<double>(tick_) * kBaseDt;
        if (t > duration_sec_ + 1e-12)
        {
            if (loop_)
            {
                tick_ = 0;
                initial_pose_published_ = false;
                ++loop_count_;
                t = 0.0;
            }
            else
            {
                finished_ = true;
                ROS_INFO("Simulation completed at t=%.3f s", duration_sec_);
                if (shutdown_on_finish_)
                    ros::shutdown();
                return;
            }
        }

        const double absolute_sim_time =
            static_cast<double>(loop_count_) * (duration_sec_ + kBaseDt) + t;
        ros::Time stamp;
        stamp.fromSec(sim_epoch_sec_ + absolute_sim_time);
        publishClock(stamp);
        publishInitialPose(stamp);

        const TruthState truth = evaluateTruth(t);
        if (tick_ % kTruthTicks == 0)
            publishTruth(t, stamp, truth);
        if (tick_ % kImuTicks == 0)
            publishImu(t, stamp, truth);
        if (tick_ % kTofTicks == 0)
            publishTof(stamp, truth);

        // One tag message every base tick; each of six tags therefore runs at 50 Hz.
        publishOneUwbTag(tick_, t, stamp, truth);
        ++tick_;
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::WallTimer wall_timer_;
    ros::WallTime wall_start_;

    ros::Publisher clock_pub_;
    ros::Publisher imu_a_pub_;
    ros::Publisher imu_b_pub_;
    ros::Publisher tof_pub_;
    ros::Publisher uwb_pub_;
    ros::Publisher world_a_pub_;
    ros::Publisher world_b_pub_;
    ros::Publisher relative_truth_pub_;
    ros::Publisher relative_twist_pub_;
    ros::Publisher initial_pose_pub_;

    std::mt19937 rng_;
    std::uint32_t random_seed_ = 20260714U;
    std::uint64_t tick_ = 0;
    std::uint64_t loop_count_ = 0;
    bool initial_pose_published_ = false;
    bool finished_ = false;

    double duration_sec_ = 60.0;
    double start_delay_sec_ = 1.0;
    bool loop_ = false;
    bool shutdown_on_finish_ = false;
    const double sim_epoch_sec_ = 1.0;

    double sigma_gyro_density_ = 0.002;
    double sigma_acc_density_ = 0.020;
    Eigen::Vector3d imu_b_gyro_bias_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d imu_b_acc_bias_ = Eigen::Vector3d::Zero();
    double sigma_uwb_ = 0.033;
    double uwb_bias_min_ = 0.1;
    double uwb_bias_max_ = 0.3;
    double sigma_tof_ = 0.020;
    Eigen::Vector3d init_dp_ = Eigen::Vector3d(0.25, -0.20, 0.15);
    Eigen::Vector3d init_dtheta_ = Eigen::Vector3d::Zero();

    std::string imu_a_topic_;
    std::string imu_b_topic_;
    std::string tof_topic_;
    std::string uwb_topic_;
    std::string world_a_topic_;
    std::string world_b_topic_;
    std::string relative_truth_topic_;
    std::string relative_twist_topic_;
    std::string initial_pose_topic_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "sim_data_generator_node");
    SimDataGenerator node;
    ros::spin();
    return 0;
}
