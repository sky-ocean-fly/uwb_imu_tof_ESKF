/**
 * @file uwb_imu_tof_eskf_node.cpp
 * @brief IMU + UWB + ToF ESKF for relative pose ^A_B T, with mocap truth publishing.
 *
 * State:
 *   R = R_A_B, v = v_A_AB, p = p_A_AB
 * Error state:
 *   xi = [delta_theta, delta_v, delta_p]^T, left perturbation R = Exp(delta_theta^) R_hat
 *
 * Propagation:
 *   R_{k+1} = Exp(-omega_A dt) R_k Exp(omega_B dt)
 *   v_dot   = -omega_A^ v + R f_B - f_A
 *   p_dot   = -omega_A^ p + v
 *
 * Measurement:
 *   UWB: rho_ij = ||p + R b_j - a_i||
 *   ToF vertical height mode: h = e3^T (p + R c_T)
 *   ToF slant mode:          h = - e3^T(p + R c_T) / e3^T(R s_T)
 */

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nlink_parser/LinktrackNodeframe2.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <boost/bind.hpp>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <clocale>

namespace
{

using Matrix9d = Eigen::Matrix<double, 9, 9>;
using Matrix9x12d = Eigen::Matrix<double, 9, 12>;
using Matrix12d = Eigen::Matrix<double, 12, 12>;
using Vector9d = Eigen::Matrix<double, 9, 1>;

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
    const Eigen::Matrix3d W2 = W * W;

    if (theta < 1e-8)
    {
        return Eigen::Matrix3d::Identity() + W + 0.5 * W2;
    }

    const double A = std::sin(theta) / theta;
    const double B = (1.0 - std::cos(theta)) / (theta * theta);
    return Eigen::Matrix3d::Identity() + A * W + B * W2;
}

Eigen::Matrix3d eulerZYXDegToRot(double yaw_deg, double pitch_deg, double roll_deg)
{
    const double yaw = yaw_deg * M_PI / 180.0;
    const double pitch = pitch_deg * M_PI / 180.0;
    const double roll = roll_deg * M_PI / 180.0;

    Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rx(roll, Eigen::Vector3d::UnitX());
    return (Rz * Ry * Rx).toRotationMatrix();
}

Eigen::Matrix4d makeT(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

Eigen::Matrix4d inverseT(const Eigen::Matrix4d &T)
{
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    const Eigen::Vector3d t = T.block<3, 1>(0, 3);

    Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
    Ti.block<3, 3>(0, 0) = R.transpose();
    Ti.block<3, 1>(0, 3) = -R.transpose() * t;
    return Ti;
}

Eigen::Matrix3d makeR_BM_B()
{
    Eigen::Matrix3d R;
    R << 0, -1, 0,
         0, 0, 1,
        -1, 0, 0;
    return R;
}

Eigen::Matrix3d makeR_AM_A()
{
    Eigen::Matrix3d R;
    R << 0, -1, 0,
         0, 0, 1,
        -1, 0, 0;
    return R;
}

Eigen::Matrix4d computeMocapTruth_T_A_B_AssumeMEqualsAM(
    const Eigen::Matrix3d &R_M_BM,
    const Eigen::Vector3d &p_M_BM)
{
    const Eigen::Matrix3d R_AM_A = makeR_AM_A();
    const Eigen::Matrix3d R_BM_B = makeR_BM_B();

    const Eigen::Matrix4d T_M_BM = makeT(R_M_BM, p_M_BM);
    const Eigen::Matrix4d T_BM_B = makeT(R_BM_B, Eigen::Vector3d::Zero());
    const Eigen::Matrix4d T_AM_A = makeT(R_AM_A, Eigen::Vector3d::Zero());
    const Eigen::Matrix4d T_A_AM = inverseT(T_AM_A);

    return T_A_AM * T_M_BM * T_BM_B;
}

Eigen::Vector3d vec3FromImuAngular(const sensor_msgs::Imu::ConstPtr &msg)
{
    return Eigen::Vector3d(msg->angular_velocity.x,
                           msg->angular_velocity.y,
                           msg->angular_velocity.z);
}

Eigen::Vector3d vec3FromImuLinearAcc(const sensor_msgs::Imu::ConstPtr &msg)
{
    return Eigen::Vector3d(msg->linear_acceleration.x,
                           msg->linear_acceleration.y,
                           msg->linear_acceleration.z);
}

Eigen::Quaterniond quatFromPose(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    Eigen::Quaterniond q(msg->pose.orientation.w,
                         msg->pose.orientation.x,
                         msg->pose.orientation.y,
                         msg->pose.orientation.z);
    q.normalize();
    return q;
}

} // namespace

class UwbImuTofEskfNode
{
public:
    UwbImuTofEskfNode()
        : nh_(), pnh_("~"), initialized_(false), has_last_imu_pair_time_(false)
    {
        loadParams();
        loadGeometry();
        resetCovariance();

        pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(eskf_pose_topic_, 10);
        mocap_truth_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(mocap_truth_topic_, 10);

        uwb_sub_ = nh_.subscribe(uwb_topic_, 50, &UwbImuTofEskfNode::uwbCallback, this);
        tof_sub_ = nh_.subscribe(tof_topic_, 50, &UwbImuTofEskfNode::tofCallback, this);
        mocap_sub_ = nh_.subscribe(mocap_topic_, 50, &UwbImuTofEskfNode::mocapCallback, this);

        if (init_from_pose_topic_)
        {
            init_pose_sub_ = nh_.subscribe(init_pose_topic_, 10, &UwbImuTofEskfNode::initPoseCallback, this);
        }
        else if (!init_from_mocap_)
        {
            initializeFromParams();
        }

        imu_a_sub_.subscribe(nh_, imu_a_topic_, imu_queue_size_);
        imu_b_sub_.subscribe(nh_, imu_b_topic_, imu_queue_size_);
        //message_filters::Synchronizer<ImuSyncPolicy>，近似时间同步 
        //message_filters::sync_policies::ApproximateTime<sensor_msgs::Imu, sensor_msgs::Imu>
        imu_sync_.reset(new ImuSynchronizer(ImuSyncPolicy(imu_queue_size_), imu_a_sub_, imu_b_sub_));
        imu_sync_->setMaxIntervalDuration(ros::Duration(imu_sync_slop_));
        imu_sync_->registerCallback(boost::bind(&UwbImuTofEskfNode::imuPairCallback, this, _1, _2));

        ROS_INFO("UWB+IMU+ToF ESKF节点启动");
        ROS_INFO("  输出估计: %s", eskf_pose_topic_.c_str());
        ROS_INFO("  UWB话题: %s", uwb_topic_.c_str());
        ROS_INFO("  ToF话题: %s", tof_topic_.c_str());
        ROS_INFO("  IMU A话题: %s", imu_a_topic_.c_str());
        ROS_INFO("  IMU B话题: %s", imu_b_topic_.c_str());
        ROS_INFO("  初始化: %s", init_from_pose_topic_ ? init_pose_topic_.c_str() : (init_from_mocap_ ? "mocap" : "params"));
    }

private:
    using ImuSyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Imu, sensor_msgs::Imu>;
    using ImuSynchronizer = message_filters::Synchronizer<ImuSyncPolicy>;

    void loadParams()
    {
        pnh_.param<std::string>("uwb_topic", uwb_topic_, "/nlink_linktrack_nodeframe2");
        pnh_.param<std::string>("tof_topic", tof_topic_, "/ground_distance");
        pnh_.param<std::string>("imu_a_topic", imu_a_topic_, "/imu_A/data_raw");
        pnh_.param<std::string>("imu_b_topic", imu_b_topic_, "/imu_B/data_raw");
        pnh_.param<std::string>("mocap_topic", mocap_topic_, "/vrpn_client/UWB_rocket/pose");
        pnh_.param<std::string>("eskf_pose_topic", eskf_pose_topic_, "/eskf_estimated_pose");
        pnh_.param<std::string>("mocap_truth_topic", mocap_truth_topic_, "/mocap_truth_pose");
        pnh_.param<std::string>("init_pose_topic", init_pose_topic_, "/estimated_pose");
        pnh_.param<std::string>("frame_id", frame_id_, "anchor_frame_A");

        pnh_.param<bool>("init_from_pose_topic", init_from_pose_topic_, true);
        pnh_.param<bool>("init_from_mocap", init_from_mocap_, false);
        pnh_.param<bool>("tof_vertical_mode", tof_vertical_mode_, true);

        pnh_.param<int>("imu_queue_size", imu_queue_size_, 100);
        pnh_.param<double>("imu_sync_slop", imu_sync_slop_, 0.01);
        pnh_.param<double>("max_imu_dt", max_imu_dt_, 0.05);

        pnh_.param<double>("sigma_gyro_A", sigma_gyro_A_, 0.02);
        pnh_.param<double>("sigma_gyro_B", sigma_gyro_B_, 0.02);
        pnh_.param<double>("sigma_acc_A", sigma_acc_A_, 0.20);
        pnh_.param<double>("sigma_acc_B", sigma_acc_B_, 0.20);
        pnh_.param<double>("sigma_uwb", sigma_uwb_, 0.05);
        pnh_.param<double>("sigma_tof", sigma_tof_, 0.03);

        pnh_.param<double>("uwb_residual_gate", uwb_residual_gate_, 2.0);
        pnh_.param<double>("tof_residual_gate", tof_residual_gate_, 1.0);

        pnh_.param<double>("init_p_x", init_p_x_, 0.0);
        pnh_.param<double>("init_p_y", init_p_y_, 0.0);
        pnh_.param<double>("init_p_z", init_p_z_, 1.0);
        pnh_.param<double>("init_yaw_deg", init_yaw_deg_, 0.0);
        pnh_.param<double>("init_pitch_deg", init_pitch_deg_, 0.0);
        pnh_.param<double>("init_roll_deg", init_roll_deg_, 0.0);

        pnh_.param<double>("init_sigma_angle", init_sigma_angle_, 0.30);
        pnh_.param<double>("init_sigma_velocity", init_sigma_velocity_, 0.50);
        pnh_.param<double>("init_sigma_position", init_sigma_position_, 0.50);
    }

    void loadGeometry()
    {
        tag_ids_ = {5, 6, 7, 8};
        anchor_ids_ = {0, 1, 2, 3, 4, 9};

        // b_j: UWB标签在无人机B系下的位置，来自现有uwb_localization_node.cpp中的标定量。
        tag_positions_B_ = {
            Eigen::Vector3d(-0.017, 0.494, 0.037),
            Eigen::Vector3d(-0.457, -0.006, 0.046),
            Eigen::Vector3d(-0.023, -0.506, 0.058),
            Eigen::Vector3d(0.472, 0.010, 0.046)};

        // a_i: UWB基站在无人车/平台A系下的位置，来自现有uwb_localization_node.cpp中的标定量。
        anchor_positions_A_ = {
            Eigen::Vector3d(-0.677, -3.480, 2.098),
            Eigen::Vector3d(-2.863, 1.767, 0.913),
            Eigen::Vector3d(-2.880, -2.275, 0.756),
            Eigen::Vector3d(1.787, -2.296, 0.922),
            Eigen::Vector3d(1.795, 1.748, 1.038),
            Eigen::Vector3d(-0.736, 3.512, 1.813)};

        // c_T: ToF在B系下的位置，来自现有uwb_localization_node.cpp中的标定量。
        tof_position_B_ = Eigen::Vector3d(0.186, 0.680, -0.049);

        // s_T: ToF光束方向。若订阅/ground_distance的point.y垂直高度，默认不使用该方向。
        tof_ray_B_ = Eigen::Vector3d(0.0, 0.0, -1.0);
    }

    void resetCovariance()
    {
        P_.setZero();
        P_.block<3, 3>(0, 0) = init_sigma_angle_ * init_sigma_angle_ * Eigen::Matrix3d::Identity();
        P_.block<3, 3>(3, 3) = init_sigma_velocity_ * init_sigma_velocity_ * Eigen::Matrix3d::Identity();
        P_.block<3, 3>(6, 6) = init_sigma_position_ * init_sigma_position_ * Eigen::Matrix3d::Identity();
    }

    void initializeFromParams()
    {
        R_ = eulerZYXDegToRot(init_yaw_deg_, init_pitch_deg_, init_roll_deg_);
        v_.setZero();
        p_ = Eigen::Vector3d(init_p_x_, init_p_y_, init_p_z_);
        initialized_ = true;
        has_last_imu_pair_time_ = false;
        ROS_WARN("ESKF使用参数初始化：p=[%.3f %.3f %.3f], yaw/pitch/roll=[%.2f %.2f %.2f] deg",
                 p_.x(), p_.y(), p_.z(), init_yaw_deg_, init_pitch_deg_, init_roll_deg_);
    }

    int tagIdToIndex(int tag_id) const
    {
        for (size_t i = 0; i < tag_ids_.size(); ++i)
        {
            if (tag_ids_[i] == tag_id)
                return static_cast<int>(i);
        }
        return -1;
    }

    int anchorIdToIndex(int anchor_id) const
    {
        for (size_t i = 0; i < anchor_ids_.size(); ++i)
        {
            if (anchor_ids_[i] == anchor_id)
                return static_cast<int>(i);
        }
        return -1;
    }

    void normalizeRotation()
    {
        Eigen::Quaterniond q(R_);
        q.normalize();
        R_ = q.toRotationMatrix();
    }

    void initPoseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (initialized_)
            return;

        Eigen::Quaterniond q = quatFromPose(msg);
        R_ = q.toRotationMatrix();
        p_ = Eigen::Vector3d(msg->pose.position.x,
                             msg->pose.position.y,
                             msg->pose.position.z);
        v_.setZero();
        initialized_ = true;
        has_last_imu_pair_time_ = false;
        resetCovariance();

        ROS_WARN("ESKF已用初始位姿话题初始化：%s", init_pose_topic_.c_str());
        publishEstimate(msg->header.stamp);
    }

    void mocapCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        const Eigen::Quaterniond q = quatFromPose(msg);
        const Eigen::Matrix3d R_M_BM = q.toRotationMatrix();
        const Eigen::Vector3d p_M_BM(msg->pose.position.x,
                                     msg->pose.position.y,
                                     msg->pose.position.z);

        const Eigen::Matrix4d T_A_B = computeMocapTruth_T_A_B_AssumeMEqualsAM(R_M_BM, p_M_BM);
        const Eigen::Matrix3d R_A_B = T_A_B.block<3, 3>(0, 0);
        const Eigen::Vector3d p_A_B = T_A_B.block<3, 1>(0, 3);

        publishMocapTruth(R_A_B, p_A_B, msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp);

        if (init_from_mocap_ && !initialized_ && !init_from_pose_topic_)
        {
            R_ = R_A_B;
            p_ = p_A_B;
            v_.setZero();
            initialized_ = true;
            has_last_imu_pair_time_ = false;
            resetCovariance();
            ROS_WARN("ESKF已用动捕真值初始化。该模式只建议用于调试，不建议作为正式实验结果。 ");
            publishEstimate(msg->header.stamp);
        }
    }

    void imuPairCallback(const sensor_msgs::Imu::ConstPtr &imu_a,
                         const sensor_msgs::Imu::ConstPtr &imu_b)
    {
        if (!initialized_)
        {
            ROS_WARN_THROTTLE(2.0, "ESKF尚未初始化，等待初始位姿。可运行原UWB节点提供/init_pose_topic，或关闭init_from_pose_topic改用参数初始化。 ");
            return;
        }

        const ros::Time stamp_a = imu_a->header.stamp.isZero() ? ros::Time::now() : imu_a->header.stamp;
        const ros::Time stamp_b = imu_b->header.stamp.isZero() ? ros::Time::now() : imu_b->header.stamp;
        ros::Time stamp;
        stamp.fromSec((stamp_a.toSec() + stamp_b.toSec()) * 0.5);

        if (!has_last_imu_pair_time_)
        {
            last_imu_pair_time_ = stamp;
            has_last_imu_pair_time_ = true;
            return;
        }

        double dt = (stamp - last_imu_pair_time_).toSec();
        last_imu_pair_time_ = stamp;

        if (dt <= 0.0 || dt > max_imu_dt_)
        {
            ROS_WARN_THROTTLE(2.0, "IMU dt异常: %.6f s，跳过本次传播", dt);
            return;
        }

        const Eigen::Vector3d omega_A = vec3FromImuAngular(imu_a);
        const Eigen::Vector3d omega_B = vec3FromImuAngular(imu_b);
        const Eigen::Vector3d f_A = vec3FromImuLinearAcc(imu_a);
        const Eigen::Vector3d f_B = vec3FromImuLinearAcc(imu_b);

        propagate(omega_A, omega_B, f_A, f_B, dt);
        publishEstimate(stamp);
    }

    void propagate(const Eigen::Vector3d &omega_A,
                   const Eigen::Vector3d &omega_B,
                   const Eigen::Vector3d &f_A,
                   const Eigen::Vector3d &f_B,
                   double dt)
    {
        const Eigen::Matrix3d R_old = R_;
        const Eigen::Vector3d v_old = v_;
        const Eigen::Vector3d p_old = p_;

        // 名义状态传播。
        R_ = expSO3(-omega_A * dt) * R_ * expSO3(omega_B * dt);
        normalizeRotation();

        v_ = v_old + (-skew(omega_A) * v_old + R_old * f_B - f_A) * dt;
        p_ = p_old + (-skew(omega_A) * p_old + v_old) * dt;

        // 误差状态协方差传播：xi_dot = F xi + G n。
        const Eigen::Matrix3d Omega_A = skew(omega_A);
        Matrix9d F = Matrix9d::Zero();
        F.block<3, 3>(0, 0) = -Omega_A;
        F.block<3, 3>(3, 0) = -skew(R_old * f_B);
        F.block<3, 3>(3, 3) = -Omega_A;
        F.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();
        F.block<3, 3>(6, 6) = -Omega_A;

        Matrix9x12d G = Matrix9x12d::Zero();
        G.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        G.block<3, 3>(0, 3) = -R_old;
        G.block<3, 3>(3, 0) = -skew(v_old);
        G.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
        G.block<3, 3>(3, 9) = -R_old;
        G.block<3, 3>(6, 0) = -skew(p_old);

        Matrix12d Qc = Matrix12d::Zero();
        Qc.block<3, 3>(0, 0) = sigma_gyro_A_ * sigma_gyro_A_ * Eigen::Matrix3d::Identity();
        Qc.block<3, 3>(3, 3) = sigma_gyro_B_ * sigma_gyro_B_ * Eigen::Matrix3d::Identity();
        Qc.block<3, 3>(6, 6) = sigma_acc_A_ * sigma_acc_A_ * Eigen::Matrix3d::Identity();
        Qc.block<3, 3>(9, 9) = sigma_acc_B_ * sigma_acc_B_ * Eigen::Matrix3d::Identity();

        const Matrix9d Phi = Matrix9d::Identity() + F * dt;
        const Matrix9d Qd = G * Qc * G.transpose() * dt;
        P_ = Phi * P_ * Phi.transpose() + Qd;
        symmetrizeCovariance();
    }

    void uwbCallback(const nlink_parser::LinktrackNodeframe2::ConstPtr &msg)
    {
        const int tag_id = msg->id;
        if (tagIdToIndex(tag_id) < 0)
            return;

        {
            std::lock_guard<std::mutex> lock(dist_mutex_);
            for (const auto &node : msg->nodes)
            {
                const int anchor_id = node.id;
                if (anchorIdToIndex(anchor_id) >= 0)
                {
                    distances_cache_[tag_id][anchor_id] = static_cast<double>(node.dis);
                }
            }
        }

        if (!initialized_)
            return;

        updateUwb(ros::Time::now());
    }

    void updateUwb(const ros::Time &stamp)
    {
        std::map<int, std::map<int, double>> distances;
        {
            std::lock_guard<std::mutex> lock(dist_mutex_);
            distances = distances_cache_;
        }

        const int max_rows = static_cast<int>(tag_ids_.size() * anchor_ids_.size());
        Eigen::MatrixXd H(max_rows, 9);
        Eigen::VectorXd r(max_rows);
        H.setZero();
        r.setZero();

        int row = 0;
        for (size_t j = 0; j < tag_ids_.size(); ++j)
        {
            const int tag_id = tag_ids_[j];
            if (distances.find(tag_id) == distances.end())
                continue;

            for (size_t i = 0; i < anchor_ids_.size(); ++i)
            {
                const int anchor_id = anchor_ids_[i];
                auto it = distances[tag_id].find(anchor_id);
                if (it == distances[tag_id].end())
                    continue;

                const double z = it->second;
                if (!std::isfinite(z) || z <= 0.0)
                    continue;

                const Eigen::Vector3d Rb = R_ * tag_positions_B_[j];
                const Eigen::Vector3d q = p_ + Rb - anchor_positions_A_[i];
                const double rho_hat = q.norm();
                if (rho_hat < 1e-6)
                    continue;

                const double residual = z - rho_hat;
                if (uwb_residual_gate_ > 0.0 && std::fabs(residual) > uwb_residual_gate_)
                    continue;

                const Eigen::Vector3d u = q / rho_hat;
                H.block<1, 3>(row, 0) = -u.transpose() * skew(Rb);
                // H.block<1, 3>(row, 0).setZero();//UWB不修正姿态
                H.block<1, 3>(row, 3).setZero();
                // H.block<1, 3>(row, 6) = u.transpose();
                // UWB 只修正 p_x, p_y，不直接修正 p_z
                H(row, 6) = u.x();
                H(row, 7) = u.y();
                H(row, 8) = 0.0;
                r(row) = residual;
                ++row;
            }
        }

        if (row < 4)
        {
            ROS_WARN_THROTTLE(2.0, "有效UWB测距不足：%d", row);
            return;
        }

        H.conservativeResize(row, Eigen::NoChange);
        r.conservativeResize(row);

        Eigen::MatrixXd Rm = sigma_uwb_ * sigma_uwb_ * Eigen::MatrixXd::Identity(row, row);
        measurementUpdate(H, r, Rm, true);
        publishEstimate(stamp);
    }

    void tofCallback(const geometry_msgs::PointStamped::ConstPtr &msg)
    {
        if (!initialized_)
            return;

        const double z = tof_vertical_mode_ ? msg->point.y : msg->point.x;
        if (!std::isfinite(z) || z <= 0.0)
            return;

        Eigen::Matrix<double, 1, 9> H;
        H.setZero();
        double z_hat = 0.0;

        const Eigen::Vector3d e3(0.0, 0.0, 1.0);
        const Eigen::Vector3d Rc = R_ * tof_position_B_;

        if (tof_vertical_mode_)
        {
            // 当前/ground_distance的point.y已经是垂直高度时，用这个简化观测更稳。
            z_hat = e3.transpose() * (p_ + Rc);
            H.block<1, 3>(0, 0) = -e3.transpose() * skew(Rc);
            // H.block<1, 3>(0, 0).setZero();   // 先不要让ToF修正姿态
            H.block<1, 3>(0, 6) = e3.transpose();
        }
        else
        {
            // 原始斜距模式，对应推导中的 d = -alpha / beta。
            const Eigen::Vector3d Rs = R_ * tof_ray_B_;
            const double alpha = e3.transpose() * (p_ + Rc);
            const double beta = e3.transpose() * Rs;
            if (std::fabs(beta) < 1e-4)
            {
                ROS_WARN_THROTTLE(2.0, "ToF光束几乎平行于平台平面，跳过ToF更新");
                return;
            }

            z_hat = -alpha / beta;
            H.block<1, 3>(0, 0) = (1.0 / beta) * e3.transpose() * skew(Rc)
                                  - (alpha / (beta * beta)) * e3.transpose() * skew(Rs);
            // H.block<1, 3>(0, 0).setZero();   // 先不要让ToF修正姿态
            H.block<1, 3>(0, 6) = -(1.0 / beta) * e3.transpose();
        }

        const double residual = z - z_hat;
        if (tof_residual_gate_ > 0.0 && std::fabs(residual) > tof_residual_gate_)
            return;

        Eigen::VectorXd r(1);
        r(0) = residual;
        Eigen::MatrixXd Hdyn(1, 9);
        Hdyn.row(0) = H;
        Eigen::MatrixXd Rm(1, 1);
        Rm(0, 0) = sigma_tof_ * sigma_tof_;

        measurementUpdate(Hdyn, r, Rm, true);
        publishEstimate(msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp);
    }

    //加入UWB和ToF不修正姿态(true为修正，false为不修正)
    void measurementUpdate(const Eigen::MatrixXd &H,
                           const Eigen::VectorXd &r,
                           const Eigen::MatrixXd &Rm,
                           bool lock_attitude = false)
    {
        if (H.rows() == 0)
            return;

        const Matrix9d I9 = Matrix9d::Identity();
        const Eigen::MatrixXd S = H * P_ * H.transpose() + Rm;
        Eigen::MatrixXd K = (S.ldlt().solve(H * P_.transpose())).transpose();

        if (lock_attitude)
        {
            K.block(0, 0, 3, K.cols()).setZero();
        }

        Vector9d dx = K * r;

        if (lock_attitude)
        {
            dx.segment<3>(0).setZero();
        }

        const Matrix9d IKH = I9 - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * Rm * K.transpose();

        injectError(dx);
        symmetrizeCovariance();
    }

    void injectError(const Vector9d &dx)
    {
        const Eigen::Vector3d dtheta = dx.segment<3>(0);
        const Eigen::Vector3d dv = dx.segment<3>(3);
        const Eigen::Vector3d dp = dx.segment<3>(6);

        R_ = expSO3(dtheta) * R_;
        normalizeRotation();
        v_ += dv;
        p_ += dp;
    }

    void symmetrizeCovariance()
    {
        P_ = 0.5 * (P_ + P_.transpose());
    }

    void publishEstimate(const ros::Time &stamp)
    {
        if (!initialized_)
            return;

        geometry_msgs::PoseStamped msg;
        msg.header.stamp = stamp.isZero() ? ros::Time::now() : stamp;
        msg.header.frame_id = frame_id_;

        Eigen::Quaterniond q(R_);
        q.normalize();

        msg.pose.position.x = p_.x();
        msg.pose.position.y = p_.y();
        msg.pose.position.z = p_.z();
        msg.pose.orientation.w = q.w();
        msg.pose.orientation.x = q.x();
        msg.pose.orientation.y = q.y();
        msg.pose.orientation.z = q.z();

        pose_pub_.publish(msg);
    }

    void publishMocapTruth(const Eigen::Matrix3d &R,
                           const Eigen::Vector3d &p,
                           const ros::Time &stamp)
    {
        geometry_msgs::PoseStamped msg;
        msg.header.stamp = stamp.isZero() ? ros::Time::now() : stamp;
        msg.header.frame_id = frame_id_;

        Eigen::Quaterniond q(R);
        q.normalize();
        msg.pose.position.x = p.x();
        msg.pose.position.y = p.y();
        msg.pose.position.z = p.z();
        msg.pose.orientation.w = q.w();
        msg.pose.orientation.x = q.x();
        msg.pose.orientation.y = q.y();
        msg.pose.orientation.z = q.z();

        mocap_truth_pub_.publish(msg);
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    std::string uwb_topic_;
    std::string tof_topic_;
    std::string imu_a_topic_;
    std::string imu_b_topic_;
    std::string mocap_topic_;
    std::string eskf_pose_topic_;
    std::string mocap_truth_topic_;
    std::string init_pose_topic_;
    std::string frame_id_;

    bool init_from_pose_topic_;
    bool init_from_mocap_;
    bool tof_vertical_mode_;

    int imu_queue_size_;
    double imu_sync_slop_;
    double max_imu_dt_;

    double sigma_gyro_A_;
    double sigma_gyro_B_;
    double sigma_acc_A_;
    double sigma_acc_B_;
    double sigma_uwb_;
    double sigma_tof_;
    double uwb_residual_gate_;
    double tof_residual_gate_;

    double init_p_x_;
    double init_p_y_;
    double init_p_z_;
    double init_yaw_deg_;
    double init_pitch_deg_;
    double init_roll_deg_;
    double init_sigma_angle_;
    double init_sigma_velocity_;
    double init_sigma_position_;

    Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_ = Eigen::Vector3d::Zero();
    Matrix9d P_ = Matrix9d::Identity();

    bool initialized_;
    bool has_last_imu_pair_time_;
    ros::Time last_imu_pair_time_;

    std::vector<int> tag_ids_;
    std::vector<int> anchor_ids_;
    std::vector<Eigen::Vector3d> tag_positions_B_;
    std::vector<Eigen::Vector3d> anchor_positions_A_;
    Eigen::Vector3d tof_position_B_;
    Eigen::Vector3d tof_ray_B_;

    std::map<int, std::map<int, double>> distances_cache_;
    std::mutex dist_mutex_;

    ros::Subscriber uwb_sub_;
    ros::Subscriber tof_sub_;
    ros::Subscriber mocap_sub_;
    ros::Subscriber init_pose_sub_;
    ros::Publisher pose_pub_;
    ros::Publisher mocap_truth_pub_;

    message_filters::Subscriber<sensor_msgs::Imu> imu_a_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_b_sub_;
    boost::shared_ptr<ImuSynchronizer> imu_sync_;
};

int main(int argc, char **argv)
{
    // 设置locale以支持中文输出
    setlocale(LC_ALL, "zh_CN.UTF-8");
    setlocale(LC_NUMERIC, "C");  // 防止小数点变成逗号

    ros::init(argc, argv, "uwb_imu_tof_eskf_node");
    UwbImuTofEskfNode node;
    ros::spin();
    return 0;
}
