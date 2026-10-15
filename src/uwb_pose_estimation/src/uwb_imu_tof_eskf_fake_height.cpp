/**
 * @file uwb_imu_tof_eskf_fake_height.cpp
 * @brief Simulation version of IMU + UWB + ToF ESKF for relative pose ^A_B T.
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
 *   UWB fake-height mode:
 *     r = H_x delta_x + H_h delta_h_fake + n.
 *     delta_h_fake is eliminated with a Schur complement for every UWB batch
 *     and is never injected into the real p_z state.
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
#include <set>

#include "simulation_geometry.h"

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

        publish_timer_ = nh_.createTimer(
            ros::Duration(1.0 / publish_rate_hz_),
            &UwbImuTofEskfNode::publishTimerCallback,
            this);

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
        ROS_INFO("  UWB假高度: %s", use_uwb_fake_height_ ? "启用" : "关闭");
        ROS_INFO("  初始化: %s", init_from_pose_topic_ ? init_pose_topic_.c_str() : (init_from_mocap_ ? "mocap" : "params"));
    }

private:
    using ImuSyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Imu, sensor_msgs::Imu>;
    using ImuSynchronizer = message_filters::Synchronizer<ImuSyncPolicy>;
    using DistanceMap = std::map<int, std::map<int, double>>;

    void loadParams()
    {
        pnh_.param<std::string>("uwb_topic", uwb_topic_, "/nlink_linktrack_nodeframe2");
        pnh_.param<std::string>("tof_topic", tof_topic_, "/ground_distance");
        pnh_.param<std::string>("imu_a_topic", imu_a_topic_, "/imu_A/data_raw");
        pnh_.param<std::string>("imu_b_topic", imu_b_topic_, "/imu_B/data_raw");
        pnh_.param<std::string>("mocap_topic", mocap_topic_, "/sim/relative_truth_pose");
        pnh_.param<std::string>("eskf_pose_topic", eskf_pose_topic_, "/eskf_estimated_pose");
        pnh_.param<std::string>("mocap_truth_topic", mocap_truth_topic_, "/eskf_mocap_truth_pose");
        pnh_.param<std::string>("init_pose_topic", init_pose_topic_, "/estimated_pose");
        pnh_.param<std::string>("frame_id", frame_id_, "anchor_frame_A");
        pnh_.param<double>("publish_rate_hz", publish_rate_hz_, 50.0);//发布估计的和真实的相对位姿的50Hz定时器

        pnh_.param<bool>("init_from_pose_topic", init_from_pose_topic_, true);
        pnh_.param<bool>("init_from_mocap", init_from_mocap_, false);
        pnh_.param<bool>("tof_vertical_mode", tof_vertical_mode_, true);
        pnh_.param<bool>("use_uwb_fake_height", use_uwb_fake_height_, false);//“假高度”

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
        pnh_.param<double>("uwb_fake_height_min_information",
                           uwb_fake_height_min_information_, 1e-8);

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
        tag_ids_ = uwb_pose_estimation::simulation_geometry::tagIds();
        anchor_ids_ = uwb_pose_estimation::simulation_geometry::anchorIds();
        tag_positions_B_ = uwb_pose_estimation::simulation_geometry::tagPositionsB();
        anchor_positions_A_ = uwb_pose_estimation::simulation_geometry::anchorPositionsA();
        tof_position_B_ = uwb_pose_estimation::simulation_geometry::tofPositionB();
        tof_ray_B_ = uwb_pose_estimation::simulation_geometry::tofRayB();
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
    }

    void mocapCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        // Simulation truth is already ^A R_B and ^A p_AB. No mocap-frame conversion is allowed here.
        const Eigen::Quaterniond q = quatFromPose(msg);
        const Eigen::Matrix3d R_A_B = q.toRotationMatrix();
        const Eigen::Vector3d p_A_B(msg->pose.position.x,
                                    msg->pose.position.y,
                                    msg->pose.position.z);

        latest_mocap_truth_R_ = R_A_B;
        latest_mocap_truth_p_ = p_A_B;
        has_mocap_truth_ = true;

        if (init_from_mocap_ && !initialized_ && !init_from_pose_topic_)
        {
            R_ = R_A_B;
            p_ = p_A_B;
            v_.setZero();
            initialized_ = true;
            has_last_imu_pair_time_ = false;
            resetCovariance();
            ROS_WARN("ESKF initialized from direct simulation truth; use only for debugging.");
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

        ros::WallTime t_predict_start = ros::WallTime::now();
        propagate(omega_A, omega_B, f_A, f_B, dt);
        ros::WallDuration predict_elapsed = ros::WallTime::now() - t_predict_start;
        ROS_INFO("[ESKF计时] 预测(propagate) 耗时: %.3f ms, dt=%.3f ms",
                 predict_elapsed.toSec() * 1e3, dt * 1e3);
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

        // 只处理指定的4个标签
        if (tagIdToIndex(tag_id) < 0)
            return;

        // 初始化之前不缓存，保证初始化后的第一批数据是全新的
        if (!initialized_)
            return;

        DistanceMap complete_batch;
        bool batch_ready = false;

        {
            std::lock_guard<std::mutex> lock(dist_mutex_);

            /*
            * 当前消息对应一个标签。
            * 先清空该标签上一次的数据，避免当前消息缺少某个基站时，
            * 继续使用上一次该基站的旧距离。
            */
            std::map<int, double> &tag_distances =
                distances_cache_[tag_id];

            tag_distances.clear();

            for (const auto &node : msg->nodes)
            {
                const int anchor_id = node.id;

                if (anchorIdToIndex(anchor_id) < 0)
                    continue;

                const double distance = static_cast<double>(node.dis);

                if (!std::isfinite(distance) || distance <= 0.0)
                    continue;

                tag_distances[anchor_id] = distance;
            }

            // 当前标签没有任何有效测距，不认为已经收到该标签
            if (tag_distances.empty())
            {
                distances_cache_.erase(tag_id);
                return;
            }

            // 记录本轮已经收到的标签
            received_tag_ids_.insert(tag_id);

            // 收齐4个不同标签
            if (received_tag_ids_.size() == tag_ids_.size())
            {
                // 固定本轮数据，防止下一轮回调修改本次更新数据
                complete_batch = distances_cache_;

                // 开始下一轮标签数据收集
                distances_cache_.clear();
                received_tag_ids_.clear();

                batch_ready = true;
            }
        }

        // 收齐4个标签后，只进行一次UWB更新
        if (batch_ready)
        {
            updateUwb(ros::Time::now(), complete_batch);
        }
    }

    void updateUwb(const ros::Time &stamp,
                   const DistanceMap &distances)
    {
        ros::WallTime t_start = ros::WallTime::now();

        const int max_rows = static_cast<int>(tag_ids_.size() * anchor_ids_.size());
        Eigen::MatrixXd H(max_rows, 9);
        Eigen::VectorXd H_fake_height(max_rows);
        Eigen::VectorXd r(max_rows);
        H.setZero();
        H_fake_height.setZero();
        r.setZero();

        int row = 0;
        for (size_t j = 0; j < tag_ids_.size(); ++j)
        {
            const int tag_id = tag_ids_[j];

            // 查找当前标签，不能使用 distances[tag_id]
            const auto tag_it = distances.find(tag_id);
            if (tag_it == distances.end())
                continue;

            // 当前标签对应的所有基站测距
            const std::map<int, double> &tag_distances = tag_it->second;

            for (size_t i = 0; i < anchor_ids_.size(); ++i)
            {
                const int anchor_id = anchor_ids_[i];

                // 在当前标签的测距数据中查找基站
                const auto distance_it = tag_distances.find(anchor_id);
                if (distance_it == tag_distances.end())
                    continue;

                const double z = distance_it->second;
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
                if (use_uwb_fake_height_)
                {
                    /*
                     * 真实误差状态只保留 p_x、p_y 的UWB雅可比。
                     * u_z 被放入本批次独立的假高度 delta_h_fake：
                     *
                     *   residual = H_x delta_x + H_fake delta_h_fake + noise.
                     *
                     * 假高度会在 updateUwbWithFakeHeight() 中通过Schur补消去，
                     * 因而不会进入真实 p_z；真实高度继续由IMU传播和ToF更新。
                     */
                    H.block<1, 2>(row, 6) = u.head<2>().transpose();
                    H(row, 8) = 0.0;
                    H_fake_height(row) = u.z();
                }
                else
                {
                    H.block<1, 3>(row, 6) = u.transpose();
                }
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
        H_fake_height.conservativeResize(row);
        r.conservativeResize(row);

        Eigen::MatrixXd Rm = sigma_uwb_ * sigma_uwb_ * Eigen::MatrixXd::Identity(row, row);
        ROS_INFO_THROTTLE(1.0, "[UWB update] sigma_uwb_=%.6f, variance=%.6f, row=%d",
                          sigma_uwb_, sigma_uwb_ * sigma_uwb_, row);
        if (use_uwb_fake_height_)
        {
            updateUwbWithFakeHeight(H, H_fake_height, r, Rm);
        }
        else
        {
            measurementUpdate(H, r, Rm, MeasurementMode::FULL);
        }

        ros::WallDuration total_elapsed = ros::WallTime::now() - t_start;
        ROS_INFO("[ESKF计时] UWB更新全周期 耗时: %.3f ms, 有效测量数=%d",
                 total_elapsed.toSec() * 1e3, row);
    }

    void updateUwbWithFakeHeight(const Eigen::MatrixXd &H_x,
                                 const Eigen::VectorXd &H_fake_height,
                                 const Eigen::VectorXd &r,
                                 const Eigen::MatrixXd &Rm)
    {
        /*
         * 对一般测量协方差 Rm 先白化：
         *
         *   r_w = L^{-1} r, H_w = L^{-1} H_x, h_w = L^{-1} H_fake,
         *   Rm = L L^T.
         *
         * 再投影到 h_w 的正交补：
         *
         *   Pi = I - h_w (h_w^T h_w)^{-1} h_w^T.
         *
         * 因此 H_perp = Pi H_w、r_perp = Pi r_w，噪声协方差为 I。
         * 这与文档中的
         *   W_perp = W - W H_fake
         *              (H_fake^T W H_fake)^{-1} H_fake^T W
         * 完全等价，但不需要显式求 Rm^{-1}。
         */
        if (H_x.rows() == 0 ||
            H_fake_height.size() != H_x.rows() ||
            r.size() != H_x.rows() ||
            Rm.rows() != H_x.rows() ||
            Rm.cols() != H_x.rows())
        {
            ROS_ERROR_THROTTLE(2.0, "UWB假高度更新维度不一致，跳过本批次");
            return;
        }

        Eigen::LLT<Eigen::MatrixXd> llt(Rm);
        if (llt.info() != Eigen::Success)
        {
            ROS_ERROR_THROTTLE(2.0, "UWB测量协方差Cholesky分解失败，跳过本批次");
            return;
        }

        const Eigen::MatrixXd H_w = llt.matrixL().solve(H_x);
        const Eigen::VectorXd r_w = llt.matrixL().solve(r);
        const Eigen::VectorXd h_w = llt.matrixL().solve(H_fake_height);
        const double fake_height_information = h_w.squaredNorm();

        if (!std::isfinite(fake_height_information) ||
            fake_height_information < uwb_fake_height_min_information_)
        {
            /*
             * 当前几何对竖直位移几乎不敏感，无法估计假高度。
             * 此时直接使用已经去掉真实 p_z 列的 H_x，并继续禁止注入 p_z。
             */
            ROS_WARN_THROTTLE(
                2.0,
                "UWB假高度不可观：H_fake^T R^-1 H_fake=%.3e，退化为不更新真实p_z",
                fake_height_information);
            measurementUpdate(H_x, r, Rm, MeasurementMode::UWB_FAKE_HEIGHT);
            return;
        }

        const Eigen::MatrixXd projector =
            Eigen::MatrixXd::Identity(H_x.rows(), H_x.rows()) -
            (h_w * h_w.transpose()) / fake_height_information;

        const Eigen::MatrixXd H_perp = projector * H_w;
        const Eigen::VectorXd r_perp = projector * r_w;
        const Eigen::MatrixXd R_perp =
            Eigen::MatrixXd::Identity(H_x.rows(), H_x.rows());

        const Vector9d dx =
            measurementUpdate(H_perp, r_perp, R_perp,
                              MeasurementMode::UWB_FAKE_HEIGHT);

        // 给定位姿修正后，剩余残差中最能由竖直位移解释的临时假高度。
        const double delta_h_fake =
            h_w.dot(r_w - H_w * dx) / fake_height_information;

        ROS_INFO(
            "UWB假高度: delta_h_fake=%.4f m（仅本批次使用并丢弃）, "
            "delta_p_xy=[%.4f %.4f] m, real_p_z=%.4f m",
            delta_h_fake, dx(6), dx(7), p_.z());
    }

    void tofCallback(const geometry_msgs::PointStamped::ConstPtr &msg)
    {
        if (!initialized_)
            return;

        ros::WallTime t_start = ros::WallTime::now();

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
            // 当前/ground_distance的point.y已经是垂直高度时，用这个简化观测更稳。(point.y=d_{tof}*cos(pitch)*cos(roll))
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
        ROS_INFO("ToF: measured=%.4f m, predicted=%.4f m, residual=%.4f m, p_z=%.4f m, Rc_z=%.4f m",
                 z, z_hat, residual, p_.z(), Rc.z());
        if (tof_residual_gate_ > 0.0 && std::fabs(residual) > tof_residual_gate_)
            return;

        Eigen::VectorXd r(1);
        r(0) = residual;
        Eigen::MatrixXd Hdyn(1, 9);
        Hdyn.row(0) = H;
        Eigen::MatrixXd Rm(1, 1);
        Rm(0, 0) = sigma_tof_ * sigma_tof_;

        measurementUpdate(Hdyn, r, Rm, MeasurementMode::FULL);  // ToF 解锁姿态，允许修正 roll/pitch

        ros::WallDuration total_elapsed = ros::WallTime::now() - t_start;
        ROS_INFO("[ESKF计时] ToF更新全周期 耗时: %.3f ms",
                 total_elapsed.toSec() * 1e3);
    }

    void publishTimerCallback(const ros::TimerEvent &event)
    {
        // 两个话题使用完全相同的时间戳，方便后续同步比较
        const ros::Time stamp = event.current_real;

        if (initialized_)
        {
            publishEstimate(stamp);
        }

        if (has_mocap_truth_)
        {
            publishMocapTruth(
                latest_mocap_truth_R_,
                latest_mocap_truth_p_,
                stamp);
        }
    }

    enum class MeasurementMode
    {
        FULL,
        UWB_FAKE_HEIGHT,
        UWB_XY_ONLY,
        TOF_Z_VZ_ONLY
    };

    Vector9d measurementUpdate(const Eigen::MatrixXd &H,
                               const Eigen::VectorXd &r,
                               const Eigen::MatrixXd &Rm,
                               MeasurementMode mode = MeasurementMode::FULL)
    {
        const Vector9d zero_dx = Vector9d::Zero();
        if (H.rows() == 0 ||
            H.cols() != 9 ||
            r.size() != H.rows() ||
            Rm.rows() != H.rows() ||
            Rm.cols() != H.rows())
        {
            ROS_ERROR_THROTTLE(2.0, "ESKF测量更新维度不一致，跳过本批次");
            return zero_dx;
        }

        ros::WallTime t_update_start = ros::WallTime::now();

        const Matrix9d I9 = Matrix9d::Identity();
        const Eigen::MatrixXd S = H * P_ * H.transpose() + Rm;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
        if (ldlt.info() != Eigen::Success)
        {
            ROS_ERROR_THROTTLE(2.0, "ESKF创新协方差LDLT分解失败，跳过本批次");
            return zero_dx;
        }

        const Eigen::MatrixXd K_full =
            (ldlt.solve(H * P_.transpose())).transpose();
        if (!K_full.allFinite())
        {
            ROS_ERROR_THROTTLE(2.0, "ESKF卡尔曼增益出现NaN/Inf，跳过本批次");
            return zero_dx;
        }

        Eigen::MatrixXd K =
            Eigen::MatrixXd::Zero(9, H.rows());

        if (mode == MeasurementMode::FULL)
        {
            K = K_full;
        }
        else if (mode == MeasurementMode::UWB_FAKE_HEIGHT)
        {
            /*
             * 保留当前代码原有的UWB更新能力（姿态、速度、p_x、p_y可通过
             * 直接观测或协方差交叉项得到修正），但严格禁止UWB注入真实p_z。
             * 临时竖直修正由 delta_h_fake 承担，并在本次更新后丢弃。
             */
            K = K_full;
            K.row(8).setZero();
        }
        else if (mode == MeasurementMode::UWB_XY_ONLY)
        {
            // UWB严格只修改p_x、p_y
            K.row(6) = K_full.row(6);
            K.row(7) = K_full.row(7);
        }
        else if (mode == MeasurementMode::TOF_Z_VZ_ONLY)
        {
            // ToF修改v_z和p_z
            K.row(5) = K_full.row(5);
            K.row(8) = K_full.row(8);
        }

        const Vector9d dx = K * r;

        const Matrix9d IKH = I9 - K * H;
        P_ = IKH * P_ * IKH.transpose()
             + K * Rm * K.transpose();

        injectError(dx);
        symmetrizeCovariance();

        ros::WallDuration update_elapsed = ros::WallTime::now() - t_update_start;
        ROS_INFO("[ESKF计时] 更新(measurementUpdate) 耗时: %.3f ms, 测量数=%d, 模式=%d",
                 update_elapsed.toSec() * 1e3, (int)H.rows(), (int)mode);

        return dx;
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

    //发布估计的位姿
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
    double publish_rate_hz_;

    bool init_from_pose_topic_;
    bool init_from_mocap_;
    bool tof_vertical_mode_;
    bool use_uwb_fake_height_;

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
    double uwb_fake_height_min_information_;

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

    bool has_mocap_truth_ = false;//保存最新的动捕真值
    Eigen::Matrix3d latest_mocap_truth_R_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d latest_mocap_truth_p_ = Eigen::Vector3d::Zero();

    bool initialized_;
    bool has_last_imu_pair_time_;
    ros::Time last_imu_pair_time_;

    std::vector<int> tag_ids_;
    std::vector<int> anchor_ids_;
    std::vector<Eigen::Vector3d> tag_positions_B_;
    std::vector<Eigen::Vector3d> anchor_positions_A_;
    Eigen::Vector3d tof_position_B_;
    Eigen::Vector3d tof_ray_B_;

    DistanceMap distances_cache_;
    // 当前批次已经收到的标签ID
    std::set<int> received_tag_ids_;
    std::mutex dist_mutex_;


    ros::Subscriber uwb_sub_;
    ros::Subscriber tof_sub_;
    ros::Subscriber mocap_sub_;
    ros::Subscriber init_pose_sub_;
    ros::Publisher pose_pub_;
    ros::Publisher mocap_truth_pub_;
    ros::Timer publish_timer_;

    message_filters::Subscriber<sensor_msgs::Imu> imu_a_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_b_sub_;
    boost::shared_ptr<ImuSynchronizer> imu_sync_;
};

int main(int argc, char **argv)
{
    // 设置locale以支持中文输出
    setlocale(LC_ALL, "zh_CN.UTF-8");
    setlocale(LC_NUMERIC, "C");  // 防止小数点变成逗号

    ros::init(argc, argv, "uwb_imu_tof_eskf_sim_node");
    UwbImuTofEskfNode node;
    ros::spin();
    return 0;
}
