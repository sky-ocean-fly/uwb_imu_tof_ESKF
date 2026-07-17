#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <algorithm>
#include <boost/bind.hpp>
#include <sys/stat.h>
#include <errno.h>
#include <clocale>
#include <limits>

class PoseCompareLogger
{
public:
    PoseCompareLogger()
        : pnh_("~"),
          has_t0_(false),
          t0_(0.0),
          sample_count_(0),
          sum_sq_pos_(0.0),
          sum_sq_rot_deg_(0.0),
          max_pos_(0.0),
          max_rot_deg_(0.0),
          reported_(false)
    {
        pnh_.param<std::string>("est_topic", est_topic_, "/estimated_pose");
        pnh_.param<std::string>("gt_topic", gt_topic_, "/mocap_truth_pose");
        pnh_.param<std::string>("csv_path", csv_path_, "/home/yu/catkin_workspace/dynamic_pose_compare1.csv");

        pnh_.param<int>("queue_size", queue_size_, 100);
        pnh_.param<double>("sync_slop", sync_slop_, 0.05);
        pnh_.param<std::string>("method_name", method_name_, "pose_estimator");
        pnh_.param<double>("evaluation_duration_sec", evaluation_duration_sec_, 0.0);
        pnh_.param<double>("simulation_start_stamp_sec", simulation_start_stamp_sec_, -1.0);

        // 尝试打开文件，如果失败则尝试创建目录后重试
        file_.open(csv_path_.c_str());

        if (!file_.is_open())
        {
            ROS_WARN("无法直接打开CSV文件: %s，尝试创建目录...", csv_path_.c_str());

            // 提取目录路径并尝试创建
            std::string dir_path = getParentDir(csv_path_);
            if (!dir_path.empty())
            {
                if (createDirectories(dir_path))
                {
                    // 创建目录成功后重试打开文件
                    file_.open(csv_path_.c_str());
                }
            }

            if (!file_.is_open())
            {
                ROS_FATAL("无法创建或打开CSV文件: %s", csv_path_.c_str());
                ros::shutdown();
                return;
            }
        }

        file_ << std::fixed << std::setprecision(9);

        file_ << "time,"
              << "stamp_est,stamp_gt,"
              << "est_x,est_y,est_z,"
              << "gt_x,gt_y,gt_z,"
              << "err_x,err_y,err_z,err_pos_norm,"
              << "est_roll,est_pitch,est_yaw,"
              << "gt_roll,gt_pitch,gt_yaw,"
              << "err_roll,err_pitch,err_yaw,"
              << "err_rot_deg"
              << "\n";

        est_sub_.subscribe(nh_, est_topic_, queue_size_);
        gt_sub_.subscribe(nh_, gt_topic_, queue_size_);

        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(
            SyncPolicy(queue_size_),
            est_sub_,
            gt_sub_));

        sync_->setMaxIntervalDuration(ros::Duration(sync_slop_));

        sync_->registerCallback(
            boost::bind(&PoseCompareLogger::callback, this, _1, _2));

        ROS_INFO("PoseCompareLogger启动");
        ROS_INFO("  算法估计话题: %s", est_topic_.c_str());
        ROS_INFO("  动捕真值话题: %s", gt_topic_.c_str());
        ROS_INFO("  CSV保存路径: %s", csv_path_.c_str());
        ROS_INFO("  同步时间容差: %.3f s", sync_slop_);
        if (evaluation_duration_sec_ > 0.0)
        {
            ROS_INFO("  误差统计区间: %.3f s", evaluation_duration_sec_);
        }
    }

    ~PoseCompareLogger()
    {
        // Also provide a partial result when the node is stopped early.
        if (!reported_ && sample_count_ > 0)
        {
            reportStatistics("节点退出，以下为当前已采集区间的统计");
        }
        if (file_.is_open())
        {
            file_.close();
        }
    }

private:
    using SyncPolicy =
        message_filters::sync_policies::ApproximateTime<
            geometry_msgs::PoseStamped,
            geometry_msgs::PoseStamped>;

    static double rad2deg(double rad)
    {
        return rad * 180.0 / M_PI;
    }

    static double clampValue(double value, double min_value, double max_value)
    {
        return std::max(min_value, std::min(max_value, value));
    }

    static double wrapToPi(double angle)
    {
        while (angle > M_PI)
        {
            angle -= 2.0 * M_PI;
        }

        while (angle < -M_PI)
        {
            angle += 2.0 * M_PI;
        }

        return angle;
    }

    static Eigen::Vector3d getPosition(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        return Eigen::Vector3d(
            msg->pose.position.x,
            msg->pose.position.y,
            msg->pose.position.z);
    }

    static Eigen::Quaterniond getQuaternion(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        Eigen::Quaterniond q(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z);

        q.normalize();
        return q;
    }

    static Eigen::Vector3d quaternionToRPY(const Eigen::Quaterniond &q)
    {
        double w = q.w();
        double x = q.x();
        double y = q.y();
        double z = q.z();

        double roll = std::atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y));

        double sin_pitch = 2.0 * (w * y - z * x);
        sin_pitch = clampValue(sin_pitch, -1.0, 1.0);

        double pitch = std::asin(sin_pitch);

        double yaw = std::atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z));

        return Eigen::Vector3d(roll, pitch, yaw);
    }

    static double rotationErrorDeg(const Eigen::Matrix3d &R_est,
                                   const Eigen::Matrix3d &R_gt)
    {
        Eigen::Matrix3d R_err = R_gt.transpose() * R_est;

        double cos_angle = (R_err.trace() - 1.0) / 2.0;
        cos_angle = clampValue(cos_angle, -1.0, 1.0);

        double angle = std::acos(cos_angle);

        return rad2deg(angle);
    }

    // 从文件路径中提取目录路径
    static std::string getParentDir(const std::string &file_path)
    {
        size_t pos = file_path.find_last_of("/\\");
        if (pos != std::string::npos)
        {
            return file_path.substr(0, pos);
        }
        return "";
    }

    // 递归创建目录
    static bool createDirectories(const std::string &dir_path)
    {
        // 先检查目录是否已存在
        struct stat info;
        if (stat(dir_path.c_str(), &info) == 0)
        {
            if (info.st_mode & S_IFDIR)
            {
                return true; // 目录已存在
            }
            ROS_ERROR("路径已存在但不是目录: %s", dir_path.c_str());
            return false;
        }

        // 递归创建父目录
        std::string parent = getParentDir(dir_path);
        if (!parent.empty() && !createDirectories(parent))
        {
            return false;
        }

        // 创建当前目录
        if (mkdir(dir_path.c_str(), 0755) != 0)
        {
            ROS_ERROR("创建目录失败 %s: %s", dir_path.c_str(), strerror(errno));
            return false;
        }

        ROS_INFO("已创建目录: %s", dir_path.c_str());
        return true;
    }

    void callback(const geometry_msgs::PoseStamped::ConstPtr &est_msg,
                  const geometry_msgs::PoseStamped::ConstPtr &gt_msg)
    {
        if (!file_.is_open())
        {
            return;
        }

        double stamp_est = est_msg->header.stamp.toSec();
        double stamp_gt = gt_msg->header.stamp.toSec();

        double stamp = 0.5 * (stamp_est + stamp_gt);

        if (reported_)
        {
            return;
        }

        if (!has_t0_)
        {
            t0_ = simulation_start_stamp_sec_ >= 0.0
                      ? simulation_start_stamp_sec_
                      : stamp;
            has_t0_ = true;
        }

        double time = stamp - t0_;
        if (time < 0.0)
        {
            return;
        }

        Eigen::Vector3d t_est = getPosition(est_msg);
        Eigen::Vector3d t_gt = getPosition(gt_msg);

        Eigen::Quaterniond q_est = getQuaternion(est_msg);
        Eigen::Quaterniond q_gt = getQuaternion(gt_msg);

        Eigen::Matrix3d R_est = q_est.toRotationMatrix();
        Eigen::Matrix3d R_gt = q_gt.toRotationMatrix();

        Eigen::Vector3d rpy_est = quaternionToRPY(q_est);
        Eigen::Vector3d rpy_gt = quaternionToRPY(q_gt);

        Eigen::Vector3d pos_err = t_est - t_gt;
        double err_pos_norm = pos_err.norm();

        Eigen::Vector3d rpy_err;
        rpy_err.x() = wrapToPi(rpy_est.x() - rpy_gt.x());
        rpy_err.y() = wrapToPi(rpy_est.y() - rpy_gt.y());
        rpy_err.z() = wrapToPi(rpy_est.z() - rpy_gt.z());

        double err_rot_deg = rotationErrorDeg(R_est, R_gt);

        ++sample_count_;
        sum_sq_pos_ += static_cast<long double>(err_pos_norm) * err_pos_norm;
        sum_sq_rot_deg_ += static_cast<long double>(err_rot_deg) * err_rot_deg;
        max_pos_ = std::max(max_pos_, err_pos_norm);
        max_rot_deg_ = std::max(max_rot_deg_, err_rot_deg);

        file_ << time << ","
              << stamp_est << "," << stamp_gt << ","

              << t_est.x() << "," << t_est.y() << "," << t_est.z() << ","
              << t_gt.x() << "," << t_gt.y() << "," << t_gt.z() << ","

              << pos_err.x() << "," << pos_err.y() << "," << pos_err.z() << ","
              << err_pos_norm << ","

              << rad2deg(rpy_est.x()) << ","
              << rad2deg(rpy_est.y()) << ","
              << rad2deg(rpy_est.z()) << ","

              << rad2deg(rpy_gt.x()) << ","
              << rad2deg(rpy_gt.y()) << ","
              << rad2deg(rpy_gt.z()) << ","

              << rad2deg(rpy_err.x()) << ","
              << rad2deg(rpy_err.y()) << ","
              << rad2deg(rpy_err.z()) << ","

              << err_rot_deg
              << "\n";

        file_.flush();

        // The simulator publishes a sample exactly at t=duration, so include it
        // before printing the final result.
        if (evaluation_duration_sec_ > 0.0 &&
            time + 1e-9 >= evaluation_duration_sec_)
        {
            reportStatistics("仿真评估完成");
        }
    }

    void reportStatistics(const std::string &reason)
    {
        if (reported_ || sample_count_ == 0)
        {
            return;
        }

        const double position_rmse =
            std::sqrt(static_cast<double>(sum_sq_pos_ / sample_count_));
        const double attitude_rmse_deg =
            std::sqrt(static_cast<double>(sum_sq_rot_deg_ / sample_count_));

        ROS_INFO("\n========== %s 误差统计 ==========\n"
                 "状态: %s\n"
                 "有效同步样本数: %zu\n"
                 "位置 RMSE: %.6f m\n"
                 "位置最大误差: %.6f m\n"
                 "姿态 RMSE: %.6f °\n"
                 "姿态最大误差: %.6f °\n"
                 "=====================================",
                 method_name_.c_str(), reason.c_str(), sample_count_,
                 position_rmse, max_pos_, attitude_rmse_deg, max_rot_deg_);
        reported_ = true;
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    std::string est_topic_;
    std::string gt_topic_;
    std::string csv_path_;
    std::string method_name_;

    int queue_size_;
    double sync_slop_;
    double evaluation_duration_sec_;
    double simulation_start_stamp_sec_;

    message_filters::Subscriber<geometry_msgs::PoseStamped> est_sub_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> gt_sub_;

    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    std::ofstream file_;

    bool has_t0_;
    double t0_;
    std::size_t sample_count_;
    long double sum_sq_pos_;
    long double sum_sq_rot_deg_;
    double max_pos_;
    double max_rot_deg_;
    bool reported_;
};

int main(int argc, char **argv)
{
    // 设置locale以支持中文输出
    setlocale(LC_ALL, "zh_CN.UTF-8");
    setlocale(LC_NUMERIC, "C");  // 防止小数点变成逗号

    ros::init(argc, argv, "pose_compare_logger");

    PoseCompareLogger logger;

    ros::spin();

    return 0;
}
