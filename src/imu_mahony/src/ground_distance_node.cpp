#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Vector3.h>
#include <nlink_parser/TofsenseMFrame0.h>

#include <cmath>
#include <string>
#include <vector>
#include <mutex>
#include <deque>

// ============================================================
// 地面距离解算节点
//
// 订阅:
//   /nlink_tofsensem_frame0  (nlink_parser::TofsenseMFrame0)
//       多像素 ToF 传感器数据，取中间像素的 dis (mm)
//   /imu/rpy                 (geometry_msgs::Vector3)
//       IMU 解算出的欧拉角: x=roll, y=pitch, z=yaw (单位: deg)
//
// 发布:
//   /ground_distance         (geometry_msgs::PointStamped)
//       point.x: 原始 ToF 测量距离 (m)
//       point.y: 修正后的垂直地面距离 (m)
//       point.z: 保留 (0.0)
//
// 原理:
//   ToF 传感器朝下安装，测量值为沿传感器光束方向的斜距 d。
//   当机体有倾斜时，真实垂直高度 = d * cos(pitch) * cos(roll)
//   (假设 roll/pitch 均为弧度)
// ============================================================

class GroundDistanceNode {
public:
    GroundDistanceNode()
        : nh_("~")
        , has_rpy_(false)
        , has_tof_(false)
        , last_tof_dis_mm_(0.0f)
        , pub_count_(0)
    {
        // ---- 参数 ----
        nh_.param<std::string>("tof_topic", tof_topic_,
                               std::string("/nlink_tofsensem_frame0"));
        nh_.param<std::string>("rpy_topic", rpy_topic_,
                               std::string("/imu/rpy"));
        nh_.param<std::string>("ground_topic", ground_topic_,
                               std::string("/ground_distance"));

        // 64 个像素，中间像素索引 = 31 (0-based)
        // 如果 pixel_count 是奇数，取正中间；64 是偶数，取 31 或 32 均可
        nh_.param("middle_pixel_index", middle_pixel_index_, 7);

        // ---- 订阅 ----
        sub_tof_ = nh_.subscribe(tof_topic_, 10,
                                 &GroundDistanceNode::tofCallback, this);
        sub_rpy_ = nh_.subscribe(rpy_topic_, 10,
                                 &GroundDistanceNode::rpyCallback, this);

        // ---- 发布 ----
        pub_ground_ = nh_.advertise<geometry_msgs::PointStamped>(ground_topic_, 10);

        // ---- 50Hz 固定频率发布定时器 (20ms) ----
        pub_timer_ = nh_.createTimer(ros::Duration(0.02),
                                     &GroundDistanceNode::pubTimerCallback, this);

        // ---- 帧率统计定时器 (每 5 秒打印一次) ----
        fps_timer_ = nh_.createTimer(ros::Duration(5.0),
                                     &GroundDistanceNode::fpsTimerCallback, this);

        ROS_INFO("GroundDistanceNode started.");
        ROS_INFO("  Subscribe ToF : %s", tof_topic_.c_str());
        ROS_INFO("  Subscribe RPY : %s", rpy_topic_.c_str());
        ROS_INFO("  Publish Ground : %s (geometry_msgs::PointStamped)", ground_topic_.c_str());
        ROS_INFO("  Middle pixel index : %d", middle_pixel_index_);
        ROS_INFO("  Publish rate : 50 Hz");
        ROS_INFO("  FPS report interval : 5 sec");
    }

private:
    // ---------- ToF 回调 ----------
    void tofCallback(const nlink_parser::TofsenseMFrame0::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mtx_);

        // 检查像素数量
        if (msg->pixels.empty()) {
            ROS_WARN_THROTTLE(1.0, "ToF message has no pixels!");
            return;
        }

        // 取 4 个中心像素的平均值 (5, 6, 9, 10)
        int idx_4[4] = {5, 6, 9, 10};
        last_tof_dis_mm_ = 0.0f;
        for (int i = 0; i < 4; ++i) {
            auto &pixel = msg->pixels[idx_4[i]];
            last_tof_dis_mm_ += static_cast<float>(pixel.dis);
        }
        last_tof_dis_mm_ /= 4.0f;
        has_tof_ = true;

        // 数据更新由 50Hz 定时器统一发布
    }

    // ---------- IMU RPY 回调 ----------
    void rpyCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mtx_);

        roll_  = static_cast<float>(msg->x);   // deg
        pitch_ = static_cast<float>(msg->y);   // deg
        yaw_   = static_cast<float>(msg->z);   // deg
        has_rpy_ = true;

        // 数据更新由 50Hz 定时器统一发布
    }

    // ---------- 50Hz 定时器回调 ----------
    void pubTimerCallback(const ros::TimerEvent&) {
        std::lock_guard<std::mutex> lock(mtx_);

        // 只有同时拥有 ToF 和 RPY 数据时才发布
        if (!has_tof_ || !has_rpy_) {
            return;
        }

        publishGroundDistance();
    }

    // ---------- 计算并发布地面距离 ----------
    void publishGroundDistance() {
        // 角度转弧度 (用于计算垂直距离)
        float roll_rad  = roll_  * static_cast<float>(M_PI / 180.0);
        float pitch_rad = pitch_ * static_cast<float>(M_PI / 180.0);

        // ToF 距离: mm -> m
        float tof_dis_m = last_tof_dis_mm_ / 1000.0f;

        // 垂直地面距离: d * cos(pitch) * cos(roll)
        float vertical_height = tof_dis_m *
                                std::cos(pitch_rad) *
                                std::cos(roll_rad);

        geometry_msgs::PointStamped out;
        out.header.stamp = ros::Time::now();
        out.header.frame_id = "base_link";

        out.point.x = tof_dis_m;          // 原始 ToF 斜距 (m)
        out.point.y = vertical_height;    // 修正后的垂直地面距离 (m)
        out.point.z = 0.0;                // 保留

        pub_ground_.publish(out);
        pub_count_++;
    }

    // ---------- 帧率统计定时器 ----------
    void fpsTimerCallback(const ros::TimerEvent&) {
        std::lock_guard<std::mutex> lock(mtx_);

        // 计算过去 5 秒内的发布频率
        double fps = pub_count_ / 5.0;
        ROS_INFO("=== [FPS] /ground_distance publish rate: %.1f Hz (last 5 sec, %lu msgs) ===",
                 fps, pub_count_);
        pub_count_ = 0;
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_tof_;
    ros::Subscriber sub_rpy_;
    ros::Publisher  pub_ground_;
    ros::Timer      pub_timer_;
    ros::Timer      fps_timer_;

    std::mutex mtx_;

    bool  has_rpy_;
    bool  has_tof_;

    float roll_;    // deg
    float pitch_;   // deg
    float yaw_;     // deg

    float last_tof_dis_mm_;

    int   middle_pixel_index_;

    unsigned long pub_count_;

    std::string tof_topic_;
    std::string rpy_topic_;
    std::string ground_topic_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ground_distance_node");
    GroundDistanceNode node;
    ros::spin();
    return 0;
}
