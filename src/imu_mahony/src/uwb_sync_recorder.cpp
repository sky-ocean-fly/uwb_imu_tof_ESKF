#include <ros/ros.h>
#include <geometry_msgs/Vector3.h>
#include <nlink_parser/LinktrackNodeframe2.h>

#include <fstream>
#include <sstream>
#include <string>
#include <mutex>
#include <ctime>
#include <iomanip>

// ============================================================
// UWB 同步记录节点
//
// 功能:
//   当 /nlink_linktrack_nodeframe2 (慢速) 话题发布时，
//   同时记录 /nlink_linktrack_nodeframe2 和 /ground_distance (快速) 的数据，
//   写入 CSV 文件，实现时间同步。
//
// 订阅:
//   /nlink_linktrack_nodeframe2  (nlink_parser::LinktrackNodeframe2)
//   /ground_distance             (geometry_msgs::Vector3)
//
// 输出:
//   CSV 文件，包含两部分的字段:
//     第一部分: /nlink_linktrack_nodeframe2 的字段 (与 rostopic echo -p 格式一致)
//     第二部分: /ground_distance 的字段 (x, y, z)
// ============================================================

class UWBSyncRecorder {
public:
    UWBSyncRecorder()
        : nh_("~")
        , has_ground_(false)
        , csv_initialized_(false)
    {
        // ---- 参数 ----
        nh_.param<std::string>("uwb_topic", uwb_topic_,
                               std::string("/nlink_linktrack_nodeframe2"));
        nh_.param<std::string>("ground_topic", ground_topic_,
                               std::string("/ground_distance"));
        nh_.param<std::string>("output_file", output_file_,
                               std::string("uwb_sync_record.csv"));

        // ---- 订阅 ----
        sub_uwb_ = nh_.subscribe(uwb_topic_, 10,
                                 &UWBSyncRecorder::uwbCallback, this);
        sub_ground_ = nh_.subscribe(ground_topic_, 10,
                                    &UWBSyncRecorder::groundCallback, this);

        ROS_INFO("UWBSyncRecorder started.");
        ROS_INFO("  Subscribe UWB    : %s", uwb_topic_.c_str());
        ROS_INFO("  Subscribe Ground : %s", ground_topic_.c_str());
        ROS_INFO("  Output file      : %s", output_file_.c_str());
    }

    ~UWBSyncRecorder() {
        if (csv_file_.is_open()) {
            csv_file_.close();
            ROS_INFO("CSV file closed: %s", output_file_.c_str());
        }
    }

private:
    // ---------- UWB 回调 (触发记录) ----------
    void uwbCallback(const nlink_parser::LinktrackNodeframe2::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mtx_);

        // 打开文件 (首次调用时)
        if (!csv_file_.is_open()) {
            csv_file_.open(output_file_.c_str(), std::ios::out | std::ios::trunc);
            if (!csv_file_.is_open()) {
                ROS_ERROR("Failed to open output file: %s", output_file_.c_str());
                return;
            }
            // 写入 BOM (UTF-8 BOM 可选，便于 Excel 打开)
            // csv_file_ << "\xEF\xBB\xBF";
            writeHeader();
        }

        // 写入 UWB 数据行
        writeUWBData(msg);

        // 如果有 ground_distance 数据，追加在同一行
        if (has_ground_) {
            writeGroundData();
        } else {
            // 如果没有 ground_distance 数据，补空
            csv_file_ << ",,,";
        }
        csv_file_ << "\n";
        csv_file_.flush();
    }

    // ---------- Ground Distance 回调 (缓存最新数据) ----------
    void groundCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        last_ground_ = *msg;
        has_ground_ = true;
    }

    // ---------- 写入 CSV 表头 ----------
    void writeHeader() {
        // /nlink_linktrack_nodeframe2 的字段
        csv_file_ << "%time"
                  << ",field.role"
                  << ",field.id"
                  << ",field.local_time"
                  << ",field.system_time"
                  << ",field.voltage"
                  << ",field.pos_3d0"
                  << ",field.pos_3d1"
                  << ",field.pos_3d2"
                  << ",field.eop_3d0"
                  << ",field.eop_3d1"
                  << ",field.eop_3d2"
                  << ",field.vel_3d0"
                  << ",field.vel_3d1"
                  << ",field.vel_3d2"
                  << ",field.angle_3d0"
                  << ",field.angle_3d1"
                  << ",field.angle_3d2"
                  << ",field.quaternion0"
                  << ",field.quaternion1"
                  << ",field.quaternion2"
                  << ",field.quaternion3"
                  << ",field.imu_gyro_3d0"
                  << ",field.imu_gyro_3d1"
                  << ",field.imu_gyro_3d2"
                  << ",field.imu_acc_3d0"
                  << ",field.imu_acc_3d1"
                  << ",field.imu_acc_3d2";

        // nodes 数组字段 (最多支持 9 个节点)
        for (int i = 0; i < 9; ++i) {
            csv_file_ << ",field.nodes" << i << ".role"
                      << ",field.nodes" << i << ".id"
                      << ",field.nodes" << i << ".dis"
                      << ",field.nodes" << i << ".fp_rssi"
                      << ",field.nodes" << i << ".rx_rssi";
        }

        // /ground_distance 的字段
        csv_file_ << ",ground_time"
                  << ",ground.x"
                  << ",ground.y"
                  << ",ground.z";

        csv_file_ << "\n";
    }

    // ---------- 写入 UWB 数据 ----------
    void writeUWBData(const nlink_parser::LinktrackNodeframe2::ConstPtr& msg) {
        // 时间戳 (纳秒)
        ros::Time now = ros::Time::now();
        csv_file_ << std::setprecision(16) << now.toNSec();

        csv_file_ << "," << static_cast<int>(msg->role);
        csv_file_ << "," << static_cast<int>(msg->id);
        csv_file_ << "," << msg->local_time;
        csv_file_ << "," << msg->system_time;
        csv_file_ << "," << msg->voltage;

        // pos_3d
        csv_file_ << "," << msg->pos_3d[0]
                  << "," << msg->pos_3d[1]
                  << "," << msg->pos_3d[2];

        // eop_3d
        csv_file_ << "," << msg->eop_3d[0]
                  << "," << msg->eop_3d[1]
                  << "," << msg->eop_3d[2];

        // vel_3d
        csv_file_ << "," << msg->vel_3d[0]
                  << "," << msg->vel_3d[1]
                  << "," << msg->vel_3d[2];

        // angle_3d
        csv_file_ << "," << msg->angle_3d[0]
                  << "," << msg->angle_3d[1]
                  << "," << msg->angle_3d[2];

        // quaternion
        csv_file_ << "," << msg->quaternion[0]
                  << "," << msg->quaternion[1]
                  << "," << msg->quaternion[2]
                  << "," << msg->quaternion[3];

        // imu_gyro_3d
        csv_file_ << "," << msg->imu_gyro_3d[0]
                  << "," << msg->imu_gyro_3d[1]
                  << "," << msg->imu_gyro_3d[2];

        // imu_acc_3d
        csv_file_ << "," << msg->imu_acc_3d[0]
                  << "," << msg->imu_acc_3d[1]
                  << "," << msg->imu_acc_3d[2];

        // nodes 数组
        size_t node_count = msg->nodes.size();
        for (size_t i = 0; i < 9; ++i) {
            if (i < node_count) {
                const auto& node = msg->nodes[i];
                csv_file_ << "," << static_cast<int>(node.role)
                          << "," << static_cast<int>(node.id)
                          << "," << node.dis
                          << "," << node.fp_rssi
                          << "," << node.rx_rssi;
            } else {
                csv_file_ << ",,,,,";
            }
        }
    }

    // ---------- 写入 Ground Distance 数据 ----------
    void writeGroundData() {
        ros::Time now = ros::Time::now();
        csv_file_ << "," << std::setprecision(16) << now.toNSec()
                  << "," << last_ground_.x
                  << "," << last_ground_.y
                  << "," << last_ground_.z;
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_uwb_;
    ros::Subscriber sub_ground_;

    std::mutex mtx_;
    std::ofstream csv_file_;

    bool has_ground_;
    bool csv_initialized_;

    geometry_msgs::Vector3 last_ground_;

    std::string uwb_topic_;
    std::string ground_topic_;
    std::string output_file_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "uwb_sync_recorder");
    UWBSyncRecorder node;
    ros::spin();
    return 0;
}
