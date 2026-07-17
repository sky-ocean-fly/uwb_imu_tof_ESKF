/**
 * @file uwb_localization_node.cpp
 * @brief Simulation version of the UWB+ToF two-step pose estimator
 *
 * 订阅：
 *   /nlink_linktrack_nodeframe2 (nlink_parser::LinktrackNodeframe2) - UWB标签与基站的距离数据
 *   /ground_distance (geometry_msgs::PointStamped) - TOF测量的高度数据（point.y为修正后的垂直地面距离）
 *
 * 发布：
 *   /optimization_estimated_pose (geometry_msgs::PoseStamped) - 估计的位姿
 *   /estimated_node_positions (visualization_msgs::MarkerArray) - 估计的标签位置（可视化）
 *
 * 算法流程（两步法）：
 *   第一步（节点定位）：使用标签与基站之间的距离，通过最小二乘+Ceres优化得到标签在基站坐标系下的x,y,z坐标
 *   第二步（高度替换）：用TOF测量解算出的高度（y值）替换掉标签的y坐标
 *   第三步（位姿估计）：使用SVD+Ceres优化，从标签位置估计旋转和平移
 */

#include <ros/ros.h>
#include <ros/console.h>
#include <nlink_parser/LinktrackNodeframe2.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <mutex>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <clocale>
#include <set>

#include "simulation_geometry.h"

#include "node_localization.h"
#include "pose_estimation.h"
#include "utils.h"
#include "config_loader.h"
#include "global_declare.h"

// ============================================================
// 全局变量定义（对应 global_declare.h 中的 extern 声明）
// ============================================================
int Simulate_NLOS_flag = 0;
double Weighted_Least_Squares_weight = 0.0;
int Weighted_Least_Squares_flag = 0;
int Use_ToF_IMU_Height_flag = 1;

// ============================================================
// 前向声明
// ============================================================
void performLocalization();
void publishResult(const Eigen::Matrix3d &rotation,
                   const Eigen::Vector3d &translation,
                   const Eigen::Vector3d &euler_angles,
                   double height,
                   const std::vector<Eigen::Vector3d> &node_positions);
void recordResult(const Eigen::Vector3d &euler_angles,
                  const Eigen::Vector3d &translation,
                  double height);
void publishMocapTruth(const Eigen::Matrix3d &rotation,
                       const Eigen::Vector3d &translation,
                       const ros::Time &stamp);

// ============================================================
// 全局配置
// ============================================================

// Simulation geometry shared by the generator, ESKF and optimization node.
static const std::vector<int> TAG_IDS =
    uwb_pose_estimation::simulation_geometry::tagIds();
static const std::vector<int> ANCHOR_IDS =
    uwb_pose_estimation::simulation_geometry::anchorIds();

// These aliases keep the original two-step implementation unchanged.
static const std::vector<Eigen::Vector3d> Q_J_M =
    uwb_pose_estimation::simulation_geometry::tagPositionsB();
static const std::vector<Eigen::Vector3d> ANCHOR_POSITIONS_M =
    uwb_pose_estimation::simulation_geometry::anchorPositionsA();
Eigen::Vector3d p_TOF_M =
    uwb_pose_estimation::simulation_geometry::tofPositionB();

static const std::vector<Eigen::Vector3d> Q_J_B_Calibration =
    uwb_pose_estimation::simulation_geometry::tagPositionsB();
static const std::vector<Eigen::Vector3d> ANCHOR_POSITIONS_A_Calibration =
    uwb_pose_estimation::simulation_geometry::anchorPositionsA();
Eigen::Vector3d p_TOF_B_Calibration =
    uwb_pose_estimation::simulation_geometry::tofPositionB();

// ============================================================
// 数据缓存
// ============================================================

// 距离缓存：distances_cache[tag_id][anchor_id] = distance
static std::map<int, std::map<int, double>> distances_cache;
static std::map<int, std::map<int, double>> ready_distances_cache;
static std::set<int> received_tag_ids;
static bool uwb_batch_ready = false;
static std::mutex distances_mutex;

// 高度缓存
static double current_vertical_height = 0.0;
static bool height_received = false;
static std::mutex height_mutex;

// IMU RPY 数据缓存
static double current_roll = 0.0;
static double current_pitch = 0.0;
static double current_yaw = 0.0;
static bool imu_rpy_received = false;
static std::mutex imu_mutex;
static ros::Time last_imu_time;

// ============================================================
// 动捕数据缓存：/sim/relative_truth_pose
// 表示 ^M_BM R 和 ^M P_BM  motion capture
// ============================================================
static Eigen::Matrix3d mc_R_M_BM = Eigen::Matrix3d::Identity();
static Eigen::Vector3d mc_p_M_BM = Eigen::Vector3d::Zero();

static bool mocap_received = false;
static ros::Time last_mocap_time;
static std::mutex mocap_mutex;

// Common perturbed initialization published by the simulator.
static Eigen::Matrix3d common_init_R_A_B = Eigen::Matrix3d::Identity();
static Eigen::Vector3d common_init_p_A_B = Eigen::Vector3d::Zero();
static bool common_init_received = false;
static Eigen::Matrix3d previous_estimated_R_A_B = Eigen::Matrix3d::Identity();
static Eigen::Vector3d previous_estimated_p_A_B = Eigen::Vector3d::Zero();
static bool previous_estimate_available = false;
static std::mutex init_pose_mutex;

// 时间戳
static ros::Time last_nodeframe_time;
static ros::Time last_height_time;

// 数据有效性超时（秒）
static const double DATA_TIMEOUT = 1.0;

// 输出文件
static std::ofstream result_file;
static bool file_initialized = false;

// ============================================================
// 辅助函数
// ============================================================

/**
 * @brief 初始化结果记录文件
 */
void initResultFile()
{
    if (file_initialized)
        return;

    // 生成带时间戳的文件名
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << "uwb_localization_results_"
        << std::put_time(&tm, "%Y%m%d_%H%M%S")
        << ".csv";

    result_file.open(oss.str());
    if (result_file.is_open())
    {
        result_file << "timestamp,"
                    << "est_trans_x,est_trans_y,est_trans_z,"
                    << "est_rot_z,est_rot_y,est_rot_x,"
                    << "height"
                    << std::endl;
        ROS_INFO("结果将保存到文件: %s", oss.str().c_str());
    }

    file_initialized = true;
}

/**
 * @brief 将标签ID映射到Q_J_B的索引
 * @param tag_id 标签ID（5,6,7,8）
 * @return 索引（0,1,2,3），如果无效返回-1
 */
int tagIdToIndex(int tag_id)
{
    for (size_t i = 0; i < TAG_IDS.size(); i++)
    {
        if (TAG_IDS[i] == tag_id)
            return static_cast<int>(i);
    }
    return -1;
}

/**
 * @brief 将基站ID映射到ANCHOR_POSITIONS的索引
 * @param anchor_id 基站ID（0,1,2,3,4,9）
 * @return 索引（0,1,2,3,4,5），如果无效返回-1
 */
int anchorIdToIndex(int anchor_id)
{
    for (size_t i = 0; i < ANCHOR_IDS.size(); i++)
    {
        if (ANCHOR_IDS[i] == anchor_id)
            return static_cast<int>(i);
    }
    return -1;
}

/**
 * @brief 检查数据是否有效（未超时）
 */
bool isDataValid()
{
    ros::Time now = ros::Time::now();

    if (!height_received)
    {
        ROS_WARN_THROTTLE(5.0, "尚未接收到高度数据");
        return false;
    }

    if (!imu_rpy_received)
    {
        ROS_WARN_THROTTLE(5.0, "尚未接收到IMU RPY数据");
        return false;
    }

    if (!mocap_received)
    {
        ROS_WARN_THROTTLE(5.0, "尚未接收到动捕数据 /sim/relative_truth_pose");
        return false;
    }

    if ((now - last_nodeframe_time).toSec() > DATA_TIMEOUT)
    {
        ROS_WARN_THROTTLE(5.0, "UWB数据超时（%.1f秒前）",
                          (now - last_nodeframe_time).toSec());
        return false;
    }

    if ((now - last_height_time).toSec() > DATA_TIMEOUT)
    {
        ROS_WARN_THROTTLE(5.0, "高度数据超时（%.1f秒前）",
                          (now - last_height_time).toSec());
        return false;
    }

    if ((now - last_imu_time).toSec() > DATA_TIMEOUT)
    {
        ROS_WARN_THROTTLE(5.0, "IMU数据超时");
        return false;
    }

    if ((now - last_mocap_time).toSec() > DATA_TIMEOUT)
    {
        ROS_WARN_THROTTLE(5.0, "动捕数据超时（%.1f秒前）",
                        (now - last_mocap_time).toSec());
        return false;
    }

    return true;
}

// ============================================================
// 回调函数
// ============================================================

/**
 * @brief /nlink_linktrack_nodeframe2 话题的回调函数
 *
 * 从消息中提取标签与基站之间的距离数据
 */
void nodeframeCallback(const nlink_parser::LinktrackNodeframe2::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(distances_mutex);

    const int tag_id = msg->id;
    if (tagIdToIndex(tag_id) < 0)
        return;

    std::map<int, double> &tag_distances = distances_cache[tag_id];
    tag_distances.clear();

    for (const auto &node : msg->nodes)
    {
        const int anchor_id = node.id;
        const double distance = static_cast<double>(node.dis);
        if (anchorIdToIndex(anchor_id) >= 0 && std::isfinite(distance) && distance > 0.0)
            tag_distances[anchor_id] = distance;
    }

    if (tag_distances.size() != ANCHOR_IDS.size())
    {
        distances_cache.erase(tag_id);
        received_tag_ids.erase(tag_id);
        return;
    }

    received_tag_ids.insert(tag_id);
    last_nodeframe_time = ros::Time::now();

    if (received_tag_ids.size() == TAG_IDS.size())
    {
        ready_distances_cache = distances_cache;
        distances_cache.clear();
        received_tag_ids.clear();
        uwb_batch_ready = true;
    }
}

/**
 * @brief /ground_distance 话题的回调函数
 *
 * 从消息中提取高度数据（y值）
 * 消息类型为 geometry_msgs::PointStamped，其中 point.y 为高度
 */
void groundDistanceCallback(const geometry_msgs::PointStamped::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(height_mutex);

    current_vertical_height = msg->point.y;
    height_received = true;
    last_height_time = ros::Time::now();

    ROS_DEBUG("接收到高度数据: y = %.4f m", current_vertical_height);
}

/**
 * @brief /imu/rpy 话题的回调函数
 *
 * 从消息中提取IMU RPY数据
 * 消息类型为 geometry_msgs::Vector3Stamped，其中 vector.x 为roll，vector.y 为pitch，vector.z 为yaw
 */
void imuRpyCallback(const geometry_msgs::Vector3::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(imu_mutex);
    current_roll = msg->x;
    current_pitch = msg->y;
    current_yaw = msg->z;
    imu_rpy_received = true;
    last_imu_time = ros::Time::now();
    ROS_DEBUG("IMU RPY: roll=%.2f, pitch=%.2f, yaw=%.2f",
              current_roll, current_pitch, current_yaw);
}

/**
 * @brief /sim/relative_truth_pose 话题回调
 *
 * VRPN输出 geometry_msgs::PoseStamped
 * pose.position    表示 ^M P_BM
 * pose.orientation 表示 ^M_BM R 对应的四元数
 */
void mocapRocketCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(mocap_mutex);
    Eigen::Quaterniond q(msg->pose.orientation.w,
                         msg->pose.orientation.x,
                         msg->pose.orientation.y,
                         msg->pose.orientation.z);
    q.normalize();

    // The simulator already publishes ^A R_B and ^A p_AB.
    mc_R_M_BM = q.toRotationMatrix();
    mc_p_M_BM = Eigen::Vector3d(msg->pose.position.x,
                                msg->pose.position.y,
                                msg->pose.position.z);
    mocap_received = true;
    last_mocap_time = ros::Time::now();
}

void initialPoseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(init_pose_mutex);
    Eigen::Quaterniond q(msg->pose.orientation.w,
                         msg->pose.orientation.x,
                         msg->pose.orientation.y,
                         msg->pose.orientation.z);
    q.normalize();
    common_init_R_A_B = q.toRotationMatrix();
    common_init_p_A_B = Eigen::Vector3d(msg->pose.position.x,
                                        msg->pose.position.y,
                                        msg->pose.position.z);
    common_init_received = true;
}

void localizationTimerCallback(const ros::TimerEvent& event)
{
    // Consume exactly one complete 6-tag batch per optimization update.
    {
        std::lock_guard<std::mutex> lock(distances_mutex);
        if (!uwb_batch_ready)
            return;
    }

    //检查其他数据是否有效
    if(!isDataValid()){
        return;
    }

    //执行位姿估计
    performLocalization();

}
// ============================================================
// 核心定位算法
// ============================================================

/**
 * @brief 执行两步法位姿估计
 *
 * 第一步：节点定位 - 使用距离数据估计每个标签的位置
 * 第二步：高度替换 - 用TOF测量的高度替换标签的y坐标
 * 第三步：位姿估计 - 使用SVD+Ceres估计旋转和平移
 */
void performLocalization()
{
    // ============================================================
    // 0. 坐标预处理：得到 NODE_B、ANCHOR_A、TOF_B
    // ============================================================

    bool use_init_mocap_data = true; // 有初始化动捕数据就设为true；没有就设为false

    Eigen::Matrix3d R_M_BM0 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p_M_BM0 = Eigen::Vector3d::Zero();

    Eigen::Quaterniond q(1, 0, 0, 0);  // 注意顺序是 w, x, y, z
    q.normalize();                     // 防止四元数不是单位四元数
    R_M_BM0 =  q.toRotationMatrix();

    p_M_BM0 << 0.071,0.494,1.030;

    Eigen::Matrix3d R_M_AM0 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p_M_AM0 = Eigen::Vector3d::Zero();

    if (use_init_mocap_data)
    {
        // TODO：这里填入动捕初始化时刻提供的 BM 数据
        // R_M_BM0 = ...
        // p_M_BM0 = ...

        // 如果你假设 M 与 AM 重合，那么这里保持单位阵和零向量即可
        R_M_AM0 = Eigen::Matrix3d::Identity();
        p_M_AM0 = Eigen::Vector3d::Zero();

        // 如果后面AM也由动捕定义，则改成：
        // R_M_AM0 = ...
        // p_M_AM0 = ...
    }

    CoordinatePrepResult coord =
        prepareCoordinatesForPoseEstimation(
            use_init_mocap_data,
            Q_J_M,              // 标签在M系
            ANCHOR_POSITIONS_M, // 基站在M系
            p_TOF_M,            // ToF在M系
            R_M_BM0,
            p_M_BM0,
            R_M_AM0,
            p_M_AM0);

    //初始化
    // const std::vector<Eigen::Vector3d> &Q_J_B = coord.node_B;
    // const std::vector<Eigen::Vector3d> &ANCHOR_POSITIONS_A = coord.anchor_A;
    // const Eigen::Vector3d &p_TOF_B = coord.tof_B;

    //初始化之后的
    const std::vector<Eigen::Vector3d> &Q_J_B = Q_J_B_Calibration;
    const std::vector<Eigen::Vector3d> &ANCHOR_POSITIONS_A = ANCHOR_POSITIONS_A_Calibration;
    const Eigen::Vector3d &p_TOF_B = p_TOF_B_Calibration;
    ROS_INFO("========== 坐标预处理结果 ==========");
    for (size_t i = 0; i < Q_J_B.size(); ++i)
    {
        ROS_INFO("NODE_B Tag %d: [%.6f, %.6f, %.6f]",
                 TAG_IDS[i],
                 Q_J_B[i].x(),
                 Q_J_B[i].y(),
                 Q_J_B[i].z());
    }

    for (size_t i = 0; i < ANCHOR_POSITIONS_A.size(); ++i)
    {
        ROS_INFO("ANCHOR_A Anchor %d: [%.6f, %.6f, %.6f]",
                 ANCHOR_IDS[i],
                 ANCHOR_POSITIONS_A[i].x(),
                 ANCHOR_POSITIONS_A[i].y(),
                 ANCHOR_POSITIONS_A[i].z());
    }

    ROS_INFO("TOF_B: [%.6f, %.6f, %.6f]",
             p_TOF_B.x(), p_TOF_B.y(), p_TOF_B.z());

    // ============================================================
    // 1. 获取高度、IMU、UWB距离、动捕数据
    // ============================================================
    double vertical_height;
    {
        std::lock_guard<std::mutex> lock(height_mutex);
        vertical_height = current_vertical_height;
    }

    double roll, pitch, yaw;
    {
        std::lock_guard<std::mutex> lock(imu_mutex);
        roll = current_roll;   // \phi
        pitch = current_pitch; // \theta
        yaw = current_yaw;     // \psi
    }
    ROS_INFO("当前IMU姿态：roll=%.2f, pitch=%.2f, yaw=%.2f deg", roll, pitch, yaw);

    // Get one immutable complete UWB batch; old measurements are never reused.
    std::map<int, std::map<int, double>> distances;
    {
        std::lock_guard<std::mutex> lock(distances_mutex);
        if (!uwb_batch_ready)
            return;
        distances = ready_distances_cache;
        ready_distances_cache.clear();
        uwb_batch_ready = false;
    }

    // 获取动捕数据
    Eigen::Matrix3d mc_R_M_BM_now = Eigen::Matrix3d::Identity();//动捕当前数据
    Eigen::Vector3d mc_p_M_BM_now = Eigen::Vector3d::Zero();
    {
        std::lock_guard<std::mutex> lock(mocap_mutex);

        // 实时动捕位姿：用于后面计算真值
        mc_R_M_BM_now = mc_R_M_BM;
        mc_p_M_BM_now = mc_p_M_BM;
    }

    // ============================================================
    // 2. 节点定位：估计标签在 A 坐标系下的位置
    // ============================================================
    ROS_INFO("==========================================");
    ROS_INFO("开始两步法位姿估计");
    ROS_INFO("当前高度 (TOF): %.4f m", vertical_height);

    int num_tags = TAG_IDS.size();
    std::vector<Eigen::Vector3d> estimated_node_positions_A(num_tags);

    for (size_t i = 0; i < TAG_IDS.size(); i++)
    {
        int tag_id = TAG_IDS[i];

        // 构建该标签到所有基站的距离向量
        std::vector<double> measured_distances;
        measured_distances.reserve(ANCHOR_IDS.size());

        for (int anchor_id : ANCHOR_IDS)
        {
            if (distances[tag_id].find(anchor_id) != distances[tag_id].end())
            {
                measured_distances.push_back(distances[tag_id][anchor_id]);
            }
        }

        if (measured_distances.size() < 4)
        {
            ROS_ERROR("标签%d的有效距离数据不足（只有%zu个）",
                      tag_id, measured_distances.size());
            return;
        }

        ROS_INFO("标签%d: 距离数据数量=%zu", tag_id, measured_distances.size());

        // 使用最小二乘+Ceres优化估计标签位置
        // 先使用最小二乘得到初始估计
        Eigen::Vector3d initial_estimate =
            NodeLocalizer::estimateNodePositionLeastSquares(
                ANCHOR_POSITIONS_A, measured_distances);

        ROS_DEBUG("标签%d 初始估计: [%.4f, %.4f, %.4f]",
                  tag_id, initial_estimate[0], initial_estimate[1], initial_estimate[2]);

        // 使用Ceres优化
        ExperimentConfig dummy_config;
        dummy_config.node_position.max_iterations = 100;
        dummy_config.node_position.loss_function = 1.0;
        dummy_config.node_position.linear_solver_type = "DENSE_NORMAL_CHOLESKY";

        Eigen::Vector3d estimated_pos_A =
            NodeLocalizer::estimateNodePositionCeres(
                ANCHOR_POSITIONS_A, measured_distances, initial_estimate, dummy_config);

        ROS_INFO("标签%d 节点定位结果: [%.4f, %.4f, %.4f]",
                 tag_id, estimated_pos_A[0], estimated_pos_A[1], estimated_pos_A[2]);

        // ============================================================
        // 3. TOF高度替换
        // ============================================================
        double cos_theta = std::cos(pitch * M_PI / 180.0); // pitch = theta
        double sin_theta = std::sin(pitch * M_PI / 180.0);
        double cos_phi = std::cos(roll * M_PI / 180.0); // roll = phi
        double sin_phi = std::sin(roll * M_PI / 180.0);
        Eigen::Vector3d dir(-sin_theta, cos_theta * sin_phi, cos_theta * cos_phi);
        estimated_pos_A[2] = vertical_height + (Q_J_B[i] - p_TOF_B).dot(dir);
        ; // 替换y坐标为TOF测量的高度

        ROS_INFO("标签%d 高度替换后: [%.4f, %.4f, %.4f]",
                 tag_id, estimated_pos_A[0], estimated_pos_A[1], estimated_pos_A[2]);

        estimated_node_positions_A[i] = estimated_pos_A;
    }

    // ============================================================
    // 4. 位姿估计：用 NODE_B 和 NODE_A 求 ^A_B T
    // ============================================================
    ROS_INFO("--- 位姿估计 ---");
    auto [initial_rotation_A_B, initial_translation_A_B] =
        PoseEstimator::estimatePoseSVD(
            estimated_node_positions_A,  // 标签在A系下的位置
            Q_J_B                       // 标签在B系下的安装坐标
        );

    Eigen::Vector3d init_euler =
        Utils::rotationMatrixToEuler(initial_rotation_A_B);

    ROS_INFO("SVD初始估计 ^A_B T:");
    ROS_INFO("  R_A_B Euler(ZYX): [%.4f, %.4f, %.4f] deg",
             init_euler[0], init_euler[1], init_euler[2]);
    ROS_INFO("  t_A_B: [%.4f, %.4f, %.4f] m",
             initial_translation_A_B[0],
             initial_translation_A_B[1],
             initial_translation_A_B[2]);

    ExperimentConfig pose_config;
    pose_config.pose_estimation.max_iterations = 200;
    pose_config.pose_estimation.linear_solver_type = "DENSE_NORMAL_CHOLESKY";

    Eigen::Matrix3d pose_initial_R = initial_rotation_A_B;
    Eigen::Vector3d pose_initial_p = initial_translation_A_B;
    {
        std::lock_guard<std::mutex> lock(init_pose_mutex);
        if (previous_estimate_available)
        {
            pose_initial_R = previous_estimated_R_A_B;
            pose_initial_p = previous_estimated_p_A_B;
        }
        else if (common_init_received)
        {
            pose_initial_R = common_init_R_A_B;
            pose_initial_p = common_init_p_A_B;
        }
    }

    auto [estimated_rotation_A_B, estimated_translation_A_B] =
        PoseEstimator::estimatePoseCeresKnownRollPitch(
            estimated_node_positions_A,
            Q_J_B,
            pose_initial_R,
            pose_initial_p,
            roll * M_PI / 180.0,
            pitch * M_PI / 180.0,
            pose_config
        );

    // /ground_distance 的 point.y 是 ToF 点在 A 系中的竖直高度：
    // h_tof = e_z^T (t_A_B + R_A_B * p_TOF_B)。
    // 优化结束后，用它反推出 B 系原点的竖直高度，替代优化得到的相对 z。
    const double optimized_relative_z = estimated_translation_A_B.z();
    const double body_origin_height =
        vertical_height - (estimated_rotation_A_B * p_TOF_B).z();
    estimated_translation_A_B.z() = body_origin_height;

    ROS_INFO("ToF修正机体原点高度: optimized_z=%.4f m, tof_point_height=%.4f m, "
             "tof_offset_z=%.4f m, corrected_z=%.4f m",
             optimized_relative_z,
             vertical_height,
             (estimated_rotation_A_B * p_TOF_B).z(),
             estimated_translation_A_B.z());

    {
        std::lock_guard<std::mutex> lock(init_pose_mutex);
        previous_estimated_R_A_B = estimated_rotation_A_B;
        previous_estimated_p_A_B = estimated_translation_A_B;
        previous_estimate_available = true;
    }

    Eigen::Vector3d est_euler =
        Utils::rotationMatrixToEuler(estimated_rotation_A_B);

    ROS_INFO("==========================================");
    ROS_INFO("最终估计结果 ^A_B T:");
    ROS_INFO("  R_A_B Euler(ZYX): [%.4f, %.4f, %.4f] deg",
             est_euler[0], est_euler[1], est_euler[2]);
    ROS_INFO("  t_A_B: [%.4f, %.4f, %.4f] m",
             estimated_translation_A_B[0],
             estimated_translation_A_B[1],
             estimated_translation_A_B[2]);
    ROS_INFO("==========================================");

    publishResult(
        estimated_rotation_A_B,
        estimated_translation_A_B,
        est_euler,
        vertical_height,
        estimated_node_positions_A
    );

    recordResult(est_euler, estimated_translation_A_B, vertical_height);

    // ============================================================
    // 5. 使用动捕实时位姿计算真值 ^A_B T，并评估误差
    // ============================================================
    // Direct simulation truth: no mocap marker-frame conversion.
    const Eigen::Matrix3d R_A_B_mocap = mc_R_M_BM_now;
    const Eigen::Vector3d t_A_B_mocap = mc_p_M_BM_now;

    ros::Time compare_stamp = ros::Time::now();

    publishMocapTruth(
        R_A_B_mocap,
        t_A_B_mocap,
        compare_stamp
    );

    Eigen::Vector3d mocap_euler =
        Utils::rotationMatrixToEuler(R_A_B_mocap);

    double translation_error =
        (estimated_translation_A_B - t_A_B_mocap).norm();

    Eigen::Matrix3d R_error =
        estimated_rotation_A_B * R_A_B_mocap.transpose();

    double rotation_error_deg =
        Eigen::AngleAxisd(R_error).angle() * 180.0 / M_PI;

    ROS_INFO("========== 动捕真值 ^A_B T ==========");
    ROS_INFO("  mocap R_A_B Euler(ZYX): [%.4f, %.4f, %.4f] deg",
            mocap_euler[0], mocap_euler[1], mocap_euler[2]);

    ROS_INFO("  mocap t_A_B: [%.4f, %.4f, %.4f] m",
            t_A_B_mocap.x(),
            t_A_B_mocap.y(),
            t_A_B_mocap.z());

    ROS_INFO("========== UWB估计误差 ==========");
    ROS_INFO("  平移误差: %.6f m", translation_error);
    ROS_INFO("  旋转误差: %.6f deg", rotation_error_deg);
    
}

// ============================================================
// 结果发布与记录
// ============================================================

static ros::Publisher pose_pub;
static ros::Publisher marker_pub;
static ros::Publisher mocap_truth_pub;

/**
 * @brief 发布估计结果
 */
void publishResult(const Eigen::Matrix3d &rotation,
                   const Eigen::Vector3d &translation,
                   const Eigen::Vector3d &euler_angles,
                   double height,
                   const std::vector<Eigen::Vector3d> &node_positions)
{
    // 发布位姿
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = ros::Time::now();
    pose_msg.header.frame_id = "anchor_frame_A";

    Eigen::Quaterniond q(rotation);
    pose_msg.pose.position.x = translation[0];
    pose_msg.pose.position.y = translation[1];
    pose_msg.pose.position.z = translation[2];
    pose_msg.pose.orientation.w = q.w();
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();

    pose_pub.publish(pose_msg);

    // 发布标签位置标记（用于可视化）
    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i < node_positions.size(); i++)
    {
        visualization_msgs::Marker marker;
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = "anchor_frame_A";
        marker.ns = "estimated_nodes";
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = node_positions[i][0];
        marker.pose.position.y = node_positions[i][1];
        marker.pose.position.z = node_positions[i][2];
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        marker_array.markers.push_back(marker);
    }

    marker_pub.publish(marker_array);
}

/**
 * @brief 记录结果到CSV文件
 */
void recordResult(const Eigen::Vector3d &euler_angles,
                  const Eigen::Vector3d &translation,
                  double height)
{
    if (!result_file.is_open())
        return;

    ros::Time now = ros::Time::now();
    result_file << now.toSec() << ","
                << translation[0] << "," << translation[1] << "," << translation[2] << ","
                << euler_angles[0] << "," << euler_angles[1] << "," << euler_angles[2] << ","
                << height
                << std::endl;
}

void publishMocapTruth(const Eigen::Matrix3d &rotation,
                       const Eigen::Vector3d &translation,
                       const ros::Time &stamp)
{
    geometry_msgs::PoseStamped pose_msg;

    pose_msg.header.stamp = stamp;

    // 注意：这里表示的是火箭B相对于平台A的位姿，不是动捕world下的位姿
    pose_msg.header.frame_id = "anchor_frame_A";

    Eigen::Quaterniond q(rotation);
    q.normalize();

    pose_msg.pose.position.x = translation.x();
    pose_msg.pose.position.y = translation.y();
    pose_msg.pose.position.z = translation.z();

    pose_msg.pose.orientation.w = q.w();
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();

    mocap_truth_pub.publish(pose_msg);
}

// ============================================================
// 主函数
// ============================================================

int main(int argc, char **argv)
{
    // 设置locale以支持中文输出
    setlocale(LC_ALL, "zh_CN.UTF-8");
    setlocale(LC_NUMERIC, "C");  // 防止小数点变成逗号

    ros::init(argc, argv, "uwb_localization_sim_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    ROS_INFO("==========================================");
    ROS_INFO("UWB定位节点启动");
    ROS_INFO("==========================================");

    // 初始化结果文件
    initResultFile();

    // 订阅话题
    ros::Subscriber nodeframe_sub = nh.subscribe(
        "/nlink_linktrack_nodeframe2", 10, nodeframeCallback);

    ros::Subscriber height_sub = nh.subscribe(
        "/ground_distance", 10, groundDistanceCallback);

    ros::Subscriber imu_sub = nh.subscribe("/imu/rpy", 10, imuRpyCallback);

    ros::Subscriber mocap_sub = nh.subscribe(
        "/sim/relative_truth_pose", 10, mocapRocketCallback);

    ros::Subscriber initial_pose_sub = nh.subscribe(
        "/sim/initial_pose", 1, initialPoseCallback);

    // 发布话题
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>(
        "/optimization_estimated_pose", 10);

    marker_pub = nh.advertise<visualization_msgs::MarkerArray>(
        "/estimated_node_positions", 10);

    mocap_truth_pub = nh.advertise<geometry_msgs::PoseStamped>(
        "/optimization_truth_pose", 10);

    ros::Timer localization_timer = nh.createTimer(ros::Duration(0.02),
                                                    localizationTimerCallback);

    ROS_INFO("已订阅话题:");
    ROS_INFO("  - /nlink_linktrack_nodeframe2");
    ROS_INFO("  - /ground_distance");
    ROS_INFO("  - /sim/relative_truth_pose");
    ROS_INFO("已发布话题:");
    ROS_INFO("  - /optimization_estimated_pose");
    ROS_INFO("  - /estimated_node_positions");
    ROS_INFO("  - /optimization_truth_pose");
    ROS_INFO("等待数据...");

    ros::spin();

    // 关闭文件
    if (result_file.is_open())
    {
        result_file.close();
    }

    return 0;
}
