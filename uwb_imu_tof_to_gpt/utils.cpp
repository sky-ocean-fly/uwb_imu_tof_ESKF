#include "utils.h"
#include "global_declare.h"
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <map>
#include <sstream>


std::default_random_engine& Utils::getRandomEngine(int seed) {
    static std::default_random_engine engine;
    if (seed == -1) {
        std::random_device rd;
        engine.seed(rd());
    } else {
        engine.seed(seed);
    }
    return engine;
}

double Utils::generateGaussianNoise(double sigma) {
    std::normal_distribution<double> distribution(0.0, sigma);
    auto& engine = getRandomEngine();
    return distribution(engine);
}

Eigen::Matrix3d Utils::eulerToRotationMatrix(double angle_z, double angle_y, double angle_x) {
    // 转换为弧度
    double z = angle_z * M_PI / 180.0;
    double y = angle_y * M_PI / 180.0;
    double x = angle_x * M_PI / 180.0;
    
    // Z-Y-X顺序的旋转矩阵
    Eigen::Matrix3d R_z, R_y, R_x;
    
    R_z << cos(z), -sin(z), 0,
           sin(z), cos(z), 0,
           0, 0, 1;
    
    R_y << cos(y), 0, sin(y),
           0, 1, 0,
           -sin(y), 0, cos(y);
    
    R_x << 1, 0, 0,
           0, cos(x), -sin(x),
           0, sin(x), cos(x);
    
    return R_z * R_y * R_x;
}

Eigen::Vector3d Utils::rotationMatrixToEuler(const Eigen::Matrix3d& R) {
    // 将旋转矩阵转换为欧拉角 (Z-Y-X顺序)
    double cos_pitch_abs = sqrt(R(0,0) * R(0,0) + R(1,0) * R(1,0));
    
    bool singular = cos_pitch_abs < 1e-6;
    
    double x, y, z;
    if (!singular) {
        x = atan2(R(2,1), R(2,2));
        y = atan2(-R(2,0), cos_pitch_abs);
        z = atan2(R(1,0), R(0,0));
    } else {
        x = atan2(-R(1,2), R(1,1));
        y = atan2(-R(2,0), cos_pitch_abs);
        z = 0;
    }
    
    // 转换为度
    return Eigen::Vector3d(z * 180 / M_PI, y * 180 / M_PI, x * 180 / M_PI);
}

std::vector<double> Utils::generateNoisyMeasurements(
    const std::vector<Eigen::Vector3d>& anchor_positions,
    const Eigen::Vector3d& true_position,
    double sigma_d) {
    
    std::vector<double> measurements;
    measurements.reserve(anchor_positions.size());
    
    if(anchor_positions.size()> SIZE_Threshold && Simulate_NLOS_flag ==1)
    {
        int count = 0;
        double noise;
        for (const auto& anchor : anchor_positions) {
            count ++;
            double true_distance = (true_position - anchor).norm();
            if(count>SIZE_Threshold)
            {
                noise = generateGaussianNoise(2.5*sigma_d);
            }
            else
            {
                noise = generateGaussianNoise(sigma_d);
            }
            measurements.push_back(true_distance + noise);
            // std::cout << "noise = " << noise << std::endl;
        }
       
    }
    else
    {
        // std::cout << "measurements" << std::endl;
        // int count = 0;//实验所需，记得注释掉
        for (const auto& anchor : anchor_positions) {
            double true_distance = (true_position - anchor).norm();
            double noise = generateGaussianNoise(sigma_d);
            measurements.push_back(true_distance + noise);
            // if(count<3)
            //     measurements.push_back(true_distance + noise);//正偏置模拟测量比实际大的情况
            // else
            //     measurements.push_back(true_distance + noise);
            // count ++;
            // std::cout << true_distance + noise << std::endl;
        }
    }

// 原本的代码
    // for (const auto& anchor : anchor_positions) {
    //     double true_distance = (true_position - anchor).norm();
    //     double noise = generateGaussianNoise(sigma_d);
    //     measurements.push_back(true_distance + noise);
    // }
    
    return measurements;
}

void Utils::saveResults(const std::vector<std::vector<double>>& errors, 
                       const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    // 写入表头
    file << "Run,TransError,RotError,NodeError" << std::endl;
    
    // 写入数据
    for (size_t i = 0; i < errors.size(); i++) {
        file << i + 1 << "," 
             << errors[i][0] << "," 
             << errors[i][1] << "," 
             << errors[i][2] << std::endl;
    }
    
    file.close();
    std::cout << "Results saved to: " << filename << std::endl;
}

void Utils::printStatistics(const std::vector<std::vector<double>>& errors) {
    if (errors.empty()) return;
    
    int num_runs = errors.size();
    int num_metrics = errors[0].size();
    
    std::vector<double> sums(num_metrics, 0.0);
    std::vector<double> maxs(num_metrics, -INFINITY);
    std::vector<double> mins(num_metrics, INFINITY);
    
    for (const auto& error : errors) {
        for (int i = 0; i < num_metrics; i++) {
            sums[i] += error[i];
            maxs[i] = std::max(maxs[i], error[i]);
            mins[i] = std::min(mins[i], error[i]);
        }
    }
    
    std::vector<double> means(num_metrics);
    for (int i = 0; i < num_metrics; i++) {
        means[i] = sums[i] / num_runs;
    }
    
    // 计算标准差
    std::vector<double> std_devs(num_metrics, 0.0);
    for (const auto& error : errors) {
        for (int i = 0; i < num_metrics; i++) {
            double diff = error[i] - means[i];
            std_devs[i] += diff * diff;
        }
    }
    for (int i = 0; i < num_metrics; i++) {
        std_devs[i] = sqrt(std_devs[i] / num_runs);
    }
    
    std::cout << "\n=== Error Statistics (over " << num_runs << " runs) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Metric\t\tMean\t\tStd Dev\t\tMin\t\tMax" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Translation\t" << means[0] << "\t\t" << std_devs[0] << "\t\t" 
              << mins[0] << "\t\t" << maxs[0] << std::endl;
    std::cout << "Rotation\t" << means[1] << "\t\t" << std_devs[1] << "\t\t" 
              << mins[1] << "\t\t" << maxs[1] << std::endl;
    std::cout << "Node Pos\t" << means[2] << "\t\t" << std_devs[2] << "\t\t" 
              << mins[2] << "\t\t" << maxs[2] << std::endl;
}


// 简单的CSV行解析（不考虑引号内的逗号）
std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }

    return fields;
}

std::map<int, std::map<int, std::vector<double>>> Extract_Data_from_file(std::string filename) 
{
    // std::string filename = "/home/yu/UWB/anchor_node/UWB_true_12_nodeframe2.csv";  // 请替换为实际文件路径
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return{};
    }

    // 读取标题行
    std::string headerLine;
    if (!std::getline(file, headerLine)) {
        std::cerr << "文件为空或无法读取标题" << std::endl;
        return{};
    }
    std::vector<std::string> headers = parseCSVLine(headerLine);

    // for (const auto& str : headers) {
    //     std::cout << str << std::endl;
    // }

    // 查找各列的索引
    int idColIndex = -1;
    std::map<int, int> idCols;   // X -> 列索引 (field.nodesX.id)
    std::map<int, int> disCols;  // X -> 列索引 (field.nodesX.dis)

    for (size_t i = 0; i < headers.size(); ++i) {
        const std::string& col = headers[i];
        if (col == "field.id") {
            idColIndex = i;
        }
        else if (col.substr(0, 11) == "field.nodes") {
            // 尝试提取X
            size_t dotPos = col.find('.', 11);
            if (dotPos != std::string::npos) {
                std::string numStr = col.substr(11, dotPos - 11);
                // 检查是否为数字
                bool isNumber = !numStr.empty() && std::all_of(numStr.begin(), numStr.end(), ::isdigit);
                if (isNumber) {
                    int X = std::stoi(numStr);
                    if (col.substr(dotPos) == ".id") {
                        idCols[X] = i;
                    }
                    else if (col.substr(dotPos) == ".dis") {
                        disCols[X] = i;
                    }
                }
            }
        }
    }

    if (idColIndex == -1) {
        std::cerr << "未找到 field.id 列" << std::endl;
        return{};
    }

    // 数据结构：distances[source][target] = 距离列表（按时间顺序）source为运动物体,target为基站
    std::map<int, std::map<int, std::vector<double>>> distances;

    std::string line;
    int lineNum = 1; // 用于调试
    while (std::getline(file, line)) {
        ++lineNum;
        std::vector<std::string> fields = parseCSVLine(line);

        // std::cout << lineNum <<std::endl;
        if(lineNum>800)
            break;

        // 读取field.id
        int source;
        try {
            source = std::stoi(fields[idColIndex]);
        } catch (...) {
            std::cerr << "警告：第 " << lineNum << " 行 field.id 格式错误，跳过" << std::endl;
            continue;
        }

        // 仅处理源节点5~8
        if (source < 5 || source > 8) continue;

        // 遍历nodes1.id ~ nodes6.id（X=1..6）
        for (int X = 0; X <= 8; X++) {
            // 检查是否存在对应的id列
            auto idIt = idCols.find(X);
            if (idIt == idCols.end()) continue;
            int idCol = idIt->second;


            int target;
            try {
                target = std::stoi(fields[idCol]);
            } catch (...) {
                // 可能为空或非数字，跳过
                continue;
            }

            // 目标节点必须为1~4
            // if ((target < 1 || target > 4)  ) continue; 
            // 目标节点必须为0~4,9  
            if ((target < 0 || target > 4) && target!=9 ) continue; 

            // 查找对应的dis列
            auto disIt = disCols.find(X);
            if (disIt == disCols.end()) {
                std::cerr << "警告：第 " << lineNum << " 行存在 nodes" << X << ".id 但无对应 .dis 列" << std::endl;
                continue;
            }
            int disCol = disIt->second;

            double distance;
            try {
                distance = std::stod(fields[disCol]);
            } catch (...) {
                std::cerr << "警告：第 " << lineNum << " 行距离值格式错误" << std::endl;
                continue;
            }

            // 存入数据
            distances[source][target].push_back(distance);
        }
    }

    file.close();



    // // 输出结果
    // std::cout << "按时间顺序排列的距离数据：" << std::endl;
    // for (int src = 5; src <= 8; ++src) {
    //     std::cout << "源节点 " << src << ":\n";
    //     for (int tgt = 1; tgt <= 4; ++tgt) {
    //         std::cout << "  到节点 " << tgt << " : ";
    //         if (distances.count(src) && distances[src].count(tgt)) {
    //             const auto& vec = distances[src][tgt];
    //             for (size_t i = 0; i < vec.size(); ++i) {
    //                 if (i > 0) std::cout << ", ";
    //                 std::cout << distances[src][tgt][i];//vec[i];
    //             }
    //         }
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << "元素个数：" << distances[5][1].size() << std::endl;
    // std::cout << "元素个数：" << distances[5][2].size() << std::endl;
    // std::cout << "元素个数：" << distances[5][3].size() << std::endl;
    // std::cout << "元素个数：" << distances[5][4].size() << std::endl;

    // std::cout << "元素个数：" << distances[6][1].size() << std::endl;
    // std::cout << "元素个数：" << distances[6][2].size() << std::endl;
    // std::cout << "元素个数：" << distances[6][3].size() << std::endl;
    // std::cout << "元素个数：" << distances[6][4].size() << std::endl;

    // std::cout << "元素个数：" << distances[7][1].size() << std::endl;
    // std::cout << "元素个数：" << distances[7][2].size() << std::endl;
    // std::cout << "元素个数：" << distances[7][3].size() << std::endl;
    // std::cout << "元素个数：" << distances[7][4].size() << std::endl;

    // std::cout << "元素个数：" << distances[8][1].size() << std::endl;
    // std::cout << "元素个数：" << distances[8][2].size() << std::endl;
    // std::cout << "元素个数：" << distances[8][3].size() << std::endl;
    // std::cout << "元素个数：" << distances[8][4].size() << std::endl;

    return distances;
}

double Calculate_positive_bias(int x,int y)
{
    double Bias;
    switch(x)
    {
        case 5:
            switch(y)
            {
                case 0:
                    Bias = 0.0;//0.114;
                    break;
                default:
                    Bias = 0.0;
                    break;
            }
            break;
        case 6:
            switch(y)
            {
                case 0:
                    Bias = 0.0;
                    break;
                default:
                    Bias = 0.0;
                    break;
            }
            break;
        case 7:
            switch(y)
            {
                case 0:
                    Bias = 0.0;
                    break;
                default:
                    Bias = 0.0;
                    break;
            }
            break;
        case 8:
            switch(y)
            {
                case 0:
                    Bias = 0.0;
                    break;
                default:
                    Bias = 0.0;
                    break;
            }
            break;
        default:
            Bias = 0.0;
            break;
    }
    return Bias;
}

//构造齐次坐标变换矩阵
Eigen::Matrix4d makeT(const Eigen::Matrix3d& R,
                      const Eigen::Vector3d& t)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

//求齐次坐标变换矩阵的逆
Eigen::Matrix4d inverseT(const Eigen::Matrix4d& T)
{
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);

    Eigen::Matrix4d T_inv = Eigen::Matrix4d::Identity();
    T_inv.block<3, 3>(0, 0) = R.transpose();
    T_inv.block<3, 1>(0, 3) = -R.transpose() * t;

    return T_inv;
}

/**
 * @brief 对点集做旋转+平移变换
 *        p_to = R_to_from * p_from + t_to_from
 */
std::vector<Eigen::Vector3d> transformPoints(
    const std::vector<Eigen::Vector3d> & points_from,
    const Eigen::Matrix3d& R_to_from,
    const Eigen::Vector3d& t_to_from)
{
    std::vector<Eigen::Vector3d> points_to;
    points_to.reserve(points_from.size());

    for(const auto& p : points_from)
    {
        points_to.push_back(R_to_from * p + t_to_from);
    }

    return points_to;
}

static Eigen::Matrix3d makeR_W_M()
{
    Eigen::Matrix3d R;
    R <<  0,  0, -1,
         -1,  0,  0,
          0,  1,  0;
    return R;
}

static Eigen::Matrix3d makeR_BM_B()
{
    Eigen::Matrix3d R;
    R <<  0, -1,  0,
          0,  0,  1,
         -1,  0,  0;
    return R;
}

static Eigen::Matrix3d makeR_AM_A()
{
    Eigen::Matrix3d R;
    R <<  0, -1,  0,
          0,  0,  1,
         -1,  0,  0;
    return R;
}

/**
 * @brief 使用动捕已知的 BM 坐标系定义，计算标签在 B 坐标系下的坐标
 *
 * 输入：
 *   p_M    : 单个标签在动捕坐标系 M 下的坐标
 *   R_M_BM     : BM坐标系到M坐标系的旋转，即 ^M_BM R
 *   p_M_BM0    : BM坐标系原点在M坐标系下的位置，即 ^M P_BM0
 *   R_BM_B     : B坐标系到BM坐标系的旋转，即 ^BM_B R
 *
 * 输出：
 *   p_B    : 单个标签在标签坐标系 B 下的坐标
 */
Eigen::Vector3d pointM_to_pointB_WithInitBM(
    const Eigen::Vector3d& p_M,
    const Eigen::Matrix3d& R_M_BM0,
    const Eigen::Vector3d& p_M_BM0,
    const Eigen::Matrix3d& R_BM_B)
{
    // M -> BM
    Eigen::Vector3d p_BM = R_M_BM0.transpose() * (p_M - p_M_BM0);

    // BM -> B
    Eigen::Vector3d p_B = R_BM_B.transpose() * p_BM;

    return p_B;
}

//M->A
Eigen::Vector3d pointM_to_pointA_WithInitAM(
    const Eigen::Vector3d& p_M,
    const Eigen::Matrix3d& R_M_AM0,
    const Eigen::Vector3d& p_M_AM0,
    const Eigen::Matrix3d& R_AM_A)
{
    // M -> AM
    Eigen::Vector3d p_AM = R_M_AM0.transpose() * (p_M - p_M_AM0);

    // AM -> A
    Eigen::Vector3d p_A = R_AM_A.transpose() * p_AM;

    return p_A;
}

/**
 * @brief 根据4个标签在W系下的位置，构建标签坐标系B，并求标签在B系下坐标
 *
 * B系定义：
 *   原点：4个标签中心
 *   x轴：标签6 -> 标签8
 *   y轴：标签7 -> 标签5
 *   z轴：右手系 x × y
 */
BFrameFromTagsResult buildBFrameFromTagsInW(
    const std::vector<Eigen::Vector3d>& node_W)
{
    if (node_W.size() != 4) {
        throw std::runtime_error("node_W must contain exactly 4 tags: tag5, tag6, tag7, tag8.");
    }

    // TAG_IDS = {5, 6, 7, 8}
    const Eigen::Vector3d& p5_W = node_W[0];
    const Eigen::Vector3d& p6_W = node_W[1];
    const Eigen::Vector3d& p7_W = node_W[2];
    const Eigen::Vector3d& p8_W = node_W[3];

    // 1. B系原点：4个标签中心
    Eigen::Vector3d p_W_B0 = Eigen::Vector3d::Zero();
    for (const auto& p : node_W) {
        p_W_B0 += p;
    }
    p_W_B0 /= 4.0;

    // 2. x轴：标签6 -> 标签8
    Eigen::Vector3d x_axis_W = (p8_W - p6_W).normalized();

    // 3. y轴临时方向：标签7 -> 标签5
    Eigen::Vector3d y_temp_W = (p5_W - p7_W).normalized();

    // 4. z轴：右手系 z = x × y
    Eigen::Vector3d z_axis_W = x_axis_W.cross(y_temp_W).normalized();

    // 5. 重新正交化 y轴，保证旋转矩阵正交
    Eigen::Vector3d y_axis_W = z_axis_W.cross(x_axis_W).normalized();

    // 6. 构造 ^W_B R
    Eigen::Matrix3d R_W_B;
    R_W_B.col(0) = x_axis_W;
    R_W_B.col(1) = y_axis_W;
    R_W_B.col(2) = z_axis_W;

    // 7. W -> B
    std::vector<Eigen::Vector3d> node_B;
    node_B.reserve(node_W.size());

    for (const auto& p_W : node_W) {
        Eigen::Vector3d p_B = R_W_B.transpose() * (p_W - p_W_B0);
        node_B.push_back(p_B);
    }

    return {R_W_B, p_W_B0, node_B};
}

CoordinatePrepResult prepareCoordinatesForPoseEstimation(
    bool use_init_mocap_data,

    const std::vector<Eigen::Vector3d>& node_M,
    const std::vector<Eigen::Vector3d>& anchor_M,
    const Eigen::Vector3d& tof_M,

    // 初始化时刻 BM 相对 M 的位姿
    const Eigen::Matrix3d& R_M_BM0,
    const Eigen::Vector3d& p_M_BM0,

    // 初始化时刻 AM 相对 M 的位姿
    const Eigen::Matrix3d& R_M_AM0,
    const Eigen::Vector3d& p_M_AM0)
{
    CoordinatePrepResult result;

    const Eigen::Matrix3d R_W_M  = makeR_W_M();
    const Eigen::Matrix3d R_BM_B = makeR_BM_B();
    const Eigen::Matrix3d R_AM_A = makeR_AM_A();

    // ============================================================
    // 1. 先处理基站：ANCHOR_M -> ANCHOR_A
    // ============================================================
    result.anchor_A.reserve(anchor_M.size());

    if (use_init_mocap_data) {
        // 有 AM 初始化数据：M -> AM -> A
        for (const auto& p_M : anchor_M) {
            Eigen::Vector3d p_A =
                pointM_to_pointA_WithInitAM(
                    p_M,
                    R_M_AM0,
                    p_M_AM0,
                    R_AM_A
                );

            result.anchor_A.push_back(p_A);
        }
    } else {
        // 没有 AM 初始化数据：假设 M 与 AM 重合
        // 即 R_M_AM0 = I, p_M_AM0 = 0
        for (const auto& p_M : anchor_M) {
            Eigen::Vector3d p_AM = p_M;
            Eigen::Vector3d p_A = R_AM_A.transpose() * p_AM;
            result.anchor_A.push_back(p_A);
        }
    }

    // ============================================================
    // 2. 再处理标签和TOF：NODE_M / TOF_M -> NODE_B / TOF_B
    // ============================================================
    if (use_init_mocap_data) {
        // 有 BM 初始化数据：M -> BM -> B
        result.node_B.reserve(node_M.size());

        for (const auto& p_M : node_M) {
            Eigen::Vector3d p_B =
                pointM_to_pointB_WithInitBM(
                    p_M,
                    R_M_BM0,
                    p_M_BM0,
                    R_BM_B
                );

            result.node_B.push_back(p_B);
        }

        result.tof_B =
            pointM_to_pointB_WithInitBM(
                tof_M,
                R_M_BM0,
                p_M_BM0,
                R_BM_B
            );

        result.R_W_B = Eigen::Matrix3d::Identity();
        result.p_W_B0 = Eigen::Vector3d::Zero();
    } else {
        // 没有 BM 初始化数据：
        // NODE_M -> NODE_W，然后利用4个标签构造B系
        std::vector<Eigen::Vector3d> node_W;
        node_W.reserve(node_M.size());

        for (const auto& p_M : node_M) {
            Eigen::Vector3d p_W = R_W_M * p_M;
            node_W.push_back(p_W);
        }

        Eigen::Vector3d tof_W = R_W_M * tof_M;

        BFrameFromTagsResult b_frame = buildBFrameFromTagsInW(node_W);

        result.R_W_B = b_frame.R_W_B;
        result.p_W_B0 = b_frame.p_W_B0;
        result.node_B = b_frame.node_B;

        // TOF_W -> TOF_B
        result.tof_B =
            result.R_W_B.transpose() * (tof_W - result.p_W_B0);
    }

    return result;
}

//用实时动捕真值评估相对位姿
Eigen::Matrix4d computeMocapTruth_T_A_B(
    const Eigen::Matrix3d& R_M_BM,
    const Eigen::Vector3d& p_M_BM,
    const Eigen::Matrix3d& R_M_AM,
    const Eigen::Vector3d& p_M_AM)
{
    const Eigen::Matrix3d R_BM_B = makeR_BM_B();
    const Eigen::Matrix3d R_AM_A = makeR_AM_A();

    // ^M_BM T
    Eigen::Matrix4d T_M_BM = makeT(R_M_BM, p_M_BM);

    // ^BM_B T
    Eigen::Matrix4d T_BM_B = makeT(R_BM_B, Eigen::Vector3d::Zero());

    // ^M_AM T
    Eigen::Matrix4d T_M_AM = makeT(R_M_AM, p_M_AM);

    // ^AM_A T
    Eigen::Matrix4d T_AM_A = makeT(R_AM_A, Eigen::Vector3d::Zero());

    // ^A_AM T = inverse(^AM_A T)
    Eigen::Matrix4d T_A_AM = inverseT(T_AM_A);

    // ^AM_M T = inverse(^M_AM T)
    Eigen::Matrix4d T_AM_M = inverseT(T_M_AM);

    // ^A_B T = ^A_AM T * ^AM_M T * ^M_BM T * ^BM_B T
    Eigen::Matrix4d T_A_B =
        T_A_AM * T_AM_M * T_M_BM * T_BM_B;

    return T_A_B;
}

/**
 * @brief 假设 M 与 AM 重合时，根据实时动捕求真值 ^A_B T
 *
 * ^A_B T = ^A_AM T * ^AM_M T * ^M_BM T * ^BM_B T
 * 其中 M = AM，所以 ^AM_M T = I
 */
Eigen::Matrix4d computeMocapTruth_T_A_B_AssumeMEqualsAM(
    const Eigen::Matrix3d& R_M_BM,
    const Eigen::Vector3d& p_M_BM)
{
    Eigen::Matrix3d R_AM_A = makeR_AM_A();
    Eigen::Matrix3d R_BM_B = makeR_BM_B();

    Eigen::Matrix4d T_M_BM = makeT(R_M_BM, p_M_BM);
    Eigen::Matrix4d T_BM_B = makeT(R_BM_B, Eigen::Vector3d::Zero());

    Eigen::Matrix4d T_AM_A = makeT(R_AM_A, Eigen::Vector3d::Zero());
    Eigen::Matrix4d T_A_AM = inverseT(T_AM_A);

    // M = AM，所以 ^AM_M T = I
    Eigen::Matrix4d T_A_B = T_A_AM * T_M_BM * T_BM_B;

    return T_A_B;
}