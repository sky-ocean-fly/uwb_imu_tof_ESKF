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
    double sy = sqrt(R(0,0) * R(0,0) + R(1,0) * R(1,0));
    
    bool singular = sy < 1e-6;
    
    double x, y, z;
    if (!singular) {
        x = atan2(R(2,1), R(2,2));
        y = atan2(-R(2,0), sy);
        z = atan2(R(1,0), R(0,0));
    } else {
        x = atan2(-R(1,2), R(1,1));
        y = atan2(-R(2,0), sy);
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

//之前斜距更新有问题，帧率是IMU_raw、ToF之和
std::vector<double> extractFieldYEvery10Rows(const std::string& filename) 
{
    std::vector<double> result;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return result;
    }
    
    // 读取标题行
    std::string headerLine;
    if (!std::getline(file, headerLine)) {
        std::cerr << "文件为空或无法读取标题" << std::endl;
        return result;
    }
    
    // 解析标题行，找到 field.y 的列索引
    std::stringstream ss(headerLine);
    std::string field;
    int colIndex = -1;
    int currentCol = 0;
    
    while (std::getline(ss, field, ',')) {
        if (field == "field.y") {
            colIndex = currentCol;
            break;
        }
        currentCol++;
    }
    
    if (colIndex == -1) {
        std::cerr << "未找到 field.y 列" << std::endl;
        return result;
    }
    
    // 读取数据行，每隔10行提取一次
    std::string line;
    int lineCount = 0;  // 从0开始计数，第0行、第10行、第20行...被提取
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        if (lineCount % 10 == 0) {
            std::stringstream lineSs(line);
            std::string value;
            int col = 0;
            
            while (std::getline(lineSs, value, ',')) {
                if (col == colIndex) {
                    try {
                        double yValue = std::stod(value);
                        result.push_back(yValue);
                    } catch (const std::exception& e) {
                        std::cerr << "第 " << (lineCount + 2) << " 行 field.y 值格式错误: " << value << std::endl;
                    }
                    break;
                }
                col++;
            }
        }
        lineCount++;
    }
    
    file.close();
    return result;
}
