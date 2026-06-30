#ifndef UTILS_H
#define UTILS_H

#pragma once

#include <random>
#include <vector>
#include <Eigen/Dense>
#include <deque>
#include <chrono>
#include <map>
#include "config_loader.h"

class Utils {
public:
    // 随机数生成器
    static std::default_random_engine& getRandomEngine(int seed = -1);
    
    // 生成高斯噪声
    static double generateGaussianNoise(double sigma);
    
    // 欧拉角转旋转矩阵 (Z-Y-X顺序)
    static Eigen::Matrix3d eulerToRotationMatrix(double angle_z, double angle_y, double angle_x);
    
    // 旋转矩阵转欧拉角 (Z-Y-X顺序)
    static Eigen::Vector3d rotationMatrixToEuler(const Eigen::Matrix3d& R);
    
    // 生成带噪声的测量距离
    static std::vector<double> generateNoisyMeasurements(
        const std::vector<Eigen::Vector3d>& anchor_positions,
        const Eigen::Vector3d& true_position,
        double sigma_d);
    
    // 保存结果到文件
    static void saveResults(
        const std::vector<std::vector<double>>& errors,
        const std::string& filename);
    
    // 打印统计信息
    static void printStatistics(const std::vector<std::vector<double>>& errors);
};

class FixedDeque {
private:
    std::deque<double> buffer;
    static const size_t MAX_SIZE = 12;

public:
    // 添加新元素
    void add(double value) {
        buffer.push_back(value);
        
        // 如果超过最大长度，删除第一个元素
        if (buffer.size() > MAX_SIZE) {
            buffer.pop_front();
        }
    }

    // 获取数组
    std::deque<double> getArray() const {
        return buffer;
    }

    // // 打印数组
    // void print() const {
    //     for (double val : buffer) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << std::endl;
    // }

    size_t size() const {
        return buffer.size();
    }

    bool isFull() const {
        return buffer.size() >= MAX_SIZE;
    }

    // 访问特定位置的元素
    double at(size_t index) const {
        return buffer.at(index);
    }
    
};


class FixedDeque_time {
private:
    std::deque<std::chrono::system_clock::time_point> buffer;
    static const size_t MAX_SIZE = 12;

public:
    // 添加新元素
    void add(std::chrono::system_clock::time_point value) {
        buffer.push_back(value);
        
        // 如果超过最大长度，删除第一个元素
        if (buffer.size() > MAX_SIZE) {
            buffer.pop_front();
        }
    }

    // 获取数组
    std::deque<std::chrono::system_clock::time_point> getArray() const {
        return buffer;
    }

    size_t size() const {
        return buffer.size();
    }

    bool isFull() const {
        return buffer.size() >= MAX_SIZE;
    }

    // 访问特定位置的元素
    std::chrono::system_clock::time_point at(size_t index) const {
        return buffer.at(index);
    }
    
};

struct CoordinatePrepResult {
    std::vector<Eigen::Vector3d> node_B;    // 标签在B坐标系下的位置
    std::vector<Eigen::Vector3d> anchor_A;  // 基站在A坐标系下的位置
    Eigen::Vector3d tof_B;                  // ToF在B坐标系下的位置

    Eigen::Matrix3d R_W_B;                  // 如果无动捕初始化，用标签几何构造出的 ^W_B R
    Eigen::Vector3d p_W_B0;                 // 如果无动捕初始化，用标签几何构造出的 ^W P_B0
};

//情况二：没有初始化 BM 动捕数据，用 4 个标签构造 B 坐标系
struct BFrameFromTagsResult {
    Eigen::Matrix3d R_W_B;
    Eigen::Vector3d p_W_B0;
    std::vector<Eigen::Vector3d> node_B;
};

std::vector<std::string> parseCSVLine(const std::string& line);

std::map<int, std::map<int, std::vector<double>>> Extract_Data_from_file(std::string filename);

double Calculate_positive_bias(int x,int y);

Eigen::Matrix4d makeT(const Eigen::Matrix3d& R,
                      const Eigen::Vector3d& t);
                    
Eigen::Matrix4d inverseT(const Eigen::Matrix4d& T);

std::vector<Eigen::Vector3d> transformPoints(
    const std::vector<Eigen::Vector3d>& points_from,
    const Eigen::Matrix3d& R_to_from,
    const Eigen::Vector3d& t_to_from = Eigen::Vector3d::Zero());

static Eigen::Matrix3d makeR_W_M();

static Eigen::Matrix3d makeR_BM_B();

static Eigen::Matrix3d makeR_AM_A();

Eigen::Vector3d pointM_to_pointB_WithInitBM(
    const Eigen::Vector3d& p_M,
    const Eigen::Matrix3d& R_M_BM0,
    const Eigen::Vector3d& p_M_BM0,
    const Eigen::Matrix3d& R_BM_B);

Eigen::Vector3d pointM_to_pointA_WithInitAM(
    const Eigen::Vector3d& p_M,
    const Eigen::Matrix3d& R_M_AM0,
    const Eigen::Vector3d& p_M_AM0,
    const Eigen::Matrix3d& R_AM_A);

BFrameFromTagsResult buildBFrameFromTagsInW(
    const std::vector<Eigen::Vector3d>& node_W);

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
    const Eigen::Vector3d& p_M_AM0);

Eigen::Matrix4d computeMocapTruth_T_A_B_AssumeMEqualsAM(
    const Eigen::Matrix3d& R_M_BM,
    const Eigen::Vector3d& p_M_BM);

#endif
