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

std::vector<std::string> parseCSVLine(const std::string& line);

std::map<int, std::map<int, std::vector<double>>> Extract_Data_from_file(std::string filename);

double Calculate_positive_bias(int x,int y);

std::vector<double> extractFieldYEvery10Rows(const std::string& filename);

#endif
