#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#pragma once

#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "config_loader.h"

using namespace std;

// 位姿估计代价函数
class PoseCostFunction {
private:
    Eigen::Vector3d _v_j;  // 第一步估计的节点位置（frame A中）
    Eigen::Vector3d _q_j;  // 节点在frame B中的已知位置
    
public:
    PoseCostFunction(const Eigen::Vector3d& v_j, const Eigen::Vector3d& q_j) 
        : _v_j(v_j), _q_j(q_j) {}
    
    template <typename T>
    bool operator()(const T* const rotation, const T* const translation, T* residual) const {
        // rotation: 四元数 [w, x, y, z]
        // translation: 平移向量 [tx, ty, tz]
        
        // 使用四元数旋转 q_j
        T q_w = rotation[0];
        T q_x = rotation[1];
        T q_y = rotation[2];
        T q_z = rotation[3];
        
        T v_x = T(_q_j[0]);
        T v_y = T(_q_j[1]);
        T v_z = T(_q_j[2]);
        
        // 四元数旋转公式: v' = q * v * q_conjugate
        T t0 = T(2.0) * (q_y * v_z - q_z * v_y);
        T t1 = T(2.0) * (q_z * v_x - q_x * v_z);
        T t2 = T(2.0) * (q_x * v_y - q_y * v_x);
        
        T rotated_x = v_x + q_w * t0 + q_y * t2 - q_z * t1;
        T rotated_y = v_y + q_w * t1 + q_z * t0 - q_x * t2;
        T rotated_z = v_z + q_w * t2 + q_x * t1 - q_y * t0;
        
        // 加上平移
        T predicted_x = rotated_x + translation[0];
        T predicted_y = rotated_y + translation[1];
        T predicted_z = rotated_z + translation[2];
        
        // 计算残差
        residual[0] = T(_v_j[0]) - predicted_x;
        residual[1] = T(_v_j[1]) - predicted_y;
        residual[2] = T(_v_j[2]) - predicted_z;
        
        return true;
    }
};

struct PoseMetrics
{
    double translation_error;
    double rotation_error;
    double node_position_error_avg;
    vector<double> node_position_errors;
};

// 位姿估计器
class PoseEstimator {
public:
    // SVD估计（闭式解）
    static std::pair<Eigen::Matrix3d, Eigen::Vector3d> estimatePoseSVD(
        const std::vector<Eigen::Vector3d>& v_j,
        const std::vector<Eigen::Vector3d>& q_j);
    
    // Ceres优化估计
    static std::pair<Eigen::Matrix3d, Eigen::Vector3d> estimatePoseCeres(
        const std::vector<Eigen::Vector3d>& v_j,
        const std::vector<Eigen::Vector3d>& q_j,
        const Eigen::Matrix3d& initial_rotation,
        const Eigen::Vector3d& initial_translation,
        const ExperimentConfig& config);
    
    // 综合估计
    static std::pair<Eigen::Matrix3d, Eigen::Vector3d> estimatePose(
        const std::vector<Eigen::Vector3d>& v_j,
        const std::vector<Eigen::Vector3d>& q_j,
        const ExperimentConfig& config);
    
    static PoseMetrics calculateMetrics(
        const Eigen::Matrix3d& estimated_rotation,
        const Eigen::Vector3d& estimated_translation,
        const std::vector<Eigen::Vector3d>& estimated_nodes,
        const Eigen::Matrix3d& true_rotation,
        const Eigen::Vector3d& true_translation,
        const std::vector<Eigen::Vector3d>& true_nodes);
};

std::pair<Eigen::Matrix3d, Eigen::Vector3d> refinePose(const std::vector<Eigen::Vector3d>& v, const std::vector<Eigen::Vector3d>& q,
     Eigen::Matrix3d& R, Eigen::Vector3d& r);

#endif