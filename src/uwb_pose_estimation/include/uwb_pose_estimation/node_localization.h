#ifndef NODE_LOCALIZATION_H
#define NODE_LOCALIZATION_H

#pragma once

#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "config_loader.h"

// 距离测量代价函数
class RangeCostFunction {
private:
    Eigen::Vector3d _anchor_pos;  // 锚点位置 p_i
    double _measured_distance;    // 测量距离 d_ij
    double _weight;               // 权重
    
public:
    RangeCostFunction(const Eigen::Vector3d& anchor_pos, double measured_distance, double weight) 
        : _anchor_pos(anchor_pos), _measured_distance(measured_distance), _weight(weight) {}
    
    template <typename T>
    bool operator()(const T* const node_pos, T* residual) const {
        T dx = T(node_pos[0]) - T(_anchor_pos[0]);
        T dy = T(node_pos[1]) - T(_anchor_pos[1]);
        T dz = T(node_pos[2]) - T(_anchor_pos[2]);
        
        T estimated_distance = ceres::sqrt(dx * dx + dy * dy + dz * dz);
        
        residual[0] = T(_weight) * (T(_measured_distance) - estimated_distance);
        return true;
    }
};

// 节点定位器
class NodeLocalizer {
public:
    // 最小二乘估计（闭式解）
    static Eigen::Vector3d estimateNodePositionLeastSquares(
        const std::vector<Eigen::Vector3d>& anchor_positions,
        const std::vector<double>& measured_distances);
    
    // Ceres优化估计
    static Eigen::Vector3d estimateNodePositionCeres(
        const std::vector<Eigen::Vector3d>& anchor_positions,
        const std::vector<double>& measured_distances,
        const Eigen::Vector3d& initial_guess,
        const ExperimentConfig& config);

    //迭代最小二乘法
    static Eigen::Vector3d estimateNodePositionIterative(
    const std::vector<Eigen::Vector3d>& p_i,
    const std::vector<double>& d_ij,
    const Eigen::Vector3d& v_init,
    int max_iter = 50,
    double tol = 1e-6);
    
    // 综合估计（先最小二乘，再Ceres优化）
    static Eigen::Vector3d estimateNodePosition(
        const std::vector<Eigen::Vector3d>& anchor_positions,
        const std::vector<double>& measured_distances,
        const ExperimentConfig& config,
        Eigen::Vector3d true_node_positions);

    static void weightestimate_0(int sim_id,std::vector<double> measured_distances);
    static void weightestimate_15(int sim_id,std::vector<double> measured_distances);
    static void weightestimate_4(int sim_id,std::vector<double> measured_distances);
};

#endif