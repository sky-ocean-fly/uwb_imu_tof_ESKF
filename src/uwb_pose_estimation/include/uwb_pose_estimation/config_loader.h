#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

struct ExperimentConfig {
    // 实验配置
    int num_runs;
    std::string output_dir;
    
    // 节点配置
    // int num_nodes;
    std::vector<Eigen::Vector3d> node_local_positions;
    
    // 锚点配置
    // int num_anchors;
    std::vector<Eigen::Vector3d> anchor_positions;
    
    // 噪声配置
    double sigma_d;
    int random_seed;
    
    // 真实位姿
    Eigen::Vector3d true_translation;
    Eigen::Vector3d true_rotation_angles; // 欧拉角 (z, y, x)
    
    // 优化器配置
    struct {
        int max_iterations;
        double loss_function;
        std::string linear_solver_type;
    } node_position;
    
    struct {
        int max_iterations;
        std::string linear_solver_type;
    } pose_estimation;
};

class ConfigLoader {
public:
    static ExperimentConfig loadConfig(const std::string& config_file);
    static void printConfig(const ExperimentConfig& config);
};

#endif