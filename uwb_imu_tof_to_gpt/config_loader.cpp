#include "config_loader.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>

ExperimentConfig ConfigLoader::loadConfig(const std::string& config_file) {
    ExperimentConfig config;
    
    try {
        YAML::Node yaml_config = YAML::LoadFile(config_file);
        
        // 实验配置
        config.num_runs = yaml_config["experiment"]["num_runs"].as<int>();
        config.output_dir = yaml_config["experiment"]["output_dir"].as<std::string>();
        
        // 节点配置
        // config.num_nodes = yaml_config["nodes"]["num_nodes"].as<int>();
        config.node_local_positions.clear();
        for (const auto& pos : yaml_config["nodes"]["local_positions"]) {
            config.node_local_positions.push_back(
                Eigen::Vector3d(pos[0].as<double>(), 
                               pos[1].as<double>(), 
                               pos[2].as<double>()));
        }
        
        // 锚点配置
        // config.num_anchors = yaml_config["anchors"]["num_anchors"].as<int>();
        config.anchor_positions.clear();
        for (const auto& pos : yaml_config["anchors"]["positions"]) {
            config.anchor_positions.push_back(
                Eigen::Vector3d(pos[0].as<double>(), 
                               pos[1].as<double>(), 
                               pos[2].as<double>()));
        }
        
        // 噪声配置
        config.sigma_d = yaml_config["noise"]["sigma_d"].as<double>();
        config.random_seed = yaml_config["noise"]["seed"].as<int>();
        
        // 真实位姿
        auto true_trans = yaml_config["ground_truth"]["translation"];
        config.true_translation = Eigen::Vector3d(
            true_trans[0].as<double>(),
            true_trans[1].as<double>(),
            true_trans[2].as<double>());
        
        auto rot_angles = yaml_config["ground_truth"]["rotation_angles"];
        config.true_rotation_angles = Eigen::Vector3d(
            rot_angles["z"].as<double>(),
            rot_angles["y"].as<double>(),
            rot_angles["x"].as<double>());
        
        // 优化器配置
        config.node_position.max_iterations = 
            yaml_config["optimizer"]["node_position"]["max_iterations"].as<int>();
        config.node_position.loss_function = 
            yaml_config["optimizer"]["node_position"]["loss_function"].as<double>();
        config.node_position.linear_solver_type = 
            yaml_config["optimizer"]["node_position"]["linear_solver_type"].as<std::string>();
        
        config.pose_estimation.max_iterations = 
            yaml_config["optimizer"]["pose_estimation"]["max_iterations"].as<int>();
        config.pose_estimation.linear_solver_type = 
            yaml_config["optimizer"]["pose_estimation"]["linear_solver_type"].as<std::string>();
            
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading config file: " << e.what() << std::endl;
        exit(1);
    }
    
    return config;
}

void ConfigLoader::printConfig(const ExperimentConfig& config) {
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Experiment:" << std::endl;
    std::cout << "  Num runs: " << config.num_runs << std::endl;
    std::cout << "  Output dir: " << config.output_dir << std::endl;
    
    std::cout << "Nodes:" << std::endl;
    // std::cout << "  Num nodes: " << config.num_nodes << std::endl;
    
    std::cout << "Anchors:" << std::endl;
    // std::cout << "  Num anchors: " << config.num_anchors << std::endl;
    
    std::cout << "Noise:" << std::endl;
    std::cout << "  Sigma_d: " << config.sigma_d << std::endl;
    std::cout << "  Random seed: " << config.random_seed << std::endl;
    
    std::cout << "Ground truth:" << std::endl;
    std::cout << "  Translation: [" << config.true_translation.transpose() << "]" << std::endl;
    std::cout << "  Rotation angles (deg): [" << config.true_rotation_angles.transpose() << "]" << std::endl;
    
    std::cout << "====================" << std::endl;
}