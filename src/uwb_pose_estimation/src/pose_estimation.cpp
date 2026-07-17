#include "pose_estimation.h"
#include "utils.h"
#include <iostream>
#include <cmath>

std::pair<Eigen::Matrix3d, Eigen::Vector3d> PoseEstimator::estimatePoseSVD(
    const std::vector<Eigen::Vector3d>& v_j,
    const std::vector<Eigen::Vector3d>& q_j) {
    
    int N_B = v_j.size();
    
    // 计算均值
    Eigen::Vector3d v_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d q_mean = Eigen::Vector3d::Zero();
    
    for(int j = 0; j < N_B; j++) {
        v_mean += v_j[j];
        q_mean += q_j[j];
    }
    v_mean /= N_B;
    q_mean /= N_B;
    
    // 去中心化
    std::vector<Eigen::Vector3d> v_bar(N_B);
    std::vector<Eigen::Vector3d> q_bar(N_B);
    
    for(int j = 0; j < N_B; j++) {
        v_bar[j] = v_j[j] - v_mean;
        q_bar[j] = q_j[j] - q_mean;
    }
    
    // 计算矩阵H
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for(int j = 0; j < N_B; j++) {
        H += v_bar[j] * q_bar[j].transpose();
    }
    
    // SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // 计算旋转矩阵
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if((U * V.transpose()).determinant() < 0) {
        D(2, 2) = -1.0;
    }
    
    Eigen::Matrix3d R = U * D * V.transpose();
    Eigen::Vector3d t = v_mean - R * q_mean;
    
    return std::make_pair(R, t);
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> PoseEstimator::estimatePoseCeres(
    const std::vector<Eigen::Vector3d>& v_j,
    const std::vector<Eigen::Vector3d>& q_j,
    const Eigen::Matrix3d& initial_rotation,
    const Eigen::Vector3d& initial_translation,
    const ExperimentConfig& config) {
    
    // 将旋转矩阵转换为四元数
    Eigen::Quaterniond q_init(initial_rotation);
    double rotation[4] = {q_init.w(), q_init.x(), q_init.y(), q_init.z()};
    double translation[3] = {initial_translation[0], 
                            initial_translation[1], 
                            initial_translation[2]};
    
    // 构建优化问题
    ceres::Problem problem;
    
    // 添加四元数单位长度约束
    problem.AddParameterBlock(rotation, 4, new ceres::EigenQuaternionParameterization());
    
    // 添加残差块
    for (size_t j = 0; j < v_j.size(); j++) {
        ceres::CostFunction* cost_function = 
            new ceres::AutoDiffCostFunction<PoseCostFunction, 3, 4, 3>(
                new PoseCostFunction(v_j[j], q_j[j]));
            // new ceres::NumericDiffCostFunction<PoseCostFunction, ceres::CENTRAL, 3, 4, 3>(
            //     new PoseCostFunction(v_j[j], q_j[j]));
        
        problem.AddResidualBlock(cost_function, nullptr, rotation, translation);
    }
    
    // 设置优化选项
    ceres::Solver::Options options;
    options.max_num_iterations = config.pose_estimation.max_iterations;
    options.minimizer_progress_to_stdout = false;
    
    // 设置线性求解器
    if (config.pose_estimation.linear_solver_type == "DENSE_NORMAL_CHOLESKY") {
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    } else if (config.pose_estimation.linear_solver_type == "DENSE_QR") {
        options.linear_solver_type = ceres::DENSE_QR;
    }
    
    // 求解问题
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 将四元数转换回旋转矩阵
    Eigen::Quaterniond est_q(rotation[0], rotation[1], rotation[2], rotation[3]);
    Eigen::Matrix3d est_rotation = est_q.toRotationMatrix();
    Eigen::Vector3d est_translation(translation[0], translation[1], translation[2]);
    
    return std::make_pair(est_rotation, est_translation);
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> PoseEstimator::estimatePoseCeresKnownRollPitch(
    const std::vector<Eigen::Vector3d>& v_j,
    const std::vector<Eigen::Vector3d>& q_j,
    const Eigen::Matrix3d& initial_rotation,
    const Eigen::Vector3d& initial_translation,
    double known_roll_rad,
    double known_pitch_rad,
    const ExperimentConfig& config) {

    double yaw[1] = {std::atan2(initial_rotation(1, 0), initial_rotation(0, 0))};
    double translation[3] = {initial_translation.x(), initial_translation.y(),
                             initial_translation.z()};

    ceres::Problem problem;
    for (size_t j = 0; j < v_j.size(); ++j) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<PoseKnownRollPitchCostFunction, 3, 1, 3>(
                new PoseKnownRollPitchCostFunction(v_j[j], q_j[j],
                                                   known_roll_rad, known_pitch_rad));
        problem.AddResidualBlock(cost_function, nullptr, yaw, translation);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = config.pose_estimation.max_iterations;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = config.pose_estimation.linear_solver_type == "DENSE_QR"
                                   ? ceres::DENSE_QR
                                   : ceres::DENSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    const Eigen::Matrix3d rotation =
        (Eigen::AngleAxisd(yaw[0], Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(known_pitch_rad, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(known_roll_rad, Eigen::Vector3d::UnitX())).toRotationMatrix();
    return std::make_pair(rotation,
                          Eigen::Vector3d(translation[0], translation[1], translation[2]));
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> PoseEstimator::estimatePose(
    const std::vector<Eigen::Vector3d>& v_j,
    const std::vector<Eigen::Vector3d>& q_j,
    const ExperimentConfig& config) {
    
    // 先用SVD得到初始估计
    auto [initial_rotation, initial_translation] = estimatePoseSVD(v_j, q_j);

    // initial_rotation = Utils::eulerToRotationMatrix(45,3,4);
    // initial_translation << 1,3,3;

    // 再用Ceres进行优化
    return estimatePoseCeres(v_j, q_j, initial_rotation, initial_translation, config);

    // //再用迭代最小二乘
    // Eigen::Vector3d initial_test_trans;
    // initial_test_trans << 4,5,100;
    // return refinePose(v_j,q_j,initial_rotation,initial_test_trans);
}

PoseMetrics PoseEstimator::calculateMetrics(
    const Eigen::Matrix3d& estimated_rotation,
    const Eigen::Vector3d& estimated_translation,
    const std::vector<Eigen::Vector3d>& estimated_nodes,
    const Eigen::Matrix3d& true_rotation,
    const Eigen::Vector3d& true_translation,
    const std::vector<Eigen::Vector3d>& true_nodes)
{
    PoseMetrics metrics;
    // 计算平移误差
    metrics.translation_error = (true_translation - estimated_translation).norm();
    
    // // 计算旋转误差（四元数）
    // Eigen::Quaterniond true_q(true_rotation);
    // Eigen::Quaterniond est_q(estimated_rotation);
    // double dot_product = std::abs(true_q.dot(est_q));
    // metrics.rotation_error = 2 * std::acos(std::min(1.0, dot_product)) * 180 / M_PI;

    //计算旋转误差（轴角）
    Eigen::Matrix3d rotation_error_mat = estimated_rotation * true_rotation.transpose();
    Eigen::AngleAxisd rotation_error_aa(rotation_error_mat);
    double rotation_error = rotation_error_aa.angle() * 180.0 / M_PI;
    metrics.rotation_error = rotation_error;
    
    metrics.node_position_errors.reserve(estimated_nodes.size());
    // 计算节点位置平均误差
    double total_node_error = 0.0;
    for (size_t i = 0; i < estimated_nodes.size(); i++) {
        double error = (true_nodes[i] - estimated_nodes[i]).norm();
        metrics.node_position_errors.push_back(error);
        total_node_error += error;
    }
    metrics.node_position_error_avg = total_node_error / estimated_nodes.size();
    
    return metrics;
}


// 计算旋转矩阵的叉积矩阵
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d mat;
    mat <<  0, -v(2),  v(1),
           v(2),  0, -v(0),
          -v(1),  v(0),  0;
    return mat;
}

// 旋转矩阵的小角增量（轴角表示）
Eigen::Matrix3d angleAxisToRotationMatrix(const Eigen::Vector3d& angleAxis) {
    double theta = angleAxis.norm();
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity();  // 对于非常小的角度，近似为单位矩阵
    }
    Eigen::Matrix3d K = skewSymmetric(angleAxis / theta);
    return Eigen::Matrix3d::Identity() + sin(theta) * K + (1 - cos(theta)) * K * K;
}

// 姿态估计函数
std::pair<Eigen::Matrix3d, Eigen::Vector3d> refinePose(const std::vector<Eigen::Vector3d>& v, const std::vector<Eigen::Vector3d>& q,
     Eigen::Matrix3d& R, Eigen::Vector3d& r) {
    int numPoints = v.size();
    Eigen::Vector3d deltaR;
    Eigen::Vector3d deltaT;
    Eigen::Matrix3d J;
    Eigen::Vector3d e_j;
    
    double lambda = 1e-5;  // 阻尼因子
    int maxIter = 100;
    double tol = 1e-15;

    for (int iter = 0; iter < maxIter; ++iter) {
        // 计算雅可比和残差
        Eigen::MatrixXd J_all(3 * numPoints, 6);  // 6维参数：3维旋转增量 + 3维平移增量
        Eigen::VectorXd e_all(3 * numPoints);

        for (int j = 0; j < numPoints; ++j) {
            e_j = v[j] - (R * q[j] + r);  // 残差
            Eigen::Matrix3d J_rot = -skewSymmetric(R * q[j]);  // 旋转部分的雅可比
            J_all.block<3, 3>(3 * j, 0) = J_rot;
            J_all.block<3, 3>(3 * j, 3) = -Eigen::Matrix3d::Identity();
            e_all.segment<3>(3 * j) = e_j;
        }

        // 求解正规方程
        Eigen::MatrixXd JTJ = J_all.transpose() * J_all + lambda * Eigen::MatrixXd::Identity(6, 6);
        Eigen::VectorXd JTe = -J_all.transpose() * e_all;

        Eigen::VectorXd delta = JTJ.ldlt().solve(JTe);  // 解正规方程

        // 更新旋转矩阵和位移向量
        deltaR = delta.head<3>();
        deltaT = delta.tail<3>();

        // 旋转矩阵更新：angle-axis 表示旋转增量
        R = angleAxisToRotationMatrix(deltaR) * R;
        r += deltaT;

        // 判断是否收敛
        if (delta.norm() < tol) {
            // std::cout << "Converged after " << iter << " iterations." << "delta =" << delta.norm() << std::endl;
            break;
        }
    }
    return std::make_pair(R,r);
    // cout << "estimated_trans" << "x" << r[0] << "  y" << r[1] << "  z" << r[2]<<endl;
}
