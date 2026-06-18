#include "node_localization.h"
#include "global_declare.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <fstream>

//节点定位初值
Eigen::Vector3d NodeLocalizer::estimateNodePositionLeastSquares(
    const std::vector<Eigen::Vector3d>& anchor_positions,
    const std::vector<double>& measured_distances) {
    
    int N_A = anchor_positions.size();
    
    // 构建 Ax = b
    Eigen::MatrixXd A(N_A, 4);
    Eigen::VectorXd b(N_A);
    
    for(int i = 0; i < N_A; i++) {
        const Eigen::Vector3d& p_i = anchor_positions[i];
        double d_ij = measured_distances[i];
        
        A(i, 0) = 1.0;
        A(i, 1) = -2.0 * p_i[0];
        A(i, 2) = -2.0 * p_i[1];
        A(i, 3) = -2.0 * p_i[2];
        
        b(i) = d_ij * d_ij - p_i.dot(p_i);
    }
    
    // 最小二乘解
    Eigen::VectorXd X = A.colPivHouseholderQr().solve(b);

    // std::cout << "X:" << X << std::endl;
    
    // 提取位置
    Eigen::Vector3d v_j;
    v_j[0] = X[1];
    v_j[1] = X[2];
    
    // 计算 z 坐标
    double x_sq = X[1] * X[1];
    double y_sq = X[2] * X[2];
    double z_sq = X[0] - x_sq - y_sq;
    
    if(X[3]<0)
        v_j[2] = -sqrt(fabs(z_sq));
    else
        v_j[2] = sqrt(fabs(z_sq));

    // std::cout << "初始解v_j:" << v_j << std::endl;
    
    return v_j;
}

//优化求解
Eigen::Vector3d NodeLocalizer::estimateNodePositionCeres(
    const std::vector<Eigen::Vector3d>& anchor_positions,
    const std::vector<double>& measured_distances,
    const Eigen::Vector3d& initial_guess,
    const ExperimentConfig& config) {
    
    double node_pos[3] = {initial_guess[0], initial_guess[1], initial_guess[2]};
    
    //加权最小二乘
    double weight;
    // 构建优化问题
    ceres::Problem problem;
    for (size_t i = 0; i < anchor_positions.size(); i++) {
        if(Weighted_Least_Squares_flag == 1)
        {
            if(i<8)
                weight = 1;
            else
                weight = Weighted_Least_Squares_weight;
        }
        else
            weight = 1;

        // std::cout << anchor_positions[i] << std::endl;
        // std::cout << measured_distances[i] << std::endl;
        ceres::CostFunction* cost_function = 
            new ceres::AutoDiffCostFunction<RangeCostFunction, 1, 3>(
                new RangeCostFunction(anchor_positions[i], measured_distances[i], weight));
            // new ceres::NumericDiffCostFunction<RangeCostFunction, ceres::CENTRAL, 1, 3>(
            //     new RangeCostFunction(anchor_positions[i], measured_distances[i], weight));

        problem.AddResidualBlock(cost_function, 
            new ceres::HuberLoss(config.node_position.loss_function), 
            node_pos);
    }
    
    // 设置优化选项
    ceres::Solver::Options options;
    options.max_num_iterations = config.node_position.max_iterations;
    options.minimizer_progress_to_stdout = false;
    
    // 设置线性求解器
    if (config.node_position.linear_solver_type == "DENSE_NORMAL_CHOLESKY") {
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    } else if (config.node_position.linear_solver_type == "DENSE_QR") {
        options.linear_solver_type = ceres::DENSE_QR;
    } else if (config.node_position.linear_solver_type == "SPARSE_NORMAL_CHOLESKY") {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    
    // 求解问题
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector3d estimated_Node_position(node_pos[0],node_pos[1],node_pos[2]);

    // //看残差大小,确定loss_function大小 结论：和噪声标准差差不多
    // for(int i = 0;i < anchor_positions.size();i++)
    // {
    //     std::cout << "residual= " << measured_distances[i] - (anchor_positions[i] - estimated_Node_position).norm()<<std::endl;//initial_guess
    //     std::cout << "measured_distances:" << measured_distances[i] << std::endl;
    //     std::cout << "anchor_positions:" << anchor_positions[i] << std::endl;
    //     std::cout << "estimated_Node_position:" << estimated_Node_position << std::endl;
    // }


    // std::cout << summary.BriefReport() << std::endl;
    
    return Eigen::Vector3d(node_pos[0], node_pos[1], node_pos[2]);
}

Eigen::Vector3d NodeLocalizer::estimateNodePositionIterative(
    const std::vector<Eigen::Vector3d>& p_i,
    const std::vector<double>& d_ij,
    const Eigen::Vector3d& v_init,
    int max_iter,
    double tol)
{
    Eigen::Vector3d v = v_init;
    int N_A = p_i.size();
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // 构建雅可比矩阵和残差向量
        Eigen::MatrixXd J(N_A, 3);
        Eigen::VectorXd r(N_A);
        
        for (int i = 0; i < N_A; ++i) {
            Eigen::Vector3d diff = v - p_i[i];
            double dist = diff.norm();
            
            // 避免除零
            if (dist < 1e-9) {
                J.row(i) = Eigen::Vector3d::Zero().transpose();
                r(i) = d_ij[i];
            } else {
                J.row(i) = diff.transpose() / dist;
                r(i) = d_ij[i] - dist;
            }
        }
        
        // 高斯-牛顿更新
        Eigen::Matrix3d H = J.transpose() * J;
        Eigen::Vector3d g = J.transpose() * r;
        
        // 求解线性系统 H * delta = g
        Eigen::Vector3d delta = H.ldlt().solve(g);
        
        // 更新估计
        v += delta;
        
        // 检查收敛
        // if (delta.norm() < tol) {
        //     std::cout << "节点定位优化在 " << iter+1 << " 次迭代后收敛" << std::endl;
        //     break;
        // }
    }
    
    return v;
}


//初值+优化
Eigen::Vector3d NodeLocalizer::estimateNodePosition(
    const std::vector<Eigen::Vector3d>& anchor_positions,
    const std::vector<double>& measured_distances,
    const ExperimentConfig& config,
    Eigen::Vector3d true_node_positions) {
    
    // 先用最小二乘法得到初始估计
    Eigen::Vector3d initial_estimate = estimateNodePositionLeastSquares(
        anchor_positions, measured_distances);

    // if(initial_estimate[1]<1)
    // initial_estimate[1] +=3.0;
    
    // initial_estimate = true_node_positions;//实验所需，需要注释掉

    // std::cout << "initial_node_positions:";
    // for(int k=0;k<3;k++)
    // {
    //     std::cout << "," << initial_estimate[k]; 
    // }
    // std::cout << std::endl;

    // initial_estimate[1] = 0.5;
    
    // 再用Ceres进行优化
    return estimateNodePositionCeres(
        anchor_positions, measured_distances, initial_estimate, config);

    // // 再用迭代最小二乘
    // return estimateNodePositionIterative(
    //     anchor_positions, measured_distances, initial_estimate);
}

std::ofstream data_file_personal("result_least3.csv");
FixedDeque_time time_suqence;
FixedDeque Y_r;
int Equation_Order = 2;
int Data_Quantity = 12;
void NodeLocalizer::weightestimate_0(int sim_id,std::vector<double> measured_distances)
{
    //尝试判别NLOS
    int anchor_size = measured_distances.size();
    double measured_distances_single = measured_distances[0];
    Y_r.add(measured_distances_single);
    time_suqence.add(std::chrono::high_resolution_clock::now());

    if(sim_id==1)
    {
        data_file_personal << "sim_id," << "sigma_m_0," << "sigma_m_15," << "sigma_m_4" << std::endl;
    }

    if(sim_id>Data_Quantity-1)
    {
        Eigen::VectorXd X(Data_Quantity);
        for(int k=0;k<Data_Quantity;k++)
        {
            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(time_suqence.at(k)-time_suqence.at(0));
            X(k) = total_duration.count()/1000.0;
        }
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Equation_Order, Equation_Order);
        Eigen::VectorXd B = Eigen::VectorXd::Zero(Equation_Order);
        for(int i=0;i<Equation_Order;i++)
        {
            for(int j=0;j<Equation_Order;j++)
                for(int k=0;k<Data_Quantity;k++)
                    A(i,j) += pow(X(k),i+j);

            for(int k=0;k<Data_Quantity;k++)
                B(i) += pow(X(k),i)*Y_r.at(k);
        }
        Eigen::VectorXd Solve_X = A.colPivHouseholderQr().solve(B);
        Eigen::VectorXd Y_s = Eigen::VectorXd::Zero(Data_Quantity);
        double square_sum = 0;
        for(int k=0;k<Data_Quantity;k++)
        {
            for(int i=0;i<Equation_Order;i++)
            {
                Y_s(k) += Solve_X(i)*pow(X(k),i);
            }
            square_sum += pow((Y_s(k)-Y_r.at(k)),2);
        }
        double sigma_m = sqrt(square_sum/Data_Quantity);
        std::cout << "sigma_m_0 = " << sigma_m ;

        data_file_personal << sim_id << "," << sigma_m << ",";
        
    }  
}

FixedDeque_time time_suqence_15;
FixedDeque Y_r_15;
void NodeLocalizer::weightestimate_15(int sim_id,std::vector<double> measured_distances)
{
    //尝试判别NLOS
    int anchor_size = measured_distances.size();
    double measured_distances_single = measured_distances[15];
    Y_r_15.add(measured_distances_single);
    time_suqence_15.add(std::chrono::high_resolution_clock::now());

    if(sim_id>Data_Quantity-1)
    {
        Eigen::VectorXd X(Data_Quantity);
        for(int k=0;k<Data_Quantity;k++)
        {
            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(time_suqence_15.at(k)-time_suqence_15.at(0));
            X(k) = total_duration.count()/1000.0;
        }
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Equation_Order, Equation_Order);
        Eigen::VectorXd B = Eigen::VectorXd::Zero(Equation_Order);
        for(int i=0;i<Equation_Order;i++)
        {
            for(int j=0;j<Equation_Order;j++)
                for(int k=0;k<Data_Quantity;k++)
                    A(i,j) += pow(X(k),i+j);

            for(int k=0;k<Data_Quantity;k++)
                B(i) += pow(X(k),i)*Y_r_15.at(k);
        }
        Eigen::VectorXd Solve_X = A.colPivHouseholderQr().solve(B);
        Eigen::VectorXd Y_s = Eigen::VectorXd::Zero(Data_Quantity);
        double square_sum = 0;
        for(int k=0;k<Data_Quantity;k++)
        {
            for(int i=0;i<Equation_Order;i++)
            {
                Y_s(k) += Solve_X(i)*pow(X(k),i);
            }
            square_sum += pow((Y_s(k)-Y_r_15.at(k)),2);
        }
        double sigma_m = sqrt(square_sum/Data_Quantity);
        std::cout << "   sigma_m_15 = " << sigma_m;

        data_file_personal << sigma_m << "," ;
    }  
}

FixedDeque_time time_suqence_4;
FixedDeque Y_r_4;
void NodeLocalizer::weightestimate_4(int sim_id,std::vector<double> measured_distances)
{
    //尝试判别NLOS
    int anchor_size = measured_distances.size();
    double measured_distances_single = measured_distances[4];
    Y_r_4.add(measured_distances_single);
    time_suqence_4.add(std::chrono::high_resolution_clock::now());

    if(sim_id>Data_Quantity-1)
    {
        Eigen::VectorXd X(Data_Quantity);
        for(int k=0;k<Data_Quantity;k++)
        {
            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(time_suqence_4.at(k)-time_suqence_4.at(0));
            X(k) = total_duration.count()/1000.0;
        }
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Equation_Order, Equation_Order);
        Eigen::VectorXd B = Eigen::VectorXd::Zero(Equation_Order);
        for(int i=0;i<Equation_Order;i++)
        {
            for(int j=0;j<Equation_Order;j++)
                for(int k=0;k<Data_Quantity;k++)
                    A(i,j) += pow(X(k),i+j);

            for(int k=0;k<Data_Quantity;k++)
                B(i) += pow(X(k),i)*Y_r_4.at(k);
        }
        Eigen::VectorXd Solve_X = A.colPivHouseholderQr().solve(B);
        Eigen::VectorXd Y_s = Eigen::VectorXd::Zero(Data_Quantity);
        double square_sum = 0;
        for(int k=0;k<Data_Quantity;k++)
        {
            for(int i=0;i<Equation_Order;i++)
            {
                Y_s(k) += Solve_X(i)*pow(X(k),i);
            }
            square_sum += pow((Y_s(k)-Y_r_4.at(k)),2);
        }
        double sigma_m = sqrt(square_sum/Data_Quantity);
        std::cout << "   sigma_m_4 = " << sigma_m << std::endl;

        data_file_personal << sigma_m << std::endl;
    }  
}