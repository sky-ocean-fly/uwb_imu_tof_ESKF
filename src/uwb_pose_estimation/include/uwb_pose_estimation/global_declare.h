#ifndef GLOBAL_DECLARE_H
#define GLOBAL_DECLARE_H

//测试遮挡效应
// Simulate_NLOS_flag:0为所有距离噪声标准差一样，1为部分距离噪声标准差更大
// size_Threshold 噪声标准差更大需要同时基站数量大于size_Threshold才起作用
extern int Simulate_NLOS_flag;
#define SIZE_Threshold 8

//加权最小二乘
//flag:0为不使用加权最小二乘,1且基站数量大于8为使用加权最小二乘
extern double Weighted_Least_Squares_weight;
extern int Weighted_Least_Squares_flag;

//1为使用ToF和IMU计算出的高度，0为使用距离对计算出的高度
extern int Use_ToF_IMU_Height_flag;

// 定义平面类型（单平面 / 多平面）
enum class PlanarType {
    Single,   // 单平面方法（可选择内部算法）
    Multi     // 多平面方法（固定算法）
};

// 定义求解方法
enum class MethodType {
    TwoStep,         // 两步法
    SingleStep,      // 单步法
    Initialization   // 初始化法
};

// 定义坐标平面类型
enum class PlaneType {
    XY,   // x-y 平面（法向量沿 z 轴）
    ZY    // z-y 平面（法向量沿 x 轴）
};


#endif