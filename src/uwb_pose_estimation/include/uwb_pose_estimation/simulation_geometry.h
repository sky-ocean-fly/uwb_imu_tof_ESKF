#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

//仿真布局
namespace uwb_pose_estimation
{
namespace simulation_geometry
{

//基站（无人车A）ID
inline const std::vector<int> &anchorIds()
{
    static const std::vector<int> ids = {0, 1, 2, 3, 4, 5};
    return ids;
}
//标签（无人机B）ID
inline const std::vector<int> &tagIds()
{
    static const std::vector<int> ids = {6, 7, 8, 9, 10, 11};
    return ids;
}
//分布在圆上
inline const std::vector<Eigen::Vector3d> &anchorPositionsA()
{
    // Six anchors uniformly distributed on a 0.5 m diameter circle in A's x-y plane.
    static const std::vector<Eigen::Vector3d> positions = [] {
        std::vector<Eigen::Vector3d> p;
        p.reserve(6);
        constexpr double radius = 0.3;//0.25;
        for (int i = 0; i < 6; ++i)
        {
            const double angle = 2.0 * M_PI * static_cast<double>(i) / 6.0;
            p.emplace_back(radius * std::cos(angle), radius * std::sin(angle), 0.0);
        }
        return p;
    }();
    return positions;
}
//分布在圆上
inline const std::vector<Eigen::Vector3d> &tagPositionsB()
{
    // Six tags uniformly distributed on a 0.3 m diameter circle in B's x-y plane.
    static const std::vector<Eigen::Vector3d> positions = [] {
        std::vector<Eigen::Vector3d> p;
        p.reserve(6);
        constexpr double radius = 0.3;//0.15;
        for (int i = 0; i < 6; ++i)
        {
            const double angle = 2.0 * M_PI * static_cast<double>(i) / 6.0;
            p.emplace_back(radius * std::cos(angle), radius * std::sin(angle), 0.0);
        }
        return p;
    }();
    return positions;
}
//ToF的位置
inline const Eigen::Vector3d &tofPositionB()
{
    static const Eigen::Vector3d p(0.0, 0.0, -0.10);
    return p;
}
//tof的光束方向
inline const Eigen::Vector3d &tofRayB()
{
    static const Eigen::Vector3d s(0.0, 0.0, -1.0);
    return s;
}

} // namespace simulation_geometry
} // namespace uwb_pose_estimation
