#include "imu_mahony/imu_mahony.h"
#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kRad2Deg = 180.0f / kPi;
}

IMU_Mahony::IMU_Mahony(float kp, float ki)
    : Kp_(kp),
      Ki_(ki),
      exInt_(0.0f),
      eyInt_(0.0f),
      ezInt_(0.0f),
      q_{1.0f, 0.0f, 0.0f, 0.0f},
      roll_(0.0f),
      pitch_(0.0f),
      yaw_(0.0f) {}

void IMU_Mahony::reset() {
    exInt_ = 0.0f;
    eyInt_ = 0.0f;
    ezInt_ = 0.0f;

    q_ = {1.0f, 0.0f, 0.0f, 0.0f};

    roll_ = 0.0f;
    pitch_ = 0.0f;
    yaw_ = 0.0f;
}

float IMU_Mahony::invSqrt(float x) {
    return 1.0f / std::sqrt(x);
}

void IMU_Mahony::normalizeQuaternion() {
    const float norm = invSqrt(q_[0] * q_[0] +
                               q_[1] * q_[1] +
                               q_[2] * q_[2] +
                               q_[3] * q_[3]);

    q_[0] *= norm;
    q_[1] *= norm;
    q_[2] *= norm;
    q_[3] *= norm;
}

void IMU_Mahony::updateRPY() {
    const float w = q_[0];
    const float x = q_[1];
    const float y = q_[2];
    const float z = q_[3];

    // roll: x轴
    const float sinr_cosp = 2.0f * (w * x + y * z);
    const float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
    roll_ = std::atan2(sinr_cosp, cosr_cosp) * kRad2Deg;

    // pitch: y轴
    float sinp = 2.0f * (w * y - z * x);
    if (sinp > 1.0f) sinp = 1.0f;
    if (sinp < -1.0f) sinp = -1.0f;
    pitch_ = std::asin(sinp) * kRad2Deg;

    // yaw: z轴
    const float siny_cosp = 2.0f * (w * z + x * y);
    const float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
    yaw_ = std::atan2(siny_cosp, cosy_cosp) * kRad2Deg;
}

void IMU_Mahony::update(const std::array<float, 3>& gyro,
                        const std::array<float, 3>& acc,
                        float dt) {
    if (dt <= 0.0f) {
        return;
    }

    float gx = gyro[0];
    float gy = gyro[1];
    float gz = gyro[2];

    float ax = acc[0];
    float ay = acc[1];
    float az = acc[2];

    // 加速度模长太小，说明数据不可靠，直接用陀螺积分
    const float acc_norm2 = ax * ax + ay * ay + az * az;
    const bool acc_valid = (acc_norm2 > 1e-12f);

    if (acc_valid) {
        // 归一化加速度
        const float norm = invSqrt(acc_norm2);
        ax *= norm;
        ay *= norm;
        az *= norm;

        // 当前四元数
        const float q0 = q_[0];
        const float q1 = q_[1];
        const float q2 = q_[2];
        const float q3 = q_[3];

        // 由四元数估计出的重力方向（机体系）
        const float vx = 2.0f * (q1 * q3 - q0 * q2);
        const float vy = 2.0f * (q0 * q1 + q2 * q3);
        const float vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

        // 误差 = 实测重力方向 × 估计重力方向
        const float ex = ay * vz - az * vy;
        const float ey = az * vx - ax * vz;
        const float ez = ax * vy - ay * vx;

        // 积分项
        exInt_ += Ki_ * ex * dt;
        eyInt_ += Ki_ * ey * dt;
        ezInt_ += Ki_ * ez * dt;

        // PI 修正后的角速度
        gx += Kp_ * ex + exInt_;
        gy += Kp_ * ey + eyInt_;
        gz += Kp_ * ez + ezInt_;
    }

    // 四元数微分方程
    const float halfDt = 0.5f * dt;

    const float q0 = q_[0];
    const float q1 = q_[1];
    const float q2 = q_[2];
    const float q3 = q_[3];

    const float qDot0 = (-q1 * gx - q2 * gy - q3 * gz) * halfDt;
    const float qDot1 = ( q0 * gx + q2 * gz - q3 * gy) * halfDt;
    const float qDot2 = ( q0 * gy - q1 * gz + q3 * gx) * halfDt;
    const float qDot3 = ( q0 * gz + q1 * gy - q2 * gx) * halfDt;

    q_[0] += qDot0;
    q_[1] += qDot1;
    q_[2] += qDot2;
    q_[3] += qDot3;

    normalizeQuaternion();
    updateRPY();
}

float IMU_Mahony::getRoll() const {
    return roll_;
}

float IMU_Mahony::getPitch() const {
    return pitch_;
}

float IMU_Mahony::getYaw() const {
    return yaw_;
}

std::array<float, 4> IMU_Mahony::getQuaternion() const {
    return q_;
}