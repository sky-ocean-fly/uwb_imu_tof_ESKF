#pragma once

#include <array>

class IMU_Mahony {
public:
    explicit IMU_Mahony(float kp = 1.0f, float ki = 0.01f);

    void reset();

    void update(const std::array<float, 3>& gyro,
                const std::array<float, 3>& acc,
                float dt);

    float getRoll() const;
    float getPitch() const;
    float getYaw() const;

    std::array<float, 4> getQuaternion() const;

private:
    static float invSqrt(float x);
    void normalizeQuaternion();
    void updateRPY();

private:
    float Kp_;
    float Ki_;

    float exInt_;
    float eyInt_;
    float ezInt_;

    // 四元数，顺序为 [w, x, y, z]
    std::array<float, 4> q_;

    float roll_;   // 单位：deg
    float pitch_;  // 单位：deg
    float yaw_;    // 单位：deg
};