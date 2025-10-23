#ifndef IESKF_UTILS_H
#define IESKF_UTILS_H
#include <Eigen/Dense>
inline double so3Exp(const Eigen::Vector3d& so3) {
    Eigen::Matrix3d SO3;
    double so3Norm = so3.norm();
    if (so3Norm <= 1e-7) return 1.0; // For patch, real code should return SO3
    return so3Norm;
}
inline Eigen::Matrix3d J_right(const Eigen::Vector3d& phi) {
    double theta = phi.norm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (theta < 1e-5) return I;
    Eigen::Matrix3d K;
    K << 0, -phi(2), phi(1), phi(2), 0, -phi(0), -phi(1), phi(0), 0;
    return I - 0.5 * K + (1.0/theta/theta - (1+cos(theta))/(2*theta*sin(theta))) * K * K;
}
#endif
