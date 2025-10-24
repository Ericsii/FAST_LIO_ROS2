#include "ieskf_map_measurement.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace ieskf_map {

static MapLocalization *g_map_localizer = nullptr;
static pcl::PointCloud<pcl::PointXYZI>::ConstPtr g_current_scan = nullptr;
static std::mutex g_scan_mutex;
static esekfom::esekf<state_t, 12, input_ikfom> *g_kf_ptr = nullptr;

void init_map_measurement(esekfom::esekf<state_t, 12, input_ikfom> &kf, MapLocalization *map_localizer_ptr) {
  g_kf_ptr = &kf;
  g_map_localizer = map_localizer_ptr;
  // Note: actual registration to kf must be done where kf.init_dyn_share is called with h_dyn_share.
}

void set_current_scan(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &scan) {
  std::lock_guard<std::mutex> lg(g_scan_mutex);
  g_current_scan = scan;
}

void h_dyn_share(state_t &x, dyn_share_t &dyn_share) {
  // basic validity init
  dyn_share.valid = false;
  dyn_share.converge = true;

  if (!g_map_localizer) return;

  pcl::PointCloud<pcl::PointXYZI>::ConstPtr scan_snapshot;
  {
    std::lock_guard<std::mutex> lg(g_scan_mutex);
    scan_snapshot = g_current_scan;
  }
  if (!scan_snapshot || scan_snapshot->empty()) {
    return;
  }

  // 1) Build pose_estimate from x (rotation + position)
  Eigen::Quaterniond q;
  // NOTE: adapt this to actual state_ikfom rot storage ordering if different
  q.w() = x.rot.coeffs()[3];
  q.x() = x.rot.coeffs()[0];
  q.y() = x.rot.coeffs()[1];
  q.z() = x.rot.coeffs()[2];
  Eigen::Matrix4d pose_est = Eigen::Matrix4d::Identity();
  pose_est.block<3,3>(0,0) = q.toRotationMatrix();
  pose_est.block<3,1>(0,3) = x.pos; // might need to add offset_T_L_I depending on convention

  // 2) compute residuals/jacobians/weights from map_localizer
  std::vector<int> indices;
  std::vector<double> residuals;
  std::vector<Eigen::RowVectorXd> jacobians6; // each row is 1x6
  std::vector<double> weights;
  int used = g_map_localizer->computeResidualsAndJacobians(scan_snapshot, pose_est, indices, residuals, jacobians6, weights);
  if (used <= 0) return;

  // 3) fill dyn_share.h (residual vector)
  int m = static_cast<int>(residuals.size());
  dyn_share.h = Eigen::Matrix<double, Eigen::Dynamic, 1>(m);
  for (int i = 0; i < m; ++i) dyn_share.h(i) = residuals[i];

  // 4) Build dyn_share.h_x: m x n  (n = state::DOF)
  const int n = state_t::DOF;
  dyn_share.h_x = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(m, n);
  dyn_share.h_x.setZero();

  // Determine indices for rotation and position in the esekf state vector
  // WARNING: The index mapping must match the state_ikfom order. Typical order: pos (0..2), rot (3..5)
  // If your state differs, adjust idx_pos and idx_rot accordingly.
  const int idx_pos = 0;
  const int idx_rot = 3;

  for (int i = 0; i < m; ++i) {
    const Eigen::RowVectorXd &H6 = jacobians6[i]; // 1x6
    // rotation part -> columns idx_rot .. idx_rot+2
    dyn_share.h_x(i, idx_rot + 0) = H6(0);
    dyn_share.h_x(i, idx_rot + 1) = H6(1);
    dyn_share.h_x(i, idx_rot + 2) = H6(2);
    // translation part -> columns idx_pos .. idx_pos+2
    dyn_share.h_x(i, idx_pos + 0) = H6(3);
    dyn_share.h_x(i, idx_pos + 1) = H6(4);
    dyn_share.h_x(i, idx_pos + 2) = H6(5);
  }

  // 5) h_v (mapping from measurement noise to measurement space)
  dyn_share.h_v = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(m, m);

  // 6) R: measurement noise covariance (m x m). Use a diagonal with LASER_POINT_COV scaled, or use weights
  dyn_share.R = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(m, m);
  dyn_share.R.setZero();
  double point_cov = 0.01; // placeholder, replace with LASER_POINT_COV param
  for (int i = 0; i < m; ++i) {
    double w = std::max(1e-6, weights[i]);
    // set R entry inversely proportional to weight (higher weight -> smaller variance)
    dyn_share.R(i, i) = point_cov / w;
  }

  dyn_share.valid = true;
  dyn_share.converge = true;
}

} // namespace ieskf_map