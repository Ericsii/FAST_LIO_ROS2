#include "map_localization.h"
#include <pcl/io/pcd_io.h>
#include <Eigen/Eigenvalues>
#include <iostream>

MapLocalization::MapLocalization() {}
MapLocalization::~MapLocalization() {}

bool MapLocalization::loadMapPCD(const std::string &pcd_path) {
  if (pcl::io::loadPCDFile<PointT>(pcd_path, *map_) == -1) {
    std::cerr << "[MapLocalization] Failed to load PCD: " << pcd_path << std::endl;
    return false;
  }
  if (map_->empty()) {
    std::cerr << "[MapLocalization] Loaded map is empty: " << pcd_path << std::endl;
    return false;
  }
  kdtree_.setInputCloud(map_);
  std::cout << "[MapLocalization] Loaded map " << pcd_path << ", points=" << map_->size() << std::endl;
  return true;
}

bool MapLocalization::fitPlaneFromNeighbors(const std::vector<int> &nn_idx, Eigen::Vector3d &centroid, Eigen::Vector3d &normal) const {
  if (nn_idx.size() < 3) return false;
  centroid.setZero();
  for (int idx : nn_idx) {
    const auto &p = map_->at(idx);
    centroid += Eigen::Vector3d(p.x, p.y, p.z);
  }
  centroid /= static_cast<double>(nn_idx.size());
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (int idx : nn_idx) {
    const auto &p = map_->at(idx);
    Eigen::Vector3d d(p.x, p.y, p.z);
    d -= centroid;
    cov += d * d.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
  if (es.info() != Eigen::Success) return false;
  // smallest eigenvalue -> normal
  normal = es.eigenvectors().col(0);
  if (normal.norm() < 1e-8) return false;
  normal.normalize();
  return true;
}

int MapLocalization::computeResidualsAndJacobians(
    const typename CloudT::ConstPtr &scan,
    const Eigen::Matrix4d &pose_estimate,
    std::vector<int> &indices_used,
    std::vector<double> &residuals,
    std::vector<Eigen::RowVectorXd> &jacobians,
    std::vector<double> &weights) const
{
  indices_used.clear();
  residuals.clear();
  jacobians.clear();
  weights.clear();

  if (!map_ || map_->empty() || !scan || scan->empty()) return 0;

  const int state_dim = 6; // [rx,ry,rz, tx,ty,tz]
  residuals.reserve(scan->size());
  jacobians.reserve(scan->size());
  weights.reserve(scan->size());

  Eigen::Matrix3d R = pose_estimate.block<3,3>(0,0);
  Eigen::Vector3d t = pose_estimate.block<3,1>(0,3);

  std::vector<int> nn_indices;
  std::vector<float> nn_dists;
  const int k = 5;

  for (size_t i = 0; i < scan->size(); ++i) {
    const auto &pt = scan->at(i);
    PointT query; query.x = pt.x; query.y = pt.y; query.z = pt.z; query.intensity = pt.intensity;

    if (kdtree_.nearestKSearch(query, k, nn_indices, nn_dists) <= 0) continue;

    Eigen::Vector3d centroid, normal;
    if (!fitPlaneFromNeighbors(nn_indices, centroid, normal)) continue;

    Eigen::Vector3d p_sensor(pt.x, pt.y, pt.z);
    Eigen::Vector3d p_map_est = R * p_sensor + t;

    double r = normal.dot(p_map_est - centroid);

    // simple gating by distance
    if (std::abs(r) > map_resolution_ * 5.0) continue;

    // simple Cauchy-like weight
    double sigma = std::max(map_resolution_, 0.03);
    double w = 1.0 / (1.0 + (r*r) / (sigma*sigma));
    if (w < weight_threshold_) continue;

    // jacobian 1x6: dr/dtheta (approx) and dr/dt
    Eigen::RowVectorXd H(state_dim);
    Eigen::Vector3d Rp = R * p_sensor;
    // small-angle rotation derivative approx: dr/dtheta = n^T * ( - R * [p]_x ) 
    // using cross product: (n x (R p)) approximates -n^T R [p]_x
    Eigen::Vector3d h_rot = normal.cross(Rp); // sign consistent with small-angle update used later
    H(0) = h_rot(0);
    H(1) = h_rot(1);
    H(2) = h_rot(2);
    H(3) = normal(0);
    H(4) = normal(1);
    H(5) = normal(2);

    indices_used.push_back(static_cast<int>(i));
    residuals.push_back(r);
    jacobians.push_back(H);
    weights.push_back(w);
  }

  return static_cast<int>(residuals.size());
}