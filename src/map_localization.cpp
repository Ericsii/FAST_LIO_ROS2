#include "map_localization.h"
#include <pcl/io/pcd_io.h>
#include <Eigen/Dense>

MapLocalization::MapLocalization() {}
MapLocalization::~MapLocalization() {}

void MapLocalization::init(rclcpp::Node* node) {
  node_ = node;
  logger_ = node->get_logger();
  // params already declared via IESKFParams::declare_params elsewhere; read copy:
  IESKFParams::get_params(node_, params_);
  RCLCPP_INFO(logger_, "MapLocalization init: map_dedupe=%d map_res=%.3f weight_th=%.3f",
              params_.enable_map_dedupe, params_.map_resolution, params_.weight_threshold);
}

bool MapLocalization::loadMapPCD(const std::string& pcd_path) {
  if (pcl::io::loadPCDFile<PointT>(pcd_path, *map_) == -1) {
    RCLCPP_ERROR(logger_, "Failed to load PCD: %s", pcd_path.c_str());
    return false;
  }
  kdtree_.setInputCloud(map_);
  RCLCPP_INFO(logger_, "Loaded map PCD %s, points=%zu", pcd_path.c_str(), map_->size());
  return true;
}

int MapLocalization::computeResidualsAndJacobians(
  const CloudT::ConstPtr& scan,
  const Eigen::Matrix4d& pose_estimate,
  std::vector<int>& indices_used,
  std::vector<double>& residuals,
  std::vector<Eigen::RowVectorXd>& jacobians,
  std::vector<double>& weights)
{
  indices_used.clear();
  residuals.clear();
  jacobians.clear();
  weights.clear();

  if (!map_ || map_->empty()) { return 0; }

  const int state_dim = 6; // using minimal pose delta [rx,ry,rz, tx,ty,tz]
  residuals.reserve(scan->size());
  jacobians.reserve(scan->size());
  weights.reserve(scan->size());

  // Precompute rotation/translation from pose_estimate
  Eigen::Matrix3d R = pose_estimate.block<3,3>(0,0);
  Eigen::Vector3d t = pose_estimate.block<3,1>(0,3);

  std::vector<int> nn_indices;
  std::vector<float> nn_dists;

  for (size_t i=0;i<scan->size();++i) {
    const auto &pt = scan->at(i);
    PointT query; query.x = pt.x; query.y = pt.y; query.z = pt.z; query.intensity = pt.intensity;

    // search k neighbors (k=5)
    int k = 5;
    if (kdtree_.nearestKSearch(query, k, nn_indices, nn_dists) <= 0) continue;

    // fit local plane
    Eigen::Vector3d centroid, normal;
    if (!fitPlaneFromNeighbors(nn_indices, centroid, normal)) continue;

    // compute point in map frame: p_map = R * p_sensor + t
    Eigen::Vector3d p_sensor(pt.x, pt.y, pt.z);
    Eigen::Vector3d p_map_est = R * p_sensor + t;

    // residual r = n^T (p_map_est - centroid)
    double r = normal.dot(p_map_est - centroid);

    // weight by residual magnitude (simple robust scheme)
    double w = 1.0;
    double abs_r = std::abs(r);
    if (abs_r > params_.map_resolution * 5.0) {
      // too far -> likely outlier
      continue;
    }
    // simple Cauchy-like weight: w = 1 / (1 + (r/sigma)^2)
    double sigma = std::max(params_.map_resolution, 0.05);
    w = 1.0 / (1.0 + (r*r)/(sigma*sigma));
    if (w < params_.weight_threshold) continue;

    // jacobian H (1x6) for small-angle rotation approx:
    // r ≈ n^T (R p + t - c) => dr/dtheta = n^T * ( - R * (p) x ) ; dr/dt = n^T
    Eigen::RowVectorXd H(state_dim);
    Eigen::Vector3d Rp = R * p_sensor;
    Eigen::Vector3d cross = Rp; // for rotation derivative use cross product matrix
    // H_rot = - n^T * (R * [p]_x) approximated as n^T * (Rp cross)
    Eigen::Vector3d h_rot = normal.cross(Rp); // note sign handling consistent with small-angle
    H(0) = h_rot(0);
    H(1) = h_rot(1);
    H(2) = h_rot(2);
    H(3) = normal(0);
    H(4) = normal(1);
    H(5) = normal(2);

    // push
    indices_used.push_back(static_cast<int>(i));
    residuals.push_back(r);
    jacobians.push_back(H);
    weights.push_back(w);
  }

  RCLCPP_DEBUG(logger_, "computeResiduals: scan size=%zu used=%zu", scan->size(), residuals.size());
  return static_cast<int>(residuals.size());
}

bool MapLocalization::fitPlaneFromNeighbors(const std::vector<int>& nn_idx, Eigen::Vector3d& centroid, Eigen::Vector3d& normal) {
  if (nn_idx.size() < 3) return false;
  // compute centroid
  centroid.setZero();
  for (int idx : nn_idx) {
    const auto &p = map_->at(idx);
    centroid += Eigen::Vector3d(p.x,p.y,p.z);
  }
  centroid /= double(nn_idx.size());
  // covariance
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (int idx : nn_idx) {
    const auto &p = map_->at(idx);
    Eigen::Vector3d d(p.x,p.y,p.z);
    d -= centroid;
    cov += d * d.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
  normal = es.eigenvectors().col(0); // smallest eigenvalue => normal
  if (normal.norm() < 1e-6) return false;
  normal.normalize();
  return true;
}