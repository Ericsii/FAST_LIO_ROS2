#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <vector>
#include <string>

class MapLocalization {
public:
  using PointT = pcl::PointXYZI;
  using CloudT = pcl::PointCloud<PointT>;

  MapLocalization();
  ~MapLocalization();

  // Load prior map from PCD file. Returns true on success.
  bool loadMapPCD(const std::string &pcd_path);

  // Set map resolution used for gating/weighting
  void setMapResolution(double r) { map_resolution_ = r; }

  // Set simple weight threshold
  void setWeightThreshold(double w) { weight_threshold_ = w; }

  // Compute residuals/jacobians/weights for a scan (scan should be in sensor coordinates).
  // pose_estimate: 4x4 transform (world <- sensor) used as linearization point.
  // Returns number of valid correspondences.
  int computeResidualsAndJacobians(
    const typename CloudT::ConstPtr &scan,
    const Eigen::Matrix4d &pose_estimate,
    std::vector<int> &indices_used,
    std::vector<double> &residuals,
    std::vector<Eigen::RowVectorXd> &jacobians,
    std::vector<double> &weights) const;

  size_t mapSize() const { return map_->size(); }

private:
  CloudT::Ptr map_{new CloudT};
  pcl::KdTreeFLANN<PointT> kdtree_;
  double map_resolution_ = 0.05;
  double weight_threshold_ = 0.02;

  // Fit plane to neighbors, return centroid and normal (normalized). Returns false if fail.
  bool fitPlaneFromNeighbors(const std::vector<int> &nn_idx, Eigen::Vector3d &centroid, Eigen::Vector3d &normal) const;
};