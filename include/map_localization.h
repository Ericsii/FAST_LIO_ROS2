#pragma once
#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include "ieskf_params.h"
#include "ieskf_utils.h"

// MapLocalization: load prior map, index it, compute scan->map residuals+Jacobian+weights
class MapLocalization {
public:
  using PointT = pcl::PointXYZI;
  using CloudT = pcl::PointCloud<PointT>;

  MapLocalization();
  ~MapLocalization();

  // 初始化：传入 node 用于读取参数/日志
  void init(rclcpp::Node* node);

  // 加载先验地图 (pcd 文件路径)，并建立 KD-tree
  bool loadMapPCD(const std::string& pcd_path);

  // 为当前扫描计算残差/雅可比并返回有效观测数
  // inputs:
  //   scan: 点云（已在雷达坐标系、去畸变、下采样处理）
  //   pose_estimate: 当前滤波器给出的位姿先验（R,t），用于初始线性化
  // outputs (by reference)
  //   indices_used: 索引（可用于统计）
  //   residuals: r vector per observation
  //   jacobians: per observation jacobian row (6 or state-dim columns)
  //   weights: per observation weight [0..1]
  int computeResidualsAndJacobians(
    const CloudT::ConstPtr& scan,
    const Eigen::Matrix4d& pose_estimate,
    std::vector<int>& indices_used,
    std::vector<double>& residuals,
    std::vector<Eigen::RowVectorXd>& jacobians,
    std::vector<double>& weights);

  // Optionally expose statistics
  size_t map_size() const { return map_->size(); }
  void setParams(const IESKFParams& p) { params_ = p; }

private:
  rclcpp::Logger logger_;
  rclcpp::Node* node_{nullptr};

  CloudT::Ptr map_{new CloudT};
  pcl::KdTreeFLANN<PointT> kdtree_;

  IESKFParams params_;

  // helpers
  bool fitPlaneFromNeighbors(const std::vector<int>& nn_idx, Eigen::Vector3d& centroid, Eigen::Vector3d& normal);
  double computePointToPlaneResidual(const Eigen::Vector3d& p_sensor, const Eigen::Matrix4d& pose, const Eigen::Vector3d& map_pt, const Eigen::Vector3d& normal);
};