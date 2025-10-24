// Existing includes...
#include "ieskf_params.h"
#include "ieskf_utils.h"
#include "map_localization.h"
// ...
MapLocalization map_localizer;

// in LaserMappingNode::LaserMappingNode() after params declared and get_params:
map_localizer.init(this);
std::string map_pcd;
this->get_parameter_or("map.pcd_path", map_pcd, std::string(""));
if (!map_pcd.empty()) {
  map_localizer.loadMapPCD(map_pcd);
}
map_localizer.setParams(ieskf_params);

  // preprocess scan -> scan_down
  std::vector<int> idx_used;
  std::vector<double> residuals;
  std::vector<Eigen::RowVectorXd> jacobians;
  std::vector<double> weights;

  int n_meas = map_localizer.computeResidualsAndJacobians(scan_down, pose_estimate, idx_used, residuals, jacobians, weights);

  if (n_meas > 6 && ieskf_params.enable) {
    // convert residuals/jacobians/weights into the format expected by IESKF
    // e.g. push to kf measurement buffer or let run_ieskf_iterations read them
    // Here: pack into the filter's measurement structure (implementation-specific)
    // then call iterated update:
    double solve_H_time = 0;
    run_ieskf_iterations(kf, ieskf_params.iter_times, ieskf_params.converge_thresh, /*point_cov*/ 0.01, solve_H_time);
  } else {
    // fallback to original update or skip
  }

// Global variables
IESKFParams ieskf_params;

LaserMappingNode::LaserMappingNode(...) {
    // Existing constructor code...
    IESKFParams::declare_params(this);
    // Existing get_parameter_or blocks...
    IESKFParams::get_params(this, ieskf_params);
}

void LaserMappingNode::map_incremental() {
    // Existing code...
    for (...) {
        // Existing loop logic...
        if (ieskf_params.enable_map_dedupe && nearest_points.size() > 0) {
            if (compute_distance(nearest_points[0], point) < ieskf_params.map_resolution) {
                need_add = false;
            }
        }
    }
}