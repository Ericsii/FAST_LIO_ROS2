#pragma once
#include <IKFoM_toolkit/esekfom/esekfom.hpp>
#include "use-ikfom.hpp"
#include "map_localization.h"
#include <mutex>
#include <memory>

// Forward: expose functions to register and access shared scan
namespace ieskf_map {

using state_t = state_ikfom;
using dyn_share_t = esekfom::dyn_share_datastruct<double>;

// Initialize measurement wrapper (call once after kf created)
void init_map_measurement(esekfom::esekf<state_t, 12, input_ikfom> &kf, MapLocalization *map_localizer_ptr);

// Set current downsampled scan (thread-safe): node should call this before each update
void set_current_scan(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &scan);

// The measurement function that will be registered to kf:
void h_dyn_share(state_t &x, dyn_share_t &dyn_share);

} // namespace ieskf_map