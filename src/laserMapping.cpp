// Existing includes...
#include "ieskf_params.h"
#include "ieskf_utils.h"

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