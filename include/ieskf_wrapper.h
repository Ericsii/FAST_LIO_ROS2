#ifndef IESKF_WRAPPER_H
#define IESKF_WRAPPER_H

#include "ieskf_params.h"

inline void run_ieskf_iterations(esekfom::esekf<state_ikfom, 12, input_ikfom> &kf, int iter_times, double converge_thresh, double point_cov, double &solve_H_time) {
    // Conservative wrapper: call existing iterated update the requested number of times.
    // This is a safe WIP: it does not change algorithm behavior, only centralizes future IESKF logic.
    for (int it = 0; it < iter_times; ++it) {
        kf.update_iterated_dyn_share_modified(point_cov, solve_H_time);
    }
}

#endif
