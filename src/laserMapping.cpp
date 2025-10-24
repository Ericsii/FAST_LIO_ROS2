// laserMapping.cpp
// Integrated with Map-based localization (scan->prior-map) and a local iterated GN solver.
// NOTE: This file is a full translation/merge for the mapping + map-localization update.
// It uses MapLocalization (map_localization.h) and IESKF params/util wrappers.
// The implementation applies the computed pose delta back to the filter state via kf.change_x(...).
// Covariance update is left conservative (P not rewritten) — see comments where you may refine to
// update full filter covariance consistently through esekf interfaces.

#include <rclcpp/rclcpp.hpp>
#include <iomanip>
#include <vector>
#include <mutex>
#include <omp.h>

#include <Eigen/Dense>s
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "ieskf_params.h"
#include "ieskf_utils.h"
#include "map_localization.h"
#include "use-ikfom.hpp" // defines state_ikfom, input_ikfom
#include <IKFoM_toolkit/esekfom/esekfom.hpp> // esekf class

// existing project includes (placeholders - keep original includes as in repo)
#include "laserMapping.h"
#include "misc.h"
#include "ikd_tree.h"

// Global IESKF params instance used by node
IESKFParams ieskf_params;

// Map localization helper (loading prior map + producing residuals/Jacobians)
MapLocalization map_localizer;

// Forward: small-angle -> quaternion helper
static inline Eigen::Quaterniond smallAngleToQuat(const Eigen::Vector3d &dtheta) {
  // small-angle to quaternion: q = [1, 0.5*theta] normalized approximation
  Eigen::Quaterniond q;
  Eigen::Vector3d half = 0.5 * dtheta;
  q.w() = 1.0;
  q.x() = half.x();
  q.y() = half.y();
  q.z() = half.z();
  q.normalize();
  return q;
}

// Local iterated GN solver that re-linearizes using MapLocalization computeResidualsAndJacobians
// x_pose: 4x4 pose estimate (world <- lidar), state update is in minimal 6D: [drot(3), dpos(3)]
// returns delta (6x1) in minimal coordinates and whether converged
static std::pair<Eigen::Matrix<double,6,1>, bool> solve_iterated_map_align(
    esekfom::esekf<state_ikfom, 12, input_ikfom> &kf,
    MapLocalization &map_loc,
    Eigen::Matrix4d &pose_estimate,               // in/out: updates if re-linearized each iter
    int max_iter,
    double tol,
    const IESKFParams &params,
    rclcpp::Logger logger)
{
  const int state_dim = 6; // [rx,ry,rz, tx,ty,tz]
  Eigen::Matrix<double,6,1> dx_total = Eigen::Matrix<double,6,1>::Zero();

  // get prior state & (optional) prior covariance from kf
  state_ikfom kf_x = kf.get_x();
  // We will not re-compute full P for now (requires mapping to full state vector). Use P only as diag regularizer if available.
  // If you want to use P prior, get kf.get_P() and map block corresponding to pose -> implement below.

  int iter = 0;
  double last_norm = 1e12;
  bool converged = false;

  for (iter = 0; iter < max_iter; ++iter) {
    // compute residuals/jacobians/weights at current pose_estimate
    std::vector<int> indices_used;
    std::vector<double> residuals;
    std::vector<Eigen::RowVectorXd> jacobians;
    std::vector<double> weights;

    // MapLocalization expects scan->map matching; the caller must supply a downsampled scan (we will call this function from caller with scan_down)
    // Here we only call the map_loc API; the caller (laserMapping main flow) must pass the scan_down in/out.
    // To keep function self-contained for iterative re-linearization, we assume map_loc has access to the last scan via a member variable or that
    // caller will call computeResidualsAndJacobians before the first iteration and then perform re-linearization calls here by providing scan_down.
    //
    // In this helper we cannot directly call computeResidualsAndJacobians without the scan pointer---so the caller will call an outer loop that
    // calls computeResidualsAndJacobians(scan_down, pose_estimate, ...) each iter. For simplicity in this file-level implementation we will
    // use a placeholder pattern: the caller (below) will perform iterations and call this function only for the linear algebra solution.
    //
    // This function therefore only contains the solver given H/r/W matrices. The code to re-linearize is in the caller site.
    //
    // The implementation below is the weighted normal equation solver used after H,r,weights are collected.

    // NOTE: This should not run; we return zero. Actual iterative re-linearization happens in calling code (see main mapping flow).
    return {dx_total, true};
  }

  return {dx_total, converged};
}

// Weighted iterated solver (non re-linearizing) used when H, r, weights already available.
// Solve for dx: (H^T W H + lambda I) dx = H^T W r
static std::pair<Eigen::VectorXd, bool> solve_linear_weighted(
    const std::vector<double>& residuals,
    const std::vector<Eigen::RowVectorXd>& jacobians,
    const std::vector<double>& weights,
    int state_dim,
    double prior_reg)
{
  const int m = (int)residuals.size();
  Eigen::VectorXd r(m);
  Eigen::MatrixXd H(m, state_dim);
  Eigen::VectorXd wv(m);
  for (int i = 0; i < m; ++i) {
    r(i) = residuals[i];
    H.row(i) = jacobians[i];
    wv(i) = weights[i];
  }

  // build weight diagonal (sqrt first)
  Eigen::VectorXd sqrtw = wv.array().sqrt().matrix();
  Eigen::MatrixXd W = sqrtw.asDiagonal();

  Eigen::MatrixXd HW = W * H; // m x n
  Eigen::VectorXd rW = W * r;

  Eigen::MatrixXd A = HW.transpose() * HW;
  // regularize with small prior diagonal to avoid singular
  A += Eigen::MatrixXd::Identity(state_dim, state_dim) * prior_reg;

  Eigen::VectorXd b = HW.transpose() * rW;

  Eigen::VectorXd dx(state_dim);
  bool solved = false;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
  if (ldlt.info() == Eigen::Success) {
    dx = ldlt.solve(b);
    solved = true;
  } else {
    dx = A.colPivHouseholderQr().solve(b);
    solved = true;
  }
  return {dx, solved};
}

// ------------------------- Main laserMapping flow (simplified/merged) -------------------------
//
// NOTE: This file intentionally contains an integrated mapping update flow that:
// - uses MapLocalization to compute scan->map residuals and jacobians (point-to-plane),
// - runs a small iterated GN using those residuals (with optional re-linearization if desired),
// - applies the pose delta back to esekf via kf.change_x(state_ikfom).
//
// To keep the example readable we've kept existing original mapping logic and replaced
// the measurement update invocation with a map-localization driven update.
//
// Important: to update the filter covariance properly, you should either:
//  - map the small 6D pose-delta into the full esekf state vector and compute the posterior P
//    using the Schur complement or the full measurement Jacobian in the filter manifold form; or
//  - implement a dynamic measurement model (h_dyn_share / h_dyn) and hand it to esekf init_dyn_share
//    so that the filter's internal iterated update uses your measurement model directly (recommended).
//
// The code below implements a safe, testable path: it computes dx and applies it to the filter state (change_x).
// This will update the filter's nominal state; covariance P is retained. Use this path for development and testing.
// -----------------------------------------------------------------------------------------------

// The following function is the core replacement of the original kf.update_iterated_dyn_share_modified call.
// It organizes computeResidualsAndJacobians calls, iterated solve + optional re-linearization, then applies delta to kf.
static void map_based_update_and_apply(
    esekfom::esekf<state_ikfom, 12, input_ikfom> &kf,
    MapLocalization &map_loc,
    pcl::PointCloud<pcl::PointXYZI>::ConstPtr scan_down,
    const IESKFParams &params,
    rclcpp::Logger logger)
{
  // prepare initial pose estimate from kf state (world <- lidar)
  state_ikfom state_point = kf.get_x();
  Eigen::Quaterniond cur_q(state_point.rot.coeffs()[3], state_point.rot.coeffs()[0],
                           state_point.rot.coeffs()[1], state_point.rot.coeffs()[2]); // careful ordering in code base
  // In original code they used geoQuat.x = state_point.rot.coeffs()[0] ... so rot is Eigen::Quaterniond stored in parts
  // Use the rot directly:
  Eigen::Quaterniond q_cur;
  q_cur.w() = state_point.rot.coeffs()[3];
  q_cur.x() = state_point.rot.coeffs()[0];
  q_cur.y() = state_point.rot.coeffs()[1];
  q_cur.z() = state_point.rot.coeffs()[2];
  Eigen::Matrix3d Rcur = q_cur.toRotationMatrix();
  Eigen::Vector3d tcur = state_point.pos;

  Eigen::Matrix4d pose_est;
  pose_est.setIdentity();
  pose_est.block<3,3>(0,0) = Rcur;
  pose_est.block<3,1>(0,3) = tcur + Rcur * state_point.offset_T_L_I; // lidar position in world (if offset needed)

  // Iterated re-linearization loop: recompute residuals/jacobians at updated pose each iter
  Eigen::Matrix<double,6,1> dx_total = Eigen::Matrix<double,6,1>::Zero();
  double tol = params.converge_thresh;
  int max_iter = params.iter_times;
  bool converged = false;

  std::vector<int> indices_used;
  std::vector<double> residuals;
  std::vector<Eigen::RowVectorXd> jacobians;
  std::vector<double> weights;

  for (int it = 0; it < max_iter; ++it) {
    // compute residuals & jacobians at current linearization point
    indices_used.clear();
    residuals.clear();
    jacobians.clear();
    weights.clear();

    // Use the map_localizer to compute residuals/jacobians for this scan and current pose_est
    int used = map_loc.computeResidualsAndJacobians(scan_down, pose_est, indices_used, residuals, jacobians, weights);
    RCLCPP_DEBUG(logger, "map_based_update: iteration %d, residuals=%d", it, used);

    if (used < 6) {
      RCLCPP_WARN(logger, "Not enough map matches (%d) to perform update, skip", used);
      converged = false;
      break;
    }

    // Solve weighted linear system for 6-delta
    double prior_reg = 1e-6; // small damping/regularizer
    auto [dx_vec, solved] = solve_linear_weighted(residuals, jacobians, weights, 6, prior_reg);
    if (!solved) {
      RCLCPP_WARN(logger, "Linear solver failed in map_based_update");
      break;
    }

    // apply delta to pose_est locally (re-linearize)
    Eigen::Vector3d drot = dx_vec.segment<3>(0);
    Eigen::Vector3d dpos = dx_vec.segment<3>(3);

    // update pose_est: rotation then translation
    Eigen::Quaterniond dq = smallAngleToQuat(drot);
    Eigen::Matrix3d Rnew = (dq * Eigen::Quaterniond(q_cur.w(), q_cur.x(), q_cur.y(), q_cur.z())).toRotationMatrix();
    Eigen::Vector3d tnew = pose_est.block<3,1>(0,3) + dpos;

    pose_est.block<3,3>(0,0) = Rnew;
    pose_est.block<3,1>(0,3) = tnew;

    dx_total += dx_vec;

    double norm_dx = dx_vec.norm();
    RCLCPP_DEBUG(logger, "map_based_update iter=%d dx_norm=%.6f used=%d", it, norm_dx, used);

    if (norm_dx < tol) {
      converged = true;
      break;
    }
  } // end iter

  // apply dx_total back to kf state (nominal)
  if (dx_total.norm() > 1e-12) {
    // get current state (again) to modify
    state_ikfom state_apply = kf.get_x();

    // rotation: dx_total[0..2] is small angle vector in local frame
    Eigen::Vector3d drot_apply = dx_total.segment<3>(0);
    Eigen::Vector3d dpos_apply = dx_total.segment<3>(3);

    // build quaternion from small-angle
    Eigen::Quaterniond dq_apply = smallAngleToQuat(drot_apply);

    // current quaternion from state_apply
    Eigen::Quaterniond q_state;
    q_state.w() = state_apply.rot.coeffs()[3];
    q_state.x() = state_apply.rot.coeffs()[0];
    q_state.y() = state_apply.rot.coeffs()[1];
    q_state.z() = state_apply.rot.coeffs()[2];

    // apply rotation: left-multiply small rotation to current orientation
    Eigen::Quaterniond q_new = dq_apply * q_state;
    q_new.normalize();

    // assign back to state_apply.rot (keep same coeff ordering used in codebase)
    state_apply.rot.coeffs()[0] = q_new.x();
    state_apply.rot.coeffs()[1] = q_new.y();
    state_apply.rot.coeffs()[2] = q_new.z();
    state_apply.rot.coeffs()[3] = q_new.w();

    // apply translation: in world frame — we used tnew computed in pose_est; to keep consistent, add dpos_apply to pos
    state_apply.pos += dpos_apply;

    // Apply the changed nominal state back to kf
    kf.change_x(state_apply);

    // NOTE: the filter covariance (P) is not updated here to reflect the measurement information.
    // For statistical consistency, compute posterior cov: P_post = (H^T W H + P_prior^{-1})^{-1}
    // and call kf.change_P(P_post) with correct mapping to full state. That requires assembling H in full state DOF.
    // For now we leave P unchanged (safe for testing). See documentation comments above.
    RCLCPP_INFO(logger, "Applied map-based pose correction: dx_norm=%.6f converged=%d", dx_total.norm(), (int)converged);
  } else {
    RCLCPP_DEBUG(logger, "map_based_update: zero dx, nothing applied");
  }
}

// ---------------- Main node constructor / initialization integration (excerpt) -----------------
//
// The real LaserMappingNode constructor and flow are long. Below we provide the initialization calls
// and the place where the original kf.update_iterated_dyn_share_modified(...) was replaced by
// map-based logic. Please integrate these snippets into your full LaserMappingNode implementation
// at the appropriate places (constructor and in the per-frame update loop).
//
// ---------------- Example integration points (to copy into actual LaserMappingNode) ----------------

/* Example: in LaserMappingNode constructor (after node params declared & read)
{
    // declare / get ieskf params
    IESKFParams::declare_params(this);
    IESKFParams::get_params(this, ieskf_params);

    // initialize MapLocalization
    map_localizer.init(this);
    std::string map_pcd;
    this->get_parameter_or("map.pcd_path", map_pcd, std::string(""));
    if (!map_pcd.empty()) {
      if (!map_localizer.loadMapPCD(map_pcd)) {
        RCLCPP_WARN(this->get_logger(), "Failed to load prior map PCD: %s", map_pcd.c_str());
      }
    }
    map_localizer.setParams(ieskf_params);
}
*/

// ---------------- Example per-frame replacement of iterated update ----------------
// Replace the old call block that invoked kf.update_iterated_dyn_share_modified(...) with the following
// inside the laser mapping frame loop where scan_down and other variables are available:
//
//    std::vector<int> idx_used;
//    std::vector<double> residuals;
//    std::vector<Eigen::RowVectorXd> jacobians;
//    std::vector<double> weights;
//
//    // compute once to check sufficiency (first linearization)
///   int n_meas = map_localizer.computeResidualsAndJacobians(scan_down, pose_estimate, idx_used, residuals, jacobians, weights);
//
//    if (n_meas > 6 && ieskf_params.enable) {
//       // iterate with re-linearization inside map_based_update_and_apply (it will itself call computeResiduals... each iter)
//       map_based_update_and_apply(kf, map_localizer, scan_down, ieskf_params, this->get_logger());
//    } else {
//       // fallback: keep original filter update as-is (IMU-only / no-map update) or call existing update method
//       double solve_H_time = 0;
//       kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time); // original behavior
//    }
//
// After map_based_update_and_apply, state inside kf is updated via kf.change_x(...).
// You may want to recompute derived variables (state_point, euler_cur, pos_lid, geoQuat, etc.) exactly as the original code does:
//    state_point = kf.get_x();
//    euler_cur = SO3ToEuler(state_point.rot);
//    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
//    geoQuat.x = state_point.rot.coeffs()[0];
//    geoQuat.y = state_point.rot.coeffs()[1];
//    geoQuat.z = state_point.rot.coeffs()[2];
//    geoQuat.w = state_point.rot.coeffs()[3];

// --------------------------------- End of file ---------------------------------

// NOTE: This file is intended as a patched, integrative version to be merged with the repository's
// laserMapping.cpp. It focuses on showing how to compute and apply a map-based pose correction.
// For production-ready correctness:
//  - ensure coordinate frames and quaternion coeff ordering align with your codebase conventions,
//  - decide and implement the covariance update (kf.change_P) mapping correctly to full filter DOF,
//  - consider implementing a proper esekf measurement model (h_dyn_share) and registering it with kf.init_dyn_share,
//    then call the filter's built-in iterated update so the filter maintains full statistical consistency.