#ifndef IESKF_PARAMS_H
#define IESKF_PARAMS_H

#include <rclcpp/rclcpp.hpp>

struct IESKFParams {
  bool enable = true;
  int iter_times = 10;
  double converge_thresh = 1e-3;
  double weight_threshold = 0.9;
  bool enable_map_dedupe = true;
  double map_resolution = 0.05;

  static void declare_params(rclcpp::Node* node) {
    node->declare_parameter("ieskf.enable", true);
    node->declare_parameter("ieskf.iter_times", 10);
    node->declare_parameter("ieskf.converge_thresh", 1e-3);
    node->declare_parameter("ieskf.weight_threshold", 0.9);
    node->declare_parameter("mapping.enable_map_dedupe", true);
    node->declare_parameter("mapping.map_resolution", 0.05);
  }
  static void get_params(rclcpp::Node* node, IESKFParams& p) {
    node->get_parameter_or("ieskf.enable", p.enable, p.enable);
    node->get_parameter_or("ieskf.iter_times", p.iter_times, p.iter_times);
    node->get_parameter_or("ieskf.converge_thresh", p.converge_thresh, p.converge_thresh);
    node->get_parameter_or("ieskf.weight_threshold", p.weight_threshold, p.weight_threshold);
    node->get_parameter_or("mapping.enable_map_dedupe", p.enable_map_dedupe, p.enable_map_dedupe);
    node->get_parameter_or("mapping.map_resolution", p.map_resolution, p.map_resolution);
    RCLCPP_INFO(node->get_logger(), "IESKF params: enable=%d iter=%d converge=%.6f weight_threshold=%.3f map_dedupe=%d map_res=%.3f",
      p.enable, p.iter_times, p.converge_thresh, p.weight_threshold, p.enable_map_dedupe, p.map_resolution);
  }
};

#endif
