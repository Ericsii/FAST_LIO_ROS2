#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    auto node = rclcpp::Node::make_shared("fastlio_mapping");
    RCLCPP_INFO(node->get_logger(), "FAST-LIO mapping node started");
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}