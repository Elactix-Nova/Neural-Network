#pragma once
#include "layer.hpp"
#include <vector>

class Reshape : public Layer {
public:
    Reshape(const std::vector<int>& new_shape);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& output_gradient, double learning_rate) override;

private:
    std::vector<int> new_shape;
    std::vector<int> old_shape;
}; 