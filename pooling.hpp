#pragma once
#include "layer.hpp"

class MaxPooling : public Layer {
public:
    MaxPooling(int kernel_size);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& output_gradient, double learning_rate) override;

private:
    int kernel_size;
    Eigen::MatrixXd max_indices;  // Store indices of max values for backward pass
};

class AveragePooling : public Layer {
public:
    AveragePooling(int kernel_size);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& output_gradient, double learning_rate) override;

private:
    int kernel_size;
}; 