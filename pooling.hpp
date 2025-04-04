#pragma once
#include "layer.hpp"

class MaxPooling : public Layer {
public:
    MaxPooling(int kernel_size, int stride = -1);
    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) override;
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) override;

private:
    int kernel_size;
    int stride;
    Eigen::MatrixXd max_indices;  // Store indices of max values for backward pass
};

class AveragePooling : public Layer {
public:
    AveragePooling(int kernel_size, int stride = -1);
    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) override;
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) override;

private:
    int kernel_size;
    int stride;
}; 