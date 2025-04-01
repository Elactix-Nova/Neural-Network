#pragma once
#include "layer.hpp"
#include <vector>
#include <random>

class Convolutional : public Layer {
public:
    Convolutional(const std::vector<int>& input_shape, int kernel_size, int depth);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& output_gradient, double learning_rate) override;
 
private:
    int depth;
    std::vector<int> input_shape;
    int input_depth;
    std::vector<int> output_shape;
    std::vector<int> kernels_shape;
    std::vector<Eigen::MatrixXd> kernels;
    std::vector<Eigen::MatrixXd> biases;
    std::random_device rd;
    std::mt19937 gen;

    // Helper functions for convolution operations
    Eigen::MatrixXd correlate2d(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel);
    Eigen::MatrixXd convolve2d(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel);
}; 