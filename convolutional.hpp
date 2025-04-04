#pragma once
#include "layer.hpp"
#include <vector>
#include <random>
#include <Eigen/Dense>

class Convolutional : public Layer {
public:
    // Constructor
    Convolutional(const std::vector<int>& input_shape, int kernel_size, int depth);

    // Forward and backward pass
    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) override;
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) override;

public: 
    // Layer parameters
    int depth;
    int input_depth;
    int input_height;
    int input_width;
    int kernel_size;
    int output_height;
    int output_width;

    // Kernels and biases
    std::vector<std::vector<Eigen::MatrixXd>> kernels; // [depth][input_depth][kernel_size x kernel_size]
    std::vector<Eigen::MatrixXd> biases; // [depth][output_height x output_width]

    // Random number generation
    std::random_device rd;
    std::mt19937 gen;

    // Helper methods for convolution operations
    Eigen::MatrixXd correlate2d(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel);
    Eigen::MatrixXd convolve2d(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel);
};