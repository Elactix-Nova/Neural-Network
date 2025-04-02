#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <Eigen/Dense>

class Convolutional {
public:
    // Constructor
    Convolutional(std::vector<int> input_shape, int kernel_size, int depth);

    // Forward and backward pass
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> backward(
        const std::vector<std::vector<std::vector<double>>>& output_gradient,
        const std::vector<std::vector<std::vector<double>>>& input,
        double learning_rate);

    // Public member variables
    int depth, input_depth, input_height, input_width;
    int kernel_size;
    int output_height, output_width;
    
    // Kernels and biases
    std::vector<std::vector<Eigen::MatrixXd>> kernels; // [depth][input_depth][kernel_size x kernel_size]
    std::vector<Eigen::MatrixXd> biases; // [depth][output_height x output_width]

private:
    // Helper methods for data conversion
    std::vector<Eigen::MatrixXd> convertToEigen(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> convertFromEigen(const std::vector<Eigen::MatrixXd>& input);
}; 