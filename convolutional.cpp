#include "convolutional.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

Convolutional::Convolutional(const std::vector<int>& input_shape, int kernel_size, int depth, 
                          int stride, int padding)
    : depth(depth), kernel_size(kernel_size), stride(stride), padding(padding), gen(rd()) {
    input_depth = input_shape[0];
    input_height = input_shape[1];
    input_width = input_shape[2];
    
    // Calculate output dimensions with stride and padding
    output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    // Resize containers
    kernels.resize(depth, vector<MatrixXd>(input_depth, MatrixXd(kernel_size, kernel_size)));
    biases.resize(depth, MatrixXd(output_height, output_width));

    // Initialize kernels and biases with random values
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    kernels[i][j](k, l) = dist(gen);
                }
            }
        }
        for (int k = 0; k < output_height; ++k) {
            for (int l = 0; l < output_width; ++l) {
                biases[i](k, l) = dist(gen);
            }
        }
    }
}

Eigen::MatrixXd Convolutional::padInput(const Eigen::MatrixXd& input) const {
    if (padding == 0) {
        return input;
    }
    
    int padded_height = input.rows() + 2 * padding;
    int padded_width = input.cols() + 2 * padding;
    Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(padded_height, padded_width);
    
    // Copy the input to the center of the padded matrix
    padded.block(padding, padding, input.rows(), input.cols()) = input;
    
    return padded;
}

std::vector<Eigen::MatrixXd> Convolutional::forward(const std::vector<Eigen::MatrixXd>& input) {
    // Store input for backward pass
    this->input = input;

    // Initialize output vector
    std::vector<Eigen::MatrixXd> output(depth, MatrixXd::Zero(output_height, output_width));

    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            // Apply padding to input
            MatrixXd padded_input = padInput(input[j]);
            
            for (int k = 0; k < output_height; ++k) {
                for (int l = 0; l < output_width; ++l) {
                    // Extract patch from padded input based on stride
                    MatrixXd patch = padded_input.block(k * stride, l * stride, kernel_size, kernel_size);
                    // Element-wise multiplication and sum (dot product)
                    output[i](k, l) += (patch.array() * kernels[i][j].array()).sum();
                }
            }
        }
        output[i] += biases[i]; // Add biases
    }

    // Store output for backward pass
    this->output = output;
    // std::cout << "Channels " << output.size() << " Height " << output[0].rows() << " Width " << output[0].cols() << std::endl;
    return output;
}

std::vector<Eigen::MatrixXd> Convolutional::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    // Initialize gradients
    std::vector<std::vector<Eigen::MatrixXd>> kernels_gradient(depth, std::vector<Eigen::MatrixXd>(input_depth, MatrixXd::Zero(kernel_size, kernel_size)));
    std::vector<Eigen::MatrixXd> input_gradient(input_depth, MatrixXd::Zero(input_height, input_width));

    // Calculate gradients for kernels
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            // Apply padding to input
            MatrixXd padded_input = padInput(input[j]);
            
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    double grad = 0.0;
                    for (int m = 0; m < output_height; ++m) {
                        for (int n = 0; n < output_width; ++n) {
                            // For each output position, multiply the corresponding input patch
                            // with the output gradient at that position
                            int input_row = m * stride + k;
                            int input_col = n * stride + l;
                            if (input_row >= 0 && input_row < padded_input.rows() &&
                                input_col >= 0 && input_col < padded_input.cols()) {
                                grad += padded_input(input_row, input_col) * output_gradient[i](m, n);
                            }
                        }
                    }
                    kernels_gradient[i][j](k, l) = grad;
                }
            }
        }
    }

    // Calculate gradients for inputs
    for (int j = 0; j < input_depth; ++j) {
        // Create padded input gradient
        MatrixXd padded_input_gradient = MatrixXd::Zero(input_height + 2 * padding, input_width + 2 * padding);
        
        for (int i = 0; i < depth; ++i) {
            for (int m = 0; m < output_height; ++m) {
                for (int n = 0; n < output_width; ++n) {
                    // For each position in the output gradient
                    for (int k = 0; k < kernel_size; ++k) {
                        for (int l = 0; l < kernel_size; ++l) {
                            int input_row = m * stride + k;
                            int input_col = n * stride + l;
                            padded_input_gradient(input_row, input_col) += 
                                kernels[i][j](k, l) * output_gradient[i](m, n);
                        }
                    }
                }
            }
        }
        
        // Extract the actual input gradient from the padded version
        if (padding > 0) {
            input_gradient[j] = padded_input_gradient.block(padding, padding, input_height, input_width);
        } else {
            input_gradient[j] = padded_input_gradient;
        }
    }

    // Update kernels and biases
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            kernels[i][j] -= learning_rate * kernels_gradient[i][j];
        }
        biases[i] -= learning_rate * output_gradient[i];
    }

    // std::cout << "Channels " << input_gradient.size() << " Height " << input_gradient[0].rows() << " Width " << input_gradient[0].cols() << std::endl;
    return input_gradient;
}