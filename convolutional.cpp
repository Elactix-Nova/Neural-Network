#include "convolutional.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

Convolutional::Convolutional(const std::vector<int>& input_shape, int kernel_size, int depth)
    : depth(depth), kernel_size(kernel_size) {
    input_depth = input_shape[0];
    input_height = input_shape[1];
    input_width = input_shape[2];
    output_height = input_height - kernel_size + 1;
    output_width = input_width - kernel_size + 1;

    // Resize containers
    kernels.resize(depth, vector<MatrixXd>(input_depth, MatrixXd(kernel_size, kernel_size)));
    biases.resize(depth, MatrixXd(output_height, output_width));

    // Initialize kernels and biases with random values
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    // kernels[i][j](k, l) = dist(gen);
                    kernels[i][j](k, l) = 2.0;
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

std::vector<Eigen::MatrixXd> Convolutional::forward(const std::vector<Eigen::MatrixXd>& input) {
    // Store input for backward pass
    this->input = input;

    // Initialize output vector
    std::vector<Eigen::MatrixXd> output(depth, MatrixXd::Zero(output_height, output_width));

    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            for (int k = 0; k < output_height; ++k) {
                for (int l = 0; l < output_width; ++l) {
                    // Extract patch from input
                    MatrixXd patch = input[j].block(k, l, kernel_size, kernel_size);
                    // Element-wise multiplication and sum (dot product)
                    output[i](k, l) += (patch.array() * kernels[i][j].array()).sum();
                }
            }
        }
        output[i] += biases[i]; // Add biases
    }

    // Store output for backward pass
    this->output = output;
    return output;
}

std::vector<Eigen::MatrixXd> Convolutional::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    // Initialize gradients
    std::vector<std::vector<Eigen::MatrixXd>> kernels_gradient(depth, std::vector<Eigen::MatrixXd>(input_depth, MatrixXd::Zero(kernel_size, kernel_size)));
    std::vector<Eigen::MatrixXd> input_gradient(input_depth, MatrixXd::Zero(input_height, input_width));

    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    double grad = 0.0;
                    for (int m = 0; m < output_height; ++m) {
                        for (int n = 0; n < output_width; ++n) {
                            // For each output position, multiply the corresponding input patch
                            // with the output gradient at that position
                            grad += input[j](m + k, n + l) * output_gradient[i](m, n);
                        }
                    }
                    kernels_gradient[i][j](k, l) = grad;
                }
            }
        }
    }

    for (int j = 0; j < input_depth; ++j) {
            for (int p = 0; p < input_height; ++p) {
                for (int q = 0; q < input_width; ++q) {
                    double grad = 0.0;
                    for (int i = 0; i < depth; ++i) {
                        for (int m = 0; m < kernel_size; ++m) {
                            for (int n = 0; n < kernel_size; ++n) {
                                int out_row = p - m;
                                int out_col = q - n;
                                if (out_row >= 0 && out_row < output_height &&
                                    out_col >= 0 && out_col < output_width) {
                                    grad += kernels[i][j](m, n) * output_gradient[i](out_row, out_col);
                                }
                            }
                        }
                    }
                    input_gradient[j](p, q) = grad;
                }
            }
        }

    // Update kernels and biases
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            kernels[i][j] -= learning_rate * kernels_gradient[i][j];
        }
        biases[i] -= learning_rate * output_gradient[i];
    }

    // cout << "\n--- Kernel Gradients ---\n";
    // for (int i = 0; i < depth; ++i) {
    //     for (int j = 0; j < input_depth; ++j) {
    //         cout << "Kernel Gradient " << i + 1 << " for input channel " << j + 1 << ":\n" 
    //              << kernels_gradient[i][j] << "\n\n";
    //     }
    // }

    return input_gradient;
}

// int main() {
//     // Define input shape, kernel size, and depth (number of output channels)
//     vector<int> input_shape = {2, 5, 5}; // 2 channels, 5x5 input
//     int kernel_size = 2;
//     int depth = 2;

//     // Create convolutional layer instance
//     Convolutional conv(input_shape, kernel_size, depth);

//     // Print initial kernels
//     cout << "Initial Kernels:\n";
//     for (int i = 0; i < depth; ++i) {
//         for (int j = 0; j < input_shape[0]; ++j) {
//             cout << "Kernel " << i + 1 << " for input channel " << j + 1 << ":\n" 
//                  << conv.kernels[i][j] << "\n\n";
//         }
//     }

//     // Print initial biases
//     cout << "Initial Biases:\n";
//     for (int i = 0; i < depth; ++i) {
//         cout << "Bias " << i + 1 << ":\n" << conv.biases[i] << "\n\n";
//     }

//     // Create a sample input as a vector of matrices
//     std::vector<Eigen::MatrixXd> input(input_shape[0]);
//     for (int i = 0; i < input_shape[0]; ++i) {
//         input[i] = Eigen::MatrixXd::Constant(input_shape[1], input_shape[2], 1.0);
//     }

//     // Print input channels
//     cout << "Input Channels:\n";
//     for (int i = 0; i < input.size(); ++i) {
//         cout << "Channel " << i + 1 << ":\n" << input[i] << endl;
//     }

//     // Perform forward pass
//     std::vector<Eigen::MatrixXd> conv_output = conv.forward(input);
    
//     cout << "\nOutput of Convolutional Forward Pass:\n";
//     for (int i = 0; i < conv_output.size(); ++i) {
//         cout << "Output Channel " << i + 1 << ":\n" << conv_output[i] << endl;
//     }

//     int output_height = input_shape[1] - kernel_size + 1;
//     int output_width = input_shape[2] - kernel_size + 1;
//     // Create a dummy output gradient for testing backward pass
//     std::vector<Eigen::MatrixXd> output_gradient(depth);
//     for (int i = 0; i < depth; ++i) {
//         output_gradient[i] = Eigen::MatrixXd::Constant(output_height, output_width, 1.0);
//     }

//     // Print output gradient
//     cout << "\nOutput Gradient:\n";
//     for (int i = 0; i < depth; ++i) {
//         cout << "Gradient for output channel " << i + 1 << ":\n" << output_gradient[i] << "\n\n";
//     }

//     // Perform backward pass and capture the input gradient
//     std::vector<Eigen::MatrixXd> input_grad = conv.backward(output_gradient, 0.01);
    
//     cout << "\n--- Input Gradients ---\n";
//     for (int i = 0; i < input_grad.size(); ++i) {
//         cout << "Gradient Channel " << i + 1 << ":\n" << input_grad[i] << endl;
//     }

//     // Print bias gradients (which are the same as output gradients)
//     cout << "\n--- Bias Gradients ---\n";
//     for (int i = 0; i < depth; ++i) {
//         cout << "Bias Gradient " << i + 1 << ":\n" << output_gradient[i] << "\n\n";
//     }

//     return 0;
// }