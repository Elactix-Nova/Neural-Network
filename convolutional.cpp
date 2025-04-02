#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include "convolutional.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

Convolutional::Convolutional(vector<int> input_shape, int kernel_size, int depth)
    : depth(depth), kernel_size(kernel_size) {
    input_depth = input_shape[0];
    input_height = input_shape[1];
    input_width = input_shape[2];
    output_height = input_height - kernel_size + 1;
    output_width = input_width - kernel_size + 1;
    
    // Resize containers
    kernels.resize(depth, vector<MatrixXd>(input_depth));
    biases.resize(depth);
    
    // Initialize kernels and biases with random values
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> d(0, 1);
    
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            kernels[i][j] = MatrixXd(kernel_size, kernel_size);
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    kernels[i][j](k, l) = d(gen);
                }
            }
        }
        
        biases[i] = MatrixXd(output_height, output_width);
        for (int k = 0; k < output_height; ++k) {
            for (int l = 0; l < output_width; ++l) {
                biases[i](k, l) = d(gen);
            }
        }
    }
}

// Helper method to convert vector<vector<vector<double>>> to vector<MatrixXd>
vector<MatrixXd> Convolutional::convertToEigen(const vector<vector<vector<double>>>& input) {
    vector<MatrixXd> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = MatrixXd(input[i].size(), input[i][0].size());
        for (size_t j = 0; j < input[i].size(); ++j) {
            for (size_t k = 0; k < input[i][j].size(); ++k) {
                result[i](j, k) = input[i][j][k];
            }
        }
    }
    return result;
}

// Helper method to convert vector<MatrixXd> back to vector<vector<vector<double>>>
vector<vector<vector<double>>> Convolutional::convertFromEigen(const vector<MatrixXd>& input) {
    vector<vector<vector<double>>> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i].resize(input[i].rows());
        for (int j = 0; j < input[i].rows(); ++j) {
            result[i][j].resize(input[i].cols());
            for (int k = 0; k < input[i].cols(); ++k) {
                result[i][j][k] = input[i](j, k);
            }
        }
    }
    return result;
}

// Forward pass using Eigen operations
vector<vector<vector<double>>> Convolutional::forward(const vector<vector<vector<double>>>& input) {
    // Convert input to Eigen format
    vector<MatrixXd> eigen_input = convertToEigen(input);
    vector<MatrixXd> eigen_output(depth);
    
    // Initialize output matrices with biases
    for (int i = 0; i < depth; ++i) {
        eigen_output[i] = biases[i];
    }
    
    // Perform convolution
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            // Perform convolution for each kernel and input channel
            for (int k = 0; k < output_height; ++k) {
                for (int l = 0; l < output_width; ++l) {
                    // Extract patch from input
                    MatrixXd patch = eigen_input[j].block(k, l, kernel_size, kernel_size);
                    // Element-wise multiplication and sum (dot product)
                    eigen_output[i](k, l) += (patch.array() * kernels[i][j].array()).sum();
                }
            }
        }
    }
    
    // Convert back to the original format
    return convertFromEigen(eigen_output);
}

// Backward pass using Eigen operations
vector<vector<vector<double>>> Convolutional::backward(const vector<vector<vector<double>>>& output_gradient,
                                        const vector<vector<vector<double>>>& input,
                                        double learning_rate) {
    // Convert to Eigen format
    vector<MatrixXd> eigen_output_grad = convertToEigen(output_gradient);
    vector<MatrixXd> eigen_input = convertToEigen(input);
    
    // Initialize gradients
    vector<vector<MatrixXd>> kernels_gradient(depth, vector<MatrixXd>(input_depth));
    vector<MatrixXd> input_gradient(input_depth);
    
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            kernels_gradient[i][j] = MatrixXd::Zero(kernel_size, kernel_size);
        }
    }
    
    for (int j = 0; j < input_depth; ++j) {
        input_gradient[j] = MatrixXd::Zero(input_height, input_width);
    }
    
    // Compute kernels gradient
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < input_depth; ++j) {
            for (int m = 0; m < kernel_size; ++m) {
                for (int n = 0; n < kernel_size; ++n) {
                    double grad = 0.0;
                    for (int k = 0; k < output_height; ++k) {
                        for (int l = 0; l < output_width; ++l) {
                            grad += eigen_input[j](k + m, l + n) * eigen_output_grad[i](k, l);
                        }
                    }
                    kernels_gradient[i][j](m, n) = grad;
                }
            }
        }
    }
    
    // Compute input gradient
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
                                grad += kernels[i][j](m, n) * eigen_output_grad[i](out_row, out_col);
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
        biases[i] -= learning_rate * eigen_output_grad[i];
    }
    
    return convertFromEigen(input_gradient);
}

// int main() {
//     // Define input shape, kernel size, and depth (number of output channels)
//     vector<int> input_shape = {2, 5, 5}; 
//     int kernel_size = 2;
//     int depth = 2;
    
//     // Create convolutional layer instance
//     Convolutional conv(input_shape, kernel_size, depth);
    
//     // Create a sample input filled with ones
//     vector<vector<vector<double>>> input(2, vector<vector<double>>(5, vector<double>(5, 1.0)));
    
//     // Print input image channels
//     cout << "Input Image Channels:" << endl;
//     for (int i = 0; i < input.size(); ++i) {
//         cout << "Channel " << i + 1 << ":\n";
//         for (const auto& row : input[i]) {
//             for (double val : row) {
//                 cout << val << " ";
//             }
//             cout << endl;
//         }
//         cout << endl;
//     }
    
//     // Print initial kernels
//     cout << "Initial Kernels:" << endl;
//     for (int i = 0; i < conv.kernels.size(); ++i) {
//         cout << "Kernel " << i + 1 << ":\n";
//         for (int j = 0; j < conv.kernels[i].size(); ++j) {
//             cout << "Channel " << j + 1 << ":\n";
//             cout << conv.kernels[i][j] << endl;
//         }
//         cout << endl;
//     }
    
//     // Print initial biases
//     cout << "Initial Biases:" << endl;
//     for (int i = 0; i < conv.biases.size(); ++i) {
//         cout << "Bias " << i + 1 << ":\n";
//         cout << conv.biases[i] << endl;
//     }
    
//     // Perform forward pass
//     vector<vector<vector<double>>> conv_output = conv.forward(input);
    
//     cout << "\nOutput of Convolutional Forward Pass:" << endl;
//     for (int i = 0; i < conv_output.size(); ++i) {
//         cout << "Output Channel " << i + 1 << ":\n";
//         for (const auto& row : conv_output[i]) {
//             for (double val : row) {
//                 cout << val << " ";
//             }
//             cout << endl;
//         }
//         cout << endl;
//     }
    
//     // Create a dummy output gradient (same shape as conv_output) filled with ones for testing backward pass
//     vector<vector<vector<double>>> output_gradient(depth, vector<vector<double>>(conv.output_height, vector<double>(conv.output_width, 1.0)));
    
//     double learning_rate = 0.01;
    
//     // Perform backward pass and capture the input gradient
//     vector<vector<vector<double>>> input_grad = conv.backward(output_gradient, input, learning_rate);
    
//     cout << "\n--- Input Gradients ---\n";
//     for (int i = 0; i < input_grad.size(); ++i) {
//         cout << "Input Gradient Channel " << i + 1 << ":\n";
//         for (const auto& row : input_grad[i]) {
//             for (double val : row) {
//                 cout << val << " ";
//             }
//             cout << endl;
//         }
//         cout << endl;
//     }
    
//     return 0;
// }