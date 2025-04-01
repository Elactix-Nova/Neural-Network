#include "convolutional.hpp"
#include <cmath>
#include <iostream>


Convolutional::Convolutional(const std::vector<int>& input_shape, int kernel_size, int depth)
    : input_shape(input_shape),
      depth(depth),
      input_depth(input_shape[0]),
      gen(rd()) {
    
    // Calculate  output shape
    output_shape = {
        depth,
        input_shape[1] - kernel_size + 1,
        input_shape[2] - kernel_size + 1
    };

    // Calculate kernels shape
    kernels_shape = {
        depth,
        input_depth,
        kernel_size,
        kernel_size
    };

    // Initialize kernels and biases with random values
    std::normal_distribution<double> dist(0.0, 1.0);
    
    // Initialize kernels
    kernels.resize(depth);
    for (int i = 0; i < depth; i++) {
        kernels[i] = Eigen::MatrixXd::Zero(input_depth, kernel_size * kernel_size);
        for (int j = 0; j < input_depth; j++) {
            for (int k = 0; k < kernel_size * kernel_size; k++) {
                kernels[i](j, k) = dist(gen);
            }
        }
    }

    // Initialize biases
    biases.resize(depth);
    for (int i = 0; i < depth; i++) {
        biases[i] = Eigen::MatrixXd::Zero(output_shape[1], output_shape[2]);
        for (int j = 0; j < output_shape[1]; j++) {
            for (int k = 0; k < output_shape[2]; k++) {
                biases[i](j, k) = dist(gen);
            }
        }
    }
}

Eigen::MatrixXd Convolutional::correlate2d(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel) {
    int input_rows = input.rows();
    int input_cols = input.cols();
    int kernel_size = static_cast<int>(std::sqrt(kernel.cols()));
    
    Eigen::MatrixXd result(input_rows - kernel_size + 1, input_cols - kernel_size + 1);
    
    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); j++) {
            double sum = 0.0;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    sum += input(i + k, j + l) * kernel(k * kernel_size + l);
                }
            }
            result(i, j) = sum;
        }
    }
    
    return result;
}

Eigen::MatrixXd Convolutional::convolve2d(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel) {
    int input_rows = input.rows();
    int input_cols = input.cols();
    int kernel_size = static_cast<int>(std::sqrt(kernel.cols()));
    
    Eigen::MatrixXd result(input_rows + kernel_size - 1, input_cols + kernel_size - 1);
    
    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); j++) {
            double sum = 0.0;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    int input_i = i - k;
                    int input_j = j - l;
                    if (input_i >= 0 && input_i < input_rows && input_j >= 0 && input_j < input_cols) {
                        sum += input(input_i, input_j) * kernel(k * kernel_size + l);
                    }
                }
            }
            result(i, j) = sum;
        }
    }
    
    return result;
}

Eigen::MatrixXd Convolutional::forward(const Eigen::MatrixXd& input) {
    this->input = input;
    
    // Reshape input to 3D (depth, height, width)
    std::vector<Eigen::MatrixXd> input_3d(input_depth);
    for (int i = 0; i < input_depth; i++) {
        input_3d[i] = input.block(i * input_shape[1], 0, input_shape[1], input_shape[2]);
    }
    
    // Initialize output
    std::vector<Eigen::MatrixXd> output_3d(depth);
    for (int i = 0; i < depth; i++) {
        output_3d[i] = biases[i];
    }
    
    // Perform convolution
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < input_depth; j++) {
            output_3d[i] += correlate2d(input_3d[j], kernels[i].row(j));
        }
    }
    
    // Flatten output back to 2D
    Eigen::MatrixXd output(depth * output_shape[1], output_shape[2]);
    for (int i = 0; i < depth; i++) {
        output.block(i * output_shape[1], 0, output_shape[1], output_shape[2]) = output_3d[i];
    }
    
    return output;
}

Eigen::MatrixXd Convolutional::backward(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    // Reshape output gradient to 3D
    std::vector<Eigen::MatrixXd> output_gradient_3d(depth);
    for (int i = 0; i < depth; i++) {
        output_gradient_3d[i] = output_gradient.block(i * output_shape[1], 0, output_shape[1], output_shape[2]);
    }
    
    // Reshape input to 3D
    std::vector<Eigen::MatrixXd> input_3d(input_depth);
    for (int i = 0; i < input_depth; i++) {
        input_3d[i] = input.block(i * input_shape[1], 0, input_shape[1], input_shape[2]);
    }
    
    // Initialize gradients
    std::vector<Eigen::MatrixXd> kernels_gradient(depth);
    for (int i = 0; i < depth; i++) {
        kernels_gradient[i] = Eigen::MatrixXd::Zero(input_depth, kernels[0].cols());
    }
    
    std::vector<Eigen::MatrixXd> input_gradient_3d(input_depth);
    for (int i = 0; i < input_depth; i++) {
        input_gradient_3d[i] = Eigen::MatrixXd::Zero(input_shape[1], input_shape[2]);
    }
    
    // Compute gradients
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < input_depth; j++) {
            kernels_gradient[i].row(j) = correlate2d(input_3d[j], output_gradient_3d[i]).transpose();
            input_gradient_3d[j] += convolve2d(output_gradient_3d[i], kernels[i].row(j));
        }
    }
    
    // Update parameters
    for (int i = 0; i < depth; i++) {
        kernels[i] -= learning_rate * kernels_gradient[i];
        biases[i] -= learning_rate * output_gradient_3d[i];
    }
    
    // Flatten input gradient back to 2D
    Eigen::MatrixXd input_gradient(input_depth * input_shape[1], input_shape[2]);
    for (int i = 0; i < input_depth; i++) {
        input_gradient.block(i * input_shape[1], 0, input_shape[1], input_shape[2]) = input_gradient_3d[i];
    }
    
    return input_gradient;
} 

void Convolutional::print_kernels() {
    for (int i = 0; i < depth; i++) {
        std::cout << "Kernel " << i << ":\n" << kernels[i] << "\n";
    }
}

void Convolutional::print_biases() {
    for (int i = 0; i < depth; i++) {
        std::cout << "Bias " << i << ":\n" << biases[i] << "\n";
    }
}
