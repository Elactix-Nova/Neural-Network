#include "pooling.hpp"

// MaxPooling implementation
MaxPooling::MaxPooling(int kernel_size, int stride) 
    : kernel_size(kernel_size), 
      stride(stride == -1 ? kernel_size : stride) {}

std::vector<Eigen::MatrixXd> MaxPooling::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    
    int output_rows = (input[0].rows() - kernel_size) / stride + 1;
    int output_cols = (input[0].cols() - kernel_size) / stride + 1;
    
    std::vector<Eigen::MatrixXd> output(1);
    output[0] = Eigen::MatrixXd(output_rows, output_cols);
    max_indices = Eigen::MatrixXd::Zero(output_rows, output_cols);
    
    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            // Get the current window
            Eigen::MatrixXd window = input[0].block(i * stride, j * stride, kernel_size, kernel_size);
            
            // Find max value and its index
            double max_val = window.maxCoeff();
            Eigen::Index max_row, max_col;
            window.maxCoeff(&max_row, &max_col);
            
            // Store the max value and its relative index
            output[0](i, j) = max_val;
            max_indices(i, j) = max_row * kernel_size + max_col;
        }
    }
    
    return output;
}

std::vector<Eigen::MatrixXd> MaxPooling::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> input_gradient(1);
    input_gradient[0] = Eigen::MatrixXd::Zero(input[0].rows(), input[0].cols());
    
    for (int i = 0; i < output_gradient[0].rows(); i++) {
        for (int j = 0; j < output_gradient[0].cols(); j++) {
            // Get the index of the max value
            int max_idx = static_cast<int>(max_indices(i, j));
            int max_row = max_idx / kernel_size;
            int max_col = max_idx % kernel_size;
            
            // Place the gradient at the position of the max value
            input_gradient[0](i * stride + max_row, j * stride + max_col) = output_gradient[0](i, j);
        }
    }
    
    return input_gradient;
}

// AveragePooling implementation
AveragePooling::AveragePooling(int kernel_size, int stride) 
    : kernel_size(kernel_size), 
      stride(stride == -1 ? kernel_size : stride) {}

std::vector<Eigen::MatrixXd> AveragePooling::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    
    int output_rows = (input[0].rows() - kernel_size) / stride + 1;
    int output_cols = (input[0].cols() - kernel_size) / stride + 1;
    
    std::vector<Eigen::MatrixXd> output(1);
    output[0] = Eigen::MatrixXd(output_rows, output_cols);
    
    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            // Get the current window and compute average
            Eigen::MatrixXd window = input[0].block(i * stride, j * stride, kernel_size, kernel_size);
            output[0](i, j) = window.mean();
        }
    }
    
    return output;
}

std::vector<Eigen::MatrixXd> AveragePooling::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> input_gradient(1);
    input_gradient[0] = Eigen::MatrixXd::Zero(input[0].rows(), input[0].cols());
    
    for (int i = 0; i < output_gradient[0].rows(); i++) {
        for (int j = 0; j < output_gradient[0].cols(); j++) {
            // Distribute the gradient evenly across the window
            double grad_value = output_gradient[0](i, j) / (kernel_size * kernel_size);
            
            // Fill the corresponding window in the input gradient
            input_gradient[0].block(i * stride, j * stride, kernel_size, kernel_size)
                .setConstant(grad_value);
        }
    }
    
    return input_gradient;
} 