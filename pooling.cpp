#include "pooling.hpp"

// MaxPooling implementation
MaxPooling::MaxPooling(int kernel_size) : kernel_size(kernel_size) {}

Eigen::MatrixXd MaxPooling::forward(const Eigen::MatrixXd& input) {
    this->input = input;
    
    int output_rows = input.rows() / kernel_size;
    int output_cols = input.cols() / kernel_size;
    
    Eigen::MatrixXd output(output_rows, output_cols);
    max_indices = Eigen::MatrixXd::Zero(output_rows, output_cols);
    
    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            // Get the current window
            Eigen::MatrixXd window = input.block(i * kernel_size, j * kernel_size, kernel_size, kernel_size);
            
            // Find max value and its index
            double max_val = window.maxCoeff();
            Eigen::Index max_row, max_col;
            window.maxCoeff(&max_row, &max_col);
            
            // Store the max value and its relative index
            output(i, j) = max_val;
            max_indices(i, j) = max_row * kernel_size + max_col;
        }
    }
    
    return output;
}

Eigen::MatrixXd MaxPooling::backward(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    Eigen::MatrixXd input_gradient = Eigen::MatrixXd::Zero(input.rows(), input.cols());
    
    for (int i = 0; i < output_gradient.rows(); i++) {
        for (int j = 0; j < output_gradient.cols(); j++) {
            // Get the index of the max value
            int max_idx = static_cast<int>(max_indices(i, j));
            int max_row = max_idx / kernel_size;
            int max_col = max_idx % kernel_size;
            
            // Place the gradient at the position of the max value
            input_gradient(i * kernel_size + max_row, j * kernel_size + max_col) = output_gradient(i, j);
        }
    }
    
    return input_gradient;
}

// AveragePooling implementation
AveragePooling::AveragePooling(int kernel_size) : kernel_size(kernel_size) {}

Eigen::MatrixXd AveragePooling::forward(const Eigen::MatrixXd& input) {
    this->input = input;
    
    int output_rows = input.rows() / kernel_size;
    int output_cols = input.cols() / kernel_size;
    
    Eigen::MatrixXd output(output_rows, output_cols);
    
    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            // Get the current window and compute average
            Eigen::MatrixXd window = input.block(i * kernel_size, j * kernel_size, kernel_size, kernel_size);
            output(i, j) = window.mean();
        }
    }
    
    return output;
}

Eigen::MatrixXd AveragePooling::backward(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    Eigen::MatrixXd input_gradient = Eigen::MatrixXd::Zero(input.rows(), input.cols());
    
    for (int i = 0; i < output_gradient.rows(); i++) {
        for (int j = 0; j < output_gradient.cols(); j++) {
            // Distribute the gradient evenly across the window
            double grad_value = output_gradient(i, j) / (kernel_size * kernel_size);
            
            // Fill the corresponding window in the input gradient
            input_gradient.block(i * kernel_size, j * kernel_size, kernel_size, kernel_size)
                .setConstant(grad_value);
        }
    }
    
    return input_gradient;
} 