#include "reshape.hpp"
#include <numeric>

Reshape::Reshape(const std::vector<int>& new_shape) : new_shape(new_shape) {}

Eigen::MatrixXd Reshape::forward(const Eigen::MatrixXd& input) {
    this->input = input;
    
    // Store the old shape for backward pass
    old_shape = {input.rows(), input.cols()};
    
    // Calculate the total number of elements
    int total_elements = input.size();
    
    // Verify that the new shape has the same number of elements
    int new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (total_elements != new_total) {
        throw std::runtime_error("Reshape: Total number of elements must remain the same");
    }
    
    // Reshape the input matrix
    Eigen::MatrixXd reshaped;
    if (new_shape.size() == 2) {
        reshaped = input.reshape(new_shape[0], new_shape[1]);
    } else {
        // For higher dimensions, we'll flatten to 2D
        int rows = new_shape[0];
        int cols = total_elements / rows;
        reshaped = input.reshape(rows, cols);
    }
    
    return reshaped;
}

Eigen::MatrixXd Reshape::backward(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    // Simply reshape back to the original shape
    return output_gradient.reshape(old_shape[0], old_shape[1]);
} 