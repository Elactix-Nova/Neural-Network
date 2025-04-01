#include "reshape.hpp"
#include <numeric>

Reshape::Reshape(const std::vector<Eigen::Index>& new_shape) : new_shape(new_shape) {}

Eigen::MatrixXd Reshape::forward(const Eigen::MatrixXd& input) {
    this->input = input;
    
    // Store the old shape for backward pass
    old_shape = {input.rows(), input.cols()};
    
    // Calculate the total number of elements
    Eigen::Index total_elements = input.size();
    
    // Verify that the new shape has the same number of elements
    Eigen::Index new_total = std::accumulate(new_shape.begin(), new_shape.end(), 
                                           static_cast<Eigen::Index>(1), std::multiplies<Eigen::Index>());
    if (total_elements != new_total) {
        throw std::runtime_error("Reshape: Total number of elements must remain the same");
    }
    
    // Create a reshaped matrix
    Eigen::MatrixXd reshaped;
    if (new_shape.size() == 2) {
        // For 2D reshape, create a new matrix with the specified dimensions
        reshaped = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            input.data(), new_shape[0], new_shape[1]);
    } else {
        // For higher dimensions, we'll flatten to 2D
        Eigen::Index rows = new_shape[0];
        Eigen::Index cols = total_elements / rows;
        reshaped = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            input.data(), rows, cols);
    }
    
    return reshaped;
}

Eigen::MatrixXd Reshape::backward(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    // Reshape back to the original shape
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        output_gradient.data(), old_shape[0], old_shape[1]);
}