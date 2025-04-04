#include "reshape.hpp"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

Reshape::Reshape(const std::vector<int>& input_shape, const std::vector<int>& output_shape)
    : input_shape(input_shape), output_shape(output_shape) {
    if (total_size(input_shape) != total_size(output_shape)) {
        throw std::invalid_argument("Total elements in input and output shapes must be the same.");
    }
}

int Reshape::total_size(const std::vector<int>& shape) {
    return shape[0] * shape[1] * shape[2];
}

std::vector<Eigen::MatrixXd> Reshape::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;

    // Flatten input matrices in row-major order
    Eigen::VectorXd flattened(total_size(input_shape));
    int idx = 0;
    for (const auto& mat : input) {
        for (int i = 0; i < mat.rows(); ++i)
            for (int j = 0; j < mat.cols(); ++j)
                flattened(idx++) = mat(i, j);
    }

    // Reshape flattened vector to output shape in row-major order
    std::vector<Eigen::MatrixXd> output(output_shape[0]);
    idx = 0;
    for (int ch = 0; ch < output_shape[0]; ++ch) {
        output[ch] = Eigen::MatrixXd(output_shape[1], output_shape[2]);
        for (int i = 0; i < output_shape[1]; ++i)
            for (int j = 0; j < output_shape[2]; ++j)
                output[ch](i, j) = flattened(idx++);
    }

    return output;
}

std::vector<Eigen::MatrixXd> Reshape::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    // Flatten output gradients in row-major order
    Eigen::VectorXd flattened(total_size(output_shape));
    int idx = 0;
    for (const auto& mat : output_gradient) {
        for (int i = 0; i < mat.rows(); ++i)
            for (int j = 0; j < mat.cols(); ++j)
                flattened(idx++) = mat(i, j);
    }

    // Reshape flattened vector back to input shape in row-major order
    std::vector<Eigen::MatrixXd> input_gradient(input_shape[0]);
    idx = 0;
    for (int ch = 0; ch < input_shape[0]; ++ch) {
        input_gradient[ch] = Eigen::MatrixXd(input_shape[1], input_shape[2]);
        for (int i = 0; i < input_shape[1]; ++i)
            for (int j = 0; j < input_shape[2]; ++j)
                input_gradient[ch](i, j) = flattened(idx++);
    }

    return input_gradient;
}