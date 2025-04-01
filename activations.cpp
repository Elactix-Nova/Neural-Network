#include "activations.hpp"

// Tanh implementation
Eigen::MatrixXd Tanh::forward(const Eigen::MatrixXd& input) {
    this->input = input;
    return input.array().tanh();
}

Eigen::MatrixXd Tanh::backward(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    return output_gradient.array() * (1 - input.array().tanh().square());
}

// Sigmoid implementation
Eigen::MatrixXd Sigmoid::forward(const Eigen::MatrixXd& input) {
    this->input = input;
    return 1.0 / (1.0 + (-input).array().exp());
}

Eigen::MatrixXd Sigmoid::backward(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    Eigen::MatrixXd sigmoid = 1.0 / (1.0 + (-input).array().exp());
    return output_gradient.array() * sigmoid.array() * (1 - sigmoid.array());
} 

Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& input) {
    this->input = input;
    return input.array().max(0);
}

Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    return output_gradient.array() * (input.array() >= 0).cast<double>();
}

Eigen::MatrixXd Softmax::forward(const Eigen::MatrixXd& input) {
    this->input = input;
    return input.array().exp() / input.array().exp().sum();
}

Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    return output_gradient.array() * (input.array().exp() / input.array().exp().sum());
}


