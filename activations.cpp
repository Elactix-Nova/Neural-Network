#include "activations.hpp"

// Tanh implementation
std::vector<Eigen::MatrixXd> Tanh::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    std::vector<Eigen::MatrixXd> output(1);
    output[0] = input[0].array().tanh();
    return output;
}

std::vector<Eigen::MatrixXd> Tanh::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> result(1);
    result[0] = output_gradient[0].array() * (1 - input[0].array().tanh().square());
    return result;
}

// Sigmoid implementation
std::vector<Eigen::MatrixXd> Sigmoid::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    std::vector<Eigen::MatrixXd> output(1);
    output[0] = 1.0 / (1.0 + (-input[0]).array().exp());
    return output;
}

std::vector<Eigen::MatrixXd> Sigmoid::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> result(1);
    Eigen::MatrixXd sigmoid = 1.0 / (1.0 + (-input[0]).array().exp());
    result[0] = output_gradient[0].array() * sigmoid.array() * (1 - sigmoid.array());
    return result;
}

// ReLU implementation
std::vector<Eigen::MatrixXd> ReLU::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    std::vector<Eigen::MatrixXd> output(1);
    output[0] = input[0].array().max(0);
    return output;
}

std::vector<Eigen::MatrixXd> ReLU::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> result(1);
    result[0] = output_gradient[0].array() * (input[0].array() >= 0).cast<double>();
    return result;
}

// Softmax implementation
std::vector<Eigen::MatrixXd> Softmax::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    std::vector<Eigen::MatrixXd> output(1);
    output[0] = input[0].array().exp() / input[0].array().exp().sum();
    return output;
}

std::vector<Eigen::MatrixXd> Softmax::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> result(1);
    result[0] = output_gradient[0].array() * (input[0].array().exp() / input[0].array().exp().sum());
    return result;
}


