#include "activations.hpp"

// Tanh implementation
std::vector<Eigen::MatrixXd> Tanh::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    std::vector<Eigen::MatrixXd> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i].array().tanh();
    }
    return output;
}


std::vector<Eigen::MatrixXd> Tanh::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> result(output_gradient.size());
    for (size_t i = 0; i < output_gradient.size(); ++i) {
        Eigen::ArrayXXd tanh_val = input[i].array().tanh();
        result[i] = output_gradient[i].array() * (1 - tanh_val.square());
    }
    return result;
}


// Sigmoid implementation
std::vector<Eigen::MatrixXd> Sigmoid::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    std::vector<Eigen::MatrixXd> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = (1.0 / (1.0 + (-input[i].array()).exp())).matrix();
    }
    return output;
}


std::vector<Eigen::MatrixXd> Sigmoid::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> result(output_gradient.size());
    for (size_t i = 0; i < output_gradient.size(); ++i) {
        Eigen::ArrayXXd sigmoid = 1.0 / (1.0 + (-input[i].array()).exp());
        result[i] = (output_gradient[i].array() * sigmoid * (1 - sigmoid)).matrix();
    }
    return result;
}


// ReLU implementation
std::vector<Eigen::MatrixXd> ReLU::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    std::vector<Eigen::MatrixXd> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i].cwiseMax(0.0);
    }
    return output;
}


std::vector<Eigen::MatrixXd> ReLU::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> result(output_gradient.size());
    for (size_t i = 0; i < output_gradient.size(); ++i) {
        result[i] = (input[i].array() > 0).select(output_gradient[i].array(), 0.0);
    }
    return result;
}

// Softmax implementation
std::vector<Eigen::MatrixXd> Softmax::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    std::vector<Eigen::MatrixXd> output(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        Eigen::ArrayXXd exps = (input[i].array() - input[i].maxCoeff()).exp();
        output[i] = (exps / exps.sum()).matrix();
    }
    return output;
}

std::vector<Eigen::MatrixXd> Softmax::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> result(output_gradient.size());
    for (size_t i = 0; i < output_gradient.size(); ++i) {
        Eigen::ArrayXXd exps = (input[i].array() - input[i].maxCoeff()).exp();
        Eigen::ArrayXXd softmax = exps / exps.sum();

        Eigen::ArrayXXd grad_input = Eigen::ArrayXXd::Zero(input[i].rows(), input[i].cols());

        for (int j = 0; j < softmax.size(); ++j) {
            double s = softmax(j);
            double grad_sum = (output_gradient[i].array() * softmax).sum();
            grad_input(j) = s * (output_gradient[i](j) - grad_sum);
        }

        result[i] = grad_input.matrix();
    }
    return result;
}


