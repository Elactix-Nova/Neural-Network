#include "dense.hpp"
#include <cmath>

Dense::Dense(int input_size, int output_size) 
    : gen(rd()) {
    // Initialize weights with random values
    std::normal_distribution<double> dist(0.0, 1.0);
    weights = Eigen::MatrixXd::Zero(output_size, input_size);
    bias = Eigen::MatrixXd::Zero(output_size, 1);
    
    for(int i = 0; i < output_size; i++) {
        for(int j = 0; j < input_size; j++) {
            weights(i, j) = dist(gen);
        }
        bias(i, 0) = dist(gen);
    }
}

Eigen::MatrixXd Dense::forward(const Eigen::MatrixXd& input) {
    this->input = input;
    return weights * input + bias;
}

Eigen::MatrixXd Dense::backward(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    Eigen::MatrixXd weights_gradient = output_gradient * input.transpose();
    Eigen::MatrixXd input_gradient = weights.transpose() * output_gradient;
    
    weights -= learning_rate * weights_gradient;
    bias -= learning_rate * output_gradient;
    
    return input_gradient;
} 