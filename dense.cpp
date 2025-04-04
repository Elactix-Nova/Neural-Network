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

std::vector<Eigen::MatrixXd> Dense::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    std::vector<Eigen::MatrixXd> output(1);
    output[0] = weights * input[0] + bias;
    return output;
}

std::vector<Eigen::MatrixXd> Dense::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    Eigen::MatrixXd weights_gradient = output_gradient[0] * input[0].transpose();
    Eigen::MatrixXd input_gradient = weights.transpose() * output_gradient[0];
    
    weights -= learning_rate * weights_gradient;
    bias -= learning_rate * output_gradient[0];
    
    std::vector<Eigen::MatrixXd> result(1);
    result[0] = input_gradient;
    return result;
} 