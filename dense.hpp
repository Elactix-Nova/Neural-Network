#pragma once
#include "layer.hpp"
#include <random>

class Dense : public Layer {
public:
    Dense(int input_size, int output_size);
    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) override;
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) override;

private:
    Eigen::MatrixXd weights;
    Eigen::MatrixXd bias;
    std::random_device rd;
    std::mt19937 gen;
};