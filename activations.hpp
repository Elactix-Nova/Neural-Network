#pragma once
#include "layer.hpp"
#include <cmath>

class Tanh : public Layer {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& output_gradient, double learning_rate) override;
};

class Sigmoid : public Layer {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& output_gradient, double learning_rate) override;
}; 

class ReLU : public Layer {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& output_gradient, double learning_rate) override;
}; 

class Softmax : public Layer {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& output_gradient, double learning_rate) override;
}; 