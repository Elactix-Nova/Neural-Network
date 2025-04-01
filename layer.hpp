#pragma once
#include <Eigen/Dense>

class Layer {
public:
    virtual ~Layer() = default;
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& output_gradient, double learning_rate) = 0;
    
protected:
    Eigen::MatrixXd input;
    Eigen::MatrixXd output;
}; 