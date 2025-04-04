#pragma once
#include <Eigen/Dense>
#include <vector>

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) = 0;
    virtual std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) = 0;
    
protected:
    std::vector<Eigen::MatrixXd> input;
    std::vector<Eigen::MatrixXd> output;
}; 