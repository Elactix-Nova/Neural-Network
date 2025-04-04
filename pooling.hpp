#ifndef POOLING_HPP
#define POOLING_HPP

#include "layer.hpp"
#include <Eigen/Dense>
#include <vector>

class MaxPooling : public Layer {
public:
    MaxPooling(int kernel_size, int stride = -1);
    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) override;
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) override;

private:
    int kernel_size, stride;
    std::vector<Eigen::MatrixXi> max_row_indices;
    std::vector<Eigen::MatrixXi> max_col_indices;
};

class AveragePooling : public Layer {
public:
    AveragePooling(int kernel_size, int stride = -1);
    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) override;
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) override;

private:
    int kernel_size, stride;
};

#endif // POOLING_HPP
