#ifndef RESHAPE_HPP
#define RESHAPE_HPP

#include "layer.hpp"
#include <vector>
#include <Eigen/Dense>

class Reshape : public Layer {
public:
    Reshape(const std::vector<int>& input_shape, const std::vector<int>& output_shape);

    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) override;
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) override;

private:
    std::vector<int> input_shape;  // [input_depth, height, width]
    std::vector<int> output_shape; // [output_depth, new_height, new_width]

    int total_size(const std::vector<int>& shape);
};

#endif // RESHAPE_HPP
