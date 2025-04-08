#ifndef POOLING_HPP
#define POOLING_HPP

#include "layer.hpp"
#include <Eigen/Dense>
#include <vector>

class MaxPooling : public Layer {
public:
    // Stride is set to kernel size when not explicitly mentioned
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


// Going to document this layer way better. 
// NOTE: Once again, padding and stride not relevant for this layer
/**
 * @brief Global Average Pooling Layer
 * 
 * This layer performs global average pooling on the input feature maps.
 * It reduces each feature map to a single value by taking the average of all elements.
 * This is commonly used in CNNs to reduce spatial dimensions while maintaining channel information.
 * e.g. (Let's say you have 128 feature maps of 5x5 each)
 This would convert (128,5,5) -> (128,1x1)
 Basically, takes a mean of entire feature map for all channels/feature maps
 */
class GlobalAvgPooling : public Layer {
public:
    /**
     * @brief Construct a new Global Average Pooling layer
     * @param kernel_size Size of the pooling window (not used in global pooling, kept for interface consistency)
     * @param stride Stride of the pooling operation (not used in global pooling, kept for interface consistency)
     */
    GlobalAvgPooling(int kernel_size = 1, int stride = 1);

    /**
     * @brief Forward pass of global average pooling
     * @param input Input feature maps [channels][height][width]
     * @return Output feature maps reduced to [channels][1][1]
     */
    std::vector<Eigen::MatrixXd> forward(const std::vector<Eigen::MatrixXd>& input) override;

    /**
     * @brief Backward pass of global average pooling
     * @param output_gradient Gradient from the next layer
     * @param learning_rate Learning rate for parameter updates (not used in pooling)
     * @return Gradient with respect to input
     */
    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) override;

private:
    int kernel_size;  // Not used in global pooling, kept for interface consistency
    int stride;       // Not used in global pooling, kept for interface consistency
    std::vector<int> input_shape;  // Store input shape for backward pass: (num_filters, )
};

#endif // POOLING_HPP
