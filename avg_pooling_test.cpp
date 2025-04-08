#include "pooling.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>

int main() {
    // Create a sample input (3 channels, 4x4)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<Eigen::MatrixXd> input = {
        Eigen::MatrixXd::Zero(4, 4),
        Eigen::MatrixXd::Zero(4, 4),
        Eigen::MatrixXd::Zero(4, 4)
    };

    // Fill matrices with random values
    for (size_t i = 0; i < input.size(); ++i) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                input[i](row, col) = dis(gen) * (i*i+1);
            }
        }
    }

    // Create GlobalAvgPooling layer
    auto pooling_layer = std::make_shared<GlobalAvgPooling>(1,1);

    // Print input
    std::cout << "Input shape: " << input.size() << " channels, " 
              << input[0].rows() << "x" << input[0].cols() << std::endl;
    std::cout << "Input values:\n";
    for (size_t i = 0; i < input.size(); ++i) {
        std::cout << "Channel " << i << ":\n" << input[i] << "\n\n";
    }

    // Forward pass
    auto output = pooling_layer->forward(input);

    // Print output
    std::cout << "Output shape: " << output.size() << " channels, "
              << output[0].rows() << "x" << output[0].cols() << std::endl;
    std::cout << "Output values:\n";
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << "Channel " << i << ":\n" << output[i] << "\n\n";
    }

    // Expected output should be 1x1 matrices with values 1, 2, and 3 respectively
    // since we're averaging 4x4 matrices of constant values

    return 0;
}
