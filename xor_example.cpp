#include "network.hpp"
#include "dense.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include <iostream>
#include <memory>
#include <vector>

int main() {
    // Create training data for XOR
    std::vector<Eigen::MatrixXd> x_train = {
        Eigen::MatrixXd::Constant(2, 1, 0),  // [0, 0]
        Eigen::MatrixXd::Constant(2, 1, 0),  // [0, 0]
        Eigen::MatrixXd::Constant(2, 1, 1),  // [1, 1]
        Eigen::MatrixXd::Constant(2, 1, 1)   // [1, 1]
    };
    x_train[1](1, 0) = 1;  // [0, 1]
    x_train[2](1, 0) = 0;  // [1, 0]

    std::vector<Eigen::MatrixXd> y_train = {
        Eigen::MatrixXd::Constant(1, 1, 0),  // 0
        Eigen::MatrixXd::Constant(1, 1, 1),  // 1
        Eigen::MatrixXd::Constant(1, 1, 1),  // 1
        Eigen::MatrixXd::Constant(1, 1, 0)   // 0
    };

    // Create network layers
    std::vector<std::shared_ptr<Layer>> layers = {
        std::make_shared<Dense>(2, 3),
        std::make_shared<Tanh>(),
        std::make_shared<Dense>(3, 1),
        std::make_shared<Tanh>()
    };

    // Create and train network
    Network network(layers);
    network.train(x_train, y_train, Loss::mse, Loss::mse_prime, 10000, 0.1);

    // Test the network
    std::cout << "\nTesting the network:" << std::endl;
    for (const auto& x : x_train) {
        Eigen::MatrixXd prediction = network.predict(x);
        std::cout << "Input: [" << x(0, 0) << ", " << x(1, 0) 
                  << "], Output: " << prediction(0, 0) << std::endl;
    }

    return 0;
} 
