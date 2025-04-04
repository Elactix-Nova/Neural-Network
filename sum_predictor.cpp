#include "network.hpp"
#include "dense.hpp"
#include "convolutional.hpp"
#include "reshape.hpp"
#include "activations.hpp"
#include "pooling.hpp"
#include "losses.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>

// Function to generate random 2x2 binary matrices and their sums
std::pair<std::vector<std::vector<Eigen::MatrixXd>>, std::vector<std::vector<Eigen::MatrixXd>>> 
generate_data(int num_samples) {
    std::vector<std::vector<Eigen::MatrixXd>> inputs;
    std::vector<std::vector<Eigen::MatrixXd>> targets;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    
    for (int i = 0; i < num_samples; ++i) {
        // Generate random 2x2 binary matrix
        Eigen::MatrixXd input(3, 3);
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                input(j, k) = dis(gen);
            }
        }
        
        // Calculate sum
        double sum = input.sum();
        
        // Create target matrix (1x1 with the sum)
        Eigen::MatrixXd target(1, 1);
        target(0, 0) = sum;
        
        inputs.push_back({input});
        targets.push_back({target});
    }
    
    return {inputs, targets};
}

// Mean Squared Error loss function
double mse(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred) {
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        loss += (y_true[i] - y_pred[i]).array().square().sum();
    }
    return loss / y_true.size();
}

// MSE derivative
std::vector<Eigen::MatrixXd> mse_prime(const std::vector<Eigen::MatrixXd>& y_true, 
                                     const std::vector<Eigen::MatrixXd>& y_pred) {
    std::vector<Eigen::MatrixXd> grad(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        grad[i] = 2.0 * (y_pred[i] - y_true[i]) / y_true.size();
    }
    return grad;
}

int main() {
    // Generate training and test data
    auto [x_train, y_train] = generate_data(1000);
    auto [x_test, y_test] = generate_data(100);
    
    // Create network layers
    std::vector<std::shared_ptr<Layer>> layers = {
        std::make_shared<Convolutional>(std::vector<int>{1, 3, 3}, 2, 4),
        // std::make_shared<ReLU>(),
        std::make_shared<AveragePooling>(1, 1),  // Input: 1x2x2, kernel: 2x2, output channels: 4
        // std::make_shared<Reshape>(std::vector<int>{4,2,2}, std::vector<int>{1,16,1}),  // Reshape to 4x1
        // std::make_shared<Dense>(16, 1),  // Dense layer to output single value
        std::make_shared<Reshape>(std::vector<int>{4,2,2}, std::vector<int>{1,16,1}),  // Reshape to 4x1
        std::make_shared<Dense>(16, 1),  // Dense layer to output single value
        std::make_shared<ReLU>()
    };

    // Create network
    Network network(layers);
    
    // Train network
    network.train(x_train, y_train, 
                 mse, 
                 mse_prime,
                 100,  // epochs
                 0.01, // learning rate
                 true); // verbose
    
    // Test network
    double total_error = 0.0;
    for (size_t i = 0; i < x_test.size(); ++i) {
        auto output = network.predict(x_test[i]);
        double predicted_sum = output[0](0, 0);
        double true_sum = y_test[i][0](0, 0);
        
        std::cout << "Input matrix:\n" << x_test[i][0] << "\n";
        std::cout << "Predicted sum: " << predicted_sum << ", True sum: " << true_sum << "\n";
        std::cout << "Error: " << std::abs(predicted_sum - true_sum) << "\n\n";
        
        total_error += std::abs(predicted_sum - true_sum);
    }
    
    std::cout << "Average absolute error: " << total_error / x_test.size() << std::endl;
    
    return 0;
} 