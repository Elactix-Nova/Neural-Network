#include "network.hpp"
#include "dense.hpp"
#include "convolutional.hpp"
#include "reshape.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>

// Function to preprocess MNIST data
std::pair<std::vector<std::vector<Eigen::MatrixXd>>, std::vector<std::vector<Eigen::MatrixXd>>> 
preprocess_data(const std::vector<std::vector<double>>& x, 
               const std::vector<int>& y, 
               int limit) {
    std::vector<std::vector<Eigen::MatrixXd>> processed_x;
    std::vector<std::vector<Eigen::MatrixXd>> processed_y;
    
    // Find indices for classes 0 and 1
    std::vector<size_t> zero_indices, one_indices;
    for (size_t i = 0; i < y.size(); ++i) {
        if (y[i] == 0 && zero_indices.size() < limit) {
            zero_indices.push_back(i);
        } else if (y[i] == 1 && one_indices.size() < limit) {
            one_indices.push_back(i);
        }
    }
    
    // Combine and shuffle indices
    std::vector<size_t> all_indices = zero_indices;
    all_indices.insert(all_indices.end(), one_indices.begin(), one_indices.end());
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_indices.begin(), all_indices.end(), g);
    
    // Process data
    for (size_t idx : all_indices) {
        // Process input (28x28 image)
        Eigen::MatrixXd img(28, 28);
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                img(i, j) = x[idx][i * 28 + j] / 255.0;
            }
        }
        processed_x.push_back({img});
        
        // Process output (one-hot encoded)
        Eigen::MatrixXd label(2, 1);
        label(0, 0) = (y[idx] == 0) ? 1.0 : 0.0;
        label(1, 0) = (y[idx] == 1) ? 1.0 : 0.0;
        processed_y.push_back({label});
    }
    
    return {processed_x, processed_y};
}

int main() {
    // Load MNIST data (in a real implementation, you'd load from files)
    // For now, we'll assume the data is loaded into x_train, y_train, x_test, y_test
    
    // Preprocess data
    auto [x_train, y_train] = preprocess_data(/* MNIST training data */, /* MNIST training labels */, 100);
    auto [x_test, y_test] = preprocess_data(/* MNIST test data */, /* MNIST test labels */, 100);
    
    // Create network layers
    std::vector<std::shared_ptr<Layer>> layers = {
        std::make_shared<Convolutional>(std::vector<int>{1, 28, 28}, 3, 5),  // Input shape: [channels, height, width], kernel: 3x3, output channels: 5
        std::make_shared<Sigmoid>(),
        std::make_shared<Reshape>(std::vector<int>{5, 26, 26}, std::vector<int>{5 * 26 * 26, 1}),
        std::make_shared<Dense>(5 * 26 * 26, 100),
        std::make_shared<Sigmoid>(),
        std::make_shared<Dense>(100, 2),
        std::make_shared<Sigmoid>()
    };
    
    // Create network
    Network network(layers);
    
    // Train network
    network.train(x_train, y_train, 
                 Loss::binary_cross_entropy, 
                 Loss::binary_cross_entropy_prime,
                 20,  // epochs
                 0.1, // learning rate
                 true); // verbose
    
    // Test network
    int correct = 0;
    for (size_t i = 0; i < x_test.size(); ++i) {
        auto output = network.predict(x_test[i]);
        int pred = (output[0](0, 0) > output[0](1, 0)) ? 0 : 1;
        int true_label = (y_test[i][0](0, 0) > y_test[i][0](1, 0)) ? 0 : 1;
        
        std::cout << "pred: " << pred << ", true: " << true_label << std::endl;
        if (pred == true_label) {
            correct++;
        }
    }
    
    std::cout << "Accuracy: " << (double)correct / x_test.size() * 100 << "%" << std::endl;
    
    return 0;
} 