#include "network.hpp"
#include <iostream>

Network::Network(const std::vector<std::shared_ptr<Layer>>& layers) : layers(layers) {}

Eigen::MatrixXd Network::predict(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd output = input;
    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void Network::train(const std::vector<Eigen::MatrixXd>& x_train,
                   const std::vector<Eigen::MatrixXd>& y_train,
                   std::function<double(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> loss,
                   std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> loss_prime,
                   int epochs,
                   double learning_rate,
                   bool verbose) {
    for (int e = 0; e < epochs; e++) {
        double error = 0;
        
        for (size_t i = 0; i < x_train.size(); i++) {
            // Forward pass
            Eigen::MatrixXd output = predict(x_train[i]);
            
            // Calculate error
            error += loss(y_train[i], output);
            
            // Backward pass
            Eigen::MatrixXd grad = loss_prime(y_train[i], output);
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad = (*it)->backward(grad, learning_rate);
            }
        }
        
        error /= x_train.size();
        if (verbose) {
            std::cout << e + 1 << "/" << epochs << ", error=" << error << std::endl;
        }
    }
} 