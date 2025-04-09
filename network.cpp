#include "network.hpp"
#include <iostream>

Network::Network(const std::vector<std::shared_ptr<Layer>>& layers, bool debug=false) : layers(layers), debug(debug) {}

// Function to print layer dimensions for debugging
void print_layer_dimensions(const std::vector<Eigen::MatrixXd>& data, const std::string& layer_name) {
    std::cout << layer_name << " dimensions: " << data.size() << " channels" << std::endl;
    if (!data.empty()) {
        std::cout << "First channel dimensions: " << data[0].rows() << "x" << data[0].cols() << std::endl;
    }
}

std::vector<Eigen::MatrixXd> Network::predict(const std::vector<Eigen::MatrixXd>& input) {
    std::vector<Eigen::MatrixXd> output = input;
    
    if (verbose){
        // Print input dimensions
        print_layer_dimensions(output, "Network input");
    }
    
    for (size_t i = 0; i < layers.size(); ++i) {
        output = layers[i]->forward(output);
        
        if (verbose){
            // Print output dimensions after each layer
            std::string layer_name = "After layer " + std::to_string(i);
            print_layer_dimensions(output, layer_name);
        }
    }
    return output;
}

void Network::train(const std::vector<std::vector<Eigen::MatrixXd>>& x_train,
                   const std::vector<std::vector<Eigen::MatrixXd>>& y_train,
                   std::function<double(const std::vector<Eigen::MatrixXd>&, const std::vector<Eigen::MatrixXd>&)> loss,
                   std::function<std::vector<Eigen::MatrixXd>(const std::vector<Eigen::MatrixXd>&, const std::vector<Eigen::MatrixXd>&)> loss_prime,
                   int epochs,
                   double learning_rate,
                   bool verbose) {
    for (int e = 0; e < epochs; e++) {
        double error = 0;
        
        for (size_t i = 0; i < x_train.size(); i++) {
            // Forward pass
            std::vector<Eigen::MatrixXd> output = predict(x_train[i]);
            
            // Calculate error
            error += loss(y_train[i], output);
            
            // Backward pass
            std::vector<Eigen::MatrixXd> grad = loss_prime(y_train[i], output);
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