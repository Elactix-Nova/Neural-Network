#pragma once
#include "layer.hpp"
#include "losses.hpp"
#include <vector>
#include <memory>
#include <functional>

class Network {
public:
    Network(const std::vector<std::shared_ptr<Layer>>& layers);
    
    std::vector<Eigen::MatrixXd> predict(const std::vector<Eigen::MatrixXd>& input);
    void train(const std::vector<std::vector<Eigen::MatrixXd>>& x_train, 
               const std::vector<std::vector<Eigen::MatrixXd>>& y_train,
               std::function<double(const std::vector<Eigen::MatrixXd>&, const std::vector<Eigen::MatrixXd>&)> loss,
               std::function<std::vector<Eigen::MatrixXd>(const std::vector<Eigen::MatrixXd>&, const std::vector<Eigen::MatrixXd>&)> loss_prime,
               int epochs = 1000,
               double learning_rate = 0.01,
               bool verbose = true);

private:
    std::vector<std::shared_ptr<Layer>> layers;
}; 