#pragma once
#include <Eigen/Dense>

namespace Loss {
    double mse(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred);
    std::vector<Eigen::MatrixXd> mse_prime(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred);
    
    double binary_cross_entropy(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred);
    std::vector<Eigen::MatrixXd> binary_cross_entropy_prime(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred);

    double cross_entropy_loss(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred);
    std::vector<Eigen::MatrixXd> cross_entropy_loss_prime(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred);
} 