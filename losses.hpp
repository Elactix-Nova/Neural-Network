#pragma once
#include <Eigen/Dense>

namespace Loss {
    double mse(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred);
    Eigen::MatrixXd mse_prime(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred);
    
    double binary_cross_entropy(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred);
    Eigen::MatrixXd binary_cross_entropy_prime(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred);
} 