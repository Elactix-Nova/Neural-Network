#include "losses.hpp"
#include <cmath>

namespace Loss {
    double mse(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
        return (y_true - y_pred).array().square().sum() / y_true.rows();
    }

    Eigen::MatrixXd mse_prime(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
        return 2.0 * (y_pred - y_true) / y_true.rows();
    }

    double binary_cross_entropy(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
        return -(y_true.array() * y_pred.array().log() + 
                (1 - y_true.array()) * (1 - y_pred.array()).log()).sum() / y_true.rows();
    }

    Eigen::MatrixXd binary_cross_entropy_prime(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
        return ((1 - y_true.array()) / (1 - y_pred.array()) - 
                y_true.array() / y_pred.array()) / y_true.rows();
    }
} 