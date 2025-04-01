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

    double cross_entropy_loss(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
        // Add small epsilon to avoid log(0)
        const double epsilon = 1e-15;
        Eigen::MatrixXd clipped_pred = y_pred.array().max(epsilon).min(1 - epsilon);
        return -(y_true.array() * clipped_pred.array().log()).sum() / y_true.rows();
    }

    Eigen::MatrixXd cross_entropy_loss_prime(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
        // Add small epsilon to avoid division by zero
        const double epsilon = 1e-15;
        Eigen::MatrixXd clipped_pred = y_pred.array().max(epsilon).min(1 - epsilon);
        return (clipped_pred - y_true) / y_true.rows();
    }
} 