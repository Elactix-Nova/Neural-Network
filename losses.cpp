#include "losses.hpp"
#include <cmath>

namespace Loss {
    double mse(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
        return (y_true - y_pred).array().square().sum() / y_true.rows();
    }

    Eigen::MatrixXd mse_prime(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) {
        return 2.0 * (y_pred - y_true) / y_true.rows();
    }

    double binary_cross_entropy(const std::vector<Eigen::MatrixXd>& y_true, 
                              const std::vector<Eigen::MatrixXd>& y_pred) {
        double loss = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            for (int j = 0; j < y_true[i].rows(); ++j) {
                for (int k = 0; k < y_true[i].cols(); ++k) {
                    double y = y_true[i](j, k);
                    double p = y_pred[i](j, k);
                    // Add small epsilon to avoid log(0)
                    loss += -(y * std::log(p + 1e-15) + (1 - y) * std::log(1 - p + 1e-15));
                }
            }
        }
        return loss / y_true.size();
    }

    std::vector<Eigen::MatrixXd> binary_cross_entropy_prime(const std::vector<Eigen::MatrixXd>& y_true, 
                                                          const std::vector<Eigen::MatrixXd>& y_pred) {
        std::vector<Eigen::MatrixXd> grad(y_true.size());
        for (size_t i = 0; i < y_true.size(); ++i) {
            grad[i] = Eigen::MatrixXd::Zero(y_true[i].rows(), y_true[i].cols());
            for (int j = 0; j < y_true[i].rows(); ++j) {
                for (int k = 0; k < y_true[i].cols(); ++k) {
                    double y = y_true[i](j, k);
                    double p = y_pred[i](j, k);
                    // Add small epsilon to avoid division by zero
                    grad[i](j, k) = -(y / (p + 1e-15) - (1 - y) / (1 - p + 1e-15));
                }
            }
        }
        return grad;
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