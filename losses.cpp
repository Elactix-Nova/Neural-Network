#include "losses.hpp"
#include <cmath>

namespace Loss {
    double mse(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred) {
        double loss = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            loss += (y_true[i] - y_pred[i]).array().square().sum();
        }
        return loss / y_true.size();
    }

    std::vector<Eigen::MatrixXd> mse_prime(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred) {
        std::vector<Eigen::MatrixXd> grad(y_true.size());
        for (size_t i = 0; i < y_true.size(); ++i) {
            grad[i] = 2.0 * (y_pred[i] - y_true[i]) / y_true[i].rows();
        }
        return grad;
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

    double cross_entropy_loss(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred) {
        const double epsilon = 1e-15;
        double loss = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            Eigen::MatrixXd clipped_pred = y_pred[i].array().max(epsilon).min(1 - epsilon);
            loss += -(y_true[i].array() * clipped_pred.array().log()).sum();
        }
        return loss / y_true.size();
    }

    std::vector<Eigen::MatrixXd> cross_entropy_loss_prime(const std::vector<Eigen::MatrixXd>& y_true, const std::vector<Eigen::MatrixXd>& y_pred) {
        const double epsilon = 1e-15;
        std::vector<Eigen::MatrixXd> grad(y_true.size());
        for (size_t i = 0; i < y_true.size(); ++i) {
            Eigen::MatrixXd clipped_pred = y_pred[i].array().max(epsilon).min(1 - epsilon);
            grad[i] = (clipped_pred - y_true[i]) / y_true[i].rows();
        }
        return grad;
    }
}