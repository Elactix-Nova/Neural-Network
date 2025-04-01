#include "losses.hpp"
#include "convolutional.hpp"
#include <iostream>
#include <iomanip>

void printMatrix(const Eigen::MatrixXd& matrix, const std::string& name) {
    std::cout << name << ":\n" << matrix << "\n\n";
}

void testMSE() {
    std::cout << "=== Testing MSE Loss ===\n";
    
    // Test case 1: Simple 2x1 matrices
    Eigen::MatrixXd y_true1(2, 1);
    y_true1 << 1.0, 0.0;
    Eigen::MatrixXd y_pred1(2, 1);
    y_pred1 << 0.8, 0.2;
    
    printMatrix(y_true1, "y_true1");
    printMatrix(y_pred1, "y_pred1");
    
    double mse_loss1 = Loss::mse(y_true1, y_pred1);
    Eigen::MatrixXd mse_grad1 = Loss::mse_prime(y_true1, y_pred1);
    
    std::cout << "MSE Loss: " << mse_loss1 << "\n";
    printMatrix(mse_grad1, "MSE Gradient");
    
    // Test case 2: 3x2 matrices
    Eigen::MatrixXd y_true2(3, 2);
    y_true2 << 1.0, 0.0,
               0.0, 1.0,
               0.5, 0.5;
    Eigen::MatrixXd y_pred2(3, 2);
    y_pred2 << 0.9, 0.1,
               0.1, 0.9,
               0.4, 0.6;
    
    printMatrix(y_true2, "y_true2");
    printMatrix(y_pred2, "y_pred2");
    
    double mse_loss2 = Loss::mse(y_true2, y_pred2);
    Eigen::MatrixXd mse_grad2 = Loss::mse_prime(y_true2, y_pred2);
    
    std::cout << "MSE Loss: " << mse_loss2 << "\n";
    printMatrix(mse_grad2, "MSE Gradient");
}

void testBinaryCrossEntropy() {
    std::cout << "\n=== Testing Binary Cross-Entropy Loss ===\n";
    
    // Test case 1: Simple 2x1 matrices
    Eigen::MatrixXd y_true1(2, 1);
    y_true1 << 1.0, 0.0;
    Eigen::MatrixXd y_pred1(2, 1);
    y_pred1 << 0.8, 0.2;
    
    printMatrix(y_true1, "y_true1");
    printMatrix(y_pred1, "y_pred1");
    
    double bce_loss1 = Loss::binary_cross_entropy(y_true1, y_pred1);
    Eigen::MatrixXd bce_grad1 = Loss::binary_cross_entropy_prime(y_true1, y_pred1);
    
    std::cout << "Binary Cross-Entropy Loss: " << bce_loss1 << "\n";
    printMatrix(bce_grad1, "Binary Cross-Entropy Gradient");
    
    // Test case 2: 3x2 matrices
    Eigen::MatrixXd y_true2(3, 2);
    y_true2 << 1.0, 0.0,
               0.0, 1.0,
               0.5, 0.5;
    Eigen::MatrixXd y_pred2(3, 2);
    y_pred2 << 0.9, 0.1,
               0.1, 0.9,
               0.4, 0.6;
    
    printMatrix(y_true2, "y_true2");
    printMatrix(y_pred2, "y_pred2");
    
    double bce_loss2 = Loss::binary_cross_entropy(y_true2, y_pred2);
    Eigen::MatrixXd bce_grad2 = Loss::binary_cross_entropy_prime(y_true2, y_pred2);
    
    std::cout << "Binary Cross-Entropy Loss: " << bce_loss2 << "\n";
    printMatrix(bce_grad2, "Binary Cross-Entropy Gradient");
}

void testConvolutional() {
    std::cout << "\n=== Testing Convolutional Layer ===\n";
    
    // Test case: 3x3 input with 2 channels, 2x2 kernel, 2 output channels
    std::vector<int> input_shape = {2, 3, 3};  // 2 channels, 3x3 input
    Convolutional conv(input_shape, 2, 2);     // 2x2 kernel, 2 output channels
    
    // Create input with 2 channels (6x3 matrix)
    Eigen::MatrixXd input(6, 3);
    // First channel (top 3x3)
    input.block(0, 0, 3, 3).setConstant(2.0);
    // Second channel (bottom 3x3)
    input.block(3, 0, 3, 3).setConstant(2.0);
    
    printMatrix(input, "Input (2 channels)");
    
    // Forward pass
    Eigen::MatrixXd output = conv.forward(input);
    printMatrix(output, "Forward Output (2 channels)");
    
    // Backward pass
    // Create gradient with 2 channels (4x2 matrix)
    Eigen::MatrixXd grad(4, 2);
    grad.setConstant(2.0);  // Set all values to 2
    
    Eigen::MatrixXd back_grad = conv.backward(grad, 0.01);
    printMatrix(back_grad, "Backward Gradient (2 channels)");
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    testMSE();
    testBinaryCrossEntropy();
    testConvolutional();
    return 0;
}
