#include "losses.hpp"
#include "convolutional.hpp"
#include "reshape.hpp"
#include <iostream>
#include <iomanip>


void printMatrixVector(const std::vector<Eigen::MatrixXd>& matrices, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < matrices.size(); ++i) {
        std::cout << "Channel " << i << ":\n" << matrices[i] << "\n\n";
    }
}

void testReshape() {
    std::cout << "\n=== Testing Reshape Layer ===\n";
    
    // Test case 1: Reshape from 2 channels of 2x2 to 1 channel of 4x2
    std::vector<int> input_shape = {2, 2, 2};   // 2 channels, 2x2 each
    std::vector<int> output_shape = {1, 4, 2};  // 1 channel, 4x2
    
    Reshape reshape(input_shape, output_shape);
    
    // Create input: 2 channels of 2x2 matrices
    std::vector<Eigen::MatrixXd> input(2);
    input[0] = Eigen::MatrixXd(2, 2);
    input[0] << 1, 2,
                3, 4;
    input[1] = Eigen::MatrixXd(2, 2);
    input[1] << 5, 6,
                7, 8;
    
    std::cout << "Input:\n";
    printMatrixVector(input, "Input matrices");
    
    // Forward pass
    std::vector<Eigen::MatrixXd> output = reshape.forward(input);
    printMatrixVector(output, "Reshaped output");
    
    // Verify total elements are preserved
    int input_elements = 0;
    for (const auto& mat : input) {
        input_elements += mat.size();
    }
    int output_elements = 0;
    for (const auto& mat : output) {
        output_elements += mat.size();
    }
    std::cout << "Total elements preserved: " << (input_elements == output_elements ? "Yes" : "No") 
              << " (Input: " << input_elements << ", Output: " << output_elements << ")\n\n";
    
    // Test backward pass
    std::vector<Eigen::MatrixXd> output_gradient(1);
    output_gradient[0] = Eigen::MatrixXd(4, 2);
    output_gradient[0] << 0.1, 0.2,
                         0.3, 0.4,
                         0.5, 0.6,
                         0.7, 0.8;
    
    std::cout << "Output gradient:\n";
    printMatrixVector(output_gradient, "Output gradient");
    
    // Backward pass
    std::vector<Eigen::MatrixXd> input_gradient = reshape.backward(output_gradient, 0.01);
    printMatrixVector(input_gradient, "Reshaped gradient");
    
    // Test case 2: Reshape from 1 channel of 4x2 back to 2 channels of 2x2
    std::vector<int> input_shape2 = {1, 4, 2};   // 1 channel, 4x2
    std::vector<int> output_shape2 = {2, 2, 2};  // 2 channels, 2x2 each
    
    Reshape reshape2(input_shape2, output_shape2);
    
    std::cout << "\nTest case 2 - Reshape back to original shape:\n";
    std::vector<Eigen::MatrixXd> input2 = output;  // Use output from previous test
    printMatrixVector(input2, "Input");
    
    std::vector<Eigen::MatrixXd> output2 = reshape2.forward(input2);
    printMatrixVector(output2, "Reshaped output (should match original input)");
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    // testMSE();
    // testBinaryCrossEntropy();
    // testConvolutional();
    testReshape();
    return 0;
}
