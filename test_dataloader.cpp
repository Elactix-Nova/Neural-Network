#include "dataloader.hpp"
#include <iostream>
#include <iomanip>

// Helper function to print a single matrix
void printMatrix(const Eigen::MatrixXd& matrix, const std::string& name) {
    std::cout << name << ":\n" << matrix << "\n";
}

// Testing function
void testDataLoader() {
    // Create sample input data
    std::vector<Eigen::MatrixXd> input_data;
    std::vector<int> labels;

    int num_classes = 10;  // Assume 10 possible classes (0-9)
    int batch_size = 4;    // Batch size per iteration

    // Generate 8 sample 3x3 matrices with labels
    for (int i = 0; i < 8; ++i) {
        Eigen::MatrixXd matrix(3, 3);
        matrix << (i + 1) * 0.1, (i + 2) * 0.1, (i + 3) * 0.1,
                  (i + 4) * 0.1, (i + 5) * 0.1, (i + 6) * 0.1,
                  (i + 7) * 0.1, (i + 8) * 0.1, (i + 9) * 0.1;
        input_data.push_back(matrix);
        labels.push_back(i % num_classes);  // Assign cyclic labels (0-9)
    }

    // Create DataLoader instance
    DataLoader loader(input_data, labels, batch_size, num_classes, true);

    std::cout << "\n==================== DataLoader Testing ====================\n";

    // Iterate through batches
    while (loader.has_next_batch()) {
        auto [batch_inputs, batch_labels] = loader.get_next_batch();
        // std::cout 
        std::cout << "\nBatch " << loader.get_num_batches() << ":\n";
        std::cout << "\nBatch " << batch_inputs.size() + 1 << ":\n";
        std::cout << "\nBatch " << loader.get_num_batches() - batch_inputs.size() + 1 << ":\n";

        // Print each sample in the batch
        for (size_t j = 0; j < batch_inputs.size(); ++j) {
            std::cout << "Sample " << j + 1 << ":\n";
            printMatrix(batch_inputs[j][0], "Input");  // Extract the first matrix in the sample
            
            // Print label (stored as std::vector<Eigen::MatrixXd>)
            for (size_t k = 0; k < batch_labels[j].size(); ++k) {
                printMatrix(batch_labels[j][k], "One-Hot Label");
            }

            std::cout << "------------------------\n";
        }
    }

    // Test Reset
    std::cout << "\n==================== Testing Reset Functionality ====================\n";
    for (int i = 0; i < 10; i++){
        loader.reset();
        auto [reset_inputs, reset_labels] = loader.get_next_batch();

        std::cout << "First batch after reset:\n";
        for (size_t j = 0; j < reset_inputs.size(); ++j) {
            std::cout << "Sample: " << j << std::endl;
            printMatrix(reset_inputs[j][0], "Input");

            for (size_t k = 0; k < reset_labels[j].size(); ++k) {
                printMatrix(reset_labels[j][k], "One-Hot Label");
            }

            std::cout << "------------------------\n";
        }
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    testDataLoader();
    return 0;
}
