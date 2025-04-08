#include "dataloader.hpp"
#include <iostream>
#include <iomanip>

// Helper to print a single matrix
void printMatrix(const Eigen::MatrixXd& m, const std::string& name) {
    std::cout << name << ":\n" << m << "\n";
}

void test_loader_two() {
    // 1) Load your dataset
    ImageFolder dataset("/Users/id19/Programming/Dev/ML CNN Assignment/test_dataset/val");
    int batch_size = 16;
    DataLoader loader(dataset, batch_size, /*shuffle=*/true);

    std::cout << "\n===== ImageFolder→DataLoader Test =====\n";
    int batch_idx = 1;

    while (loader.has_next_batch()) {
        auto [inputs, labels] = loader.get_next_batch();
        std::cout << "\nBatch " << batch_idx++ 
                  << " (size = " << inputs.size() << ")\n";
        std::cout << "-------------------------------------\n";

        for (size_t i = 0; i < inputs.size(); ++i) {
            // --- Decode the label ---
            // labels[i] is vector<Eigen::MatrixXd>, with a single one-hot matrix at [0]
            const auto& one_hot = labels[i][0];
            int label_index;
            one_hot.col(0).maxCoeff(&label_index);
            const std::string& label_str = dataset.labels[label_index];

            // --- Compute mean pixel over all channels ---
            double sum = 0.0;
            int total_pixels = 0;
            for (const auto& channel_mat : inputs[i]) {
                sum += channel_mat.sum();
                total_pixels += channel_mat.rows() * channel_mat.cols();
            }
            double mean_pixel = sum / total_pixels;

            // --- Print a meaningful summary ---
            std::cout << " Image " << i 
                      << " | Class = \"" << label_str << "\" (" << label_index << ")"
                      << " | Mean pixel = " << std::fixed << std::setprecision(4)
                      << mean_pixel << "\n";
        }
    }
}


int main() {
    std::cout << std::fixed << std::setprecision(4);
    test_loader_two();
    return 0;
}
