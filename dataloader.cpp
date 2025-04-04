#include "dataloader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

DataLoader::DataLoader(const std::string& data_path, int batch_size, bool shuffle)
    : data_path(data_path), batch_size(batch_size), shuffle(shuffle), current_batch(0) {
    load_data();
    if (shuffle) {
        shuffle_data();
    }
    num_batches = (data.size() + batch_size - 1) / batch_size;
}

void DataLoader::load_data() {
    std::ifstream file(data_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + data_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> values;
        
        // Read comma-separated values
        while (std::getline(ss, value, ',')) {
            values.push_back(std::stod(value));
        }

        // Assuming first value is label, rest is input data
        if (values.size() > 1) {
            std::vector<Eigen::MatrixXd> input;
            std::vector<Eigen::MatrixXd> label;
            
            // Create input matrix (assuming square image)
            int size = static_cast<int>(std::sqrt(values.size() - 1));
            Eigen::MatrixXd input_matrix = Eigen::Map<Eigen::MatrixXd>(values.data() + 1, size, size);
            input.push_back(input_matrix);
            
            // Create one-hot encoded label
            Eigen::MatrixXd label_matrix = Eigen::MatrixXd::Zero(10, 1);
            label_matrix(static_cast<int>(values[0]), 0) = 1.0;
            label.push_back(label_matrix);
            
            data.emplace_back(input, label);
        }
    }
}

void DataLoader::shuffle_data() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(data.begin(), data.end(), gen);
}

std::pair<std::vector<std::vector<Eigen::MatrixXd>>, std::vector<std::vector<Eigen::MatrixXd>>> 
DataLoader::get_next_batch() {
    if (!has_next_batch()) {
        throw std::runtime_error("No more batches available");
    }

    std::vector<std::vector<Eigen::MatrixXd>> batch_inputs;
    std::vector<std::vector<Eigen::MatrixXd>> batch_labels;

    int start_idx = current_batch * batch_size;
    int end_idx = std::min(start_idx + batch_size, static_cast<int>(data.size()));

    for (int i = start_idx; i < end_idx; ++i) {
        batch_inputs.push_back(data[i].first);
        batch_labels.push_back(data[i].second);
    }

    current_batch++;
    return {batch_inputs, batch_labels};
}

bool DataLoader::has_next_batch() const {
    return current_batch < num_batches;
}

void DataLoader::reset() {
    current_batch = 0;
    if (shuffle) {
        shuffle_data();
    }
}

int DataLoader::get_num_batches() const {
    return num_batches;
}
