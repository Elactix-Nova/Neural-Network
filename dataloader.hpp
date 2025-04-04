#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

class DataLoader {
public:
    // Constructor with direct matrix input
    DataLoader(const std::vector<Eigen::MatrixXd>& input_data, 
               const std::vector<int>& labels,
               int batch_size,
               int num_classes,
               bool shuffle = true);
    
    // Get next batch of data
    std::pair<std::vector<std::vector<Eigen::MatrixXd>>, std::vector<std::vector<Eigen::MatrixXd>>> get_next_batch();
    
    // Check if there are more batches
    bool has_next_batch() const;
    
    // Reset the dataloader to start from beginning
    void reset();
    
    // Get total number of batches
    int get_num_batches() const;

private:
    std::vector<std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>> data;
    int batch_size;
    bool shuffle;
    int current_batch;
    int num_batches;
    int num_classes;
    
    // Helper functions
    void shuffle_data();
    std::vector<Eigen::MatrixXd> one_hot_encode(int label);
};
