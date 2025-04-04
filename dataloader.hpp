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
    // Constructor
    DataLoader(const std::string& data_path, int batch_size, bool shuffle = true);
    
    // Get next batch of data
    std::pair<std::vector<std::vector<Eigen::MatrixXd>>, std::vector<std::vector<Eigen::MatrixXd>>> get_next_batch();
    
    // Check if there are more batches
    bool has_next_batch() const;
    
    // Reset the dataloader to start from beginning
    void reset();
    
    // Get total number of batches
    int get_num_batches() const;

private:
    std::string data_path;
    int batch_size;
    bool shuffle;
    std::vector<std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>> data;
    int current_batch;
    int num_batches;
    
    // Helper functions
    void load_data();
    void shuffle_data();
};
