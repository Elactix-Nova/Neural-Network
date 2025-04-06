#include "dataloader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
// #include "image_loader.hpp" 

DataLoader::DataLoader(const ImageFolder& image_folder,
                       int batch_size,
                       bool shuffle)
    : batch_size(batch_size),
      shuffle(shuffle),
      current_batch(0),
      num_classes(image_folder.labels.size())
{
    for (int label = 0; label < static_cast<int>(image_folder.images.size()); ++label) {
        for (const auto& img_ptr : image_folder.images[label]) {
            std::vector<Eigen::MatrixXd> input = { *img_ptr };  // Copy contents
            data.emplace_back(input, one_hot_encode(label));
        }
    }

    if (shuffle) {
        shuffle_data();
    }

    num_batches = (data.size() + batch_size - 1) / batch_size;
}
DataLoader::DataLoader(const std::vector<Eigen::MatrixXd>& input_data, 
					   const std::vector<int>& labels,
					   int batch_size,
					   int num_classes,
					   bool shuffle)
	: batch_size(batch_size), shuffle(shuffle), current_batch(0), num_classes(num_classes) {
	
	// Check if input data and labels have the same size
	if (input_data.size() != labels.size()) {
		throw std::runtime_error("Input data and labels must have the same size");
	}
	
	// Store the data
	for (size_t i = 0; i < input_data.size(); ++i) {
		std::vector<Eigen::MatrixXd> input;
		input.push_back(input_data[i]);
		data.emplace_back(input, one_hot_encode(labels[i]));
	}
	
	if (shuffle) {
		shuffle_data();
	}
	
	num_batches = (data.size() + batch_size - 1) / batch_size;
}

std::vector<Eigen::MatrixXd> DataLoader::one_hot_encode(int label) {
	std::vector<Eigen::MatrixXd> encoded;
	Eigen::MatrixXd label_matrix = Eigen::MatrixXd::Zero(num_classes, 1);
	label_matrix(label, 0) = 1.0;
	encoded.push_back(label_matrix);
	return encoded;
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