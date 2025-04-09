#pragma once

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include "image_loader.hpp"

class DataLoader {
public:
	// Construct from an ImageFolder
	DataLoader(const ImageFolder& image_folder,
			   int batch_size,
			   bool shuffle = true);

	// Construct from raw matrices + integer labels
	// Not preferred
	DataLoader(const std::vector<Eigen::MatrixXd>& input_data,
			   const std::vector<int>& labels,
			   int batch_size,
			   int num_classes,
			   bool shuffle = true);

	// Get next batch:
	//  - first:  batch_inputs  = vector< sample >,
	//                    sample = vector< channel_matrices >
	//  - second: batch_labels  = vector< sample >,
	//                    sample = vector< one_hot_matrix >
	std::pair<
		std::vector<std::vector<Eigen::MatrixXd>>,   // batch_inputs
		std::vector<std::vector<Eigen::MatrixXd>>    // batch_labels
	>
	get_next_batch();

	bool has_next_batch() const;
	void reset();
	int  get_num_batches() const;
	int batch_size;
	int num_classes;

private:
	// data[i].first  = vector<Eigen::MatrixXd>  → channels of image i
	// data[i].second = vector<Eigen::MatrixXd>  → one‑hot label for image i
	std::vector<
		std::pair<
			std::vector<Eigen::MatrixXd>,
			std::vector<Eigen::MatrixXd>
		>
	> data;

	bool shuffle;
	int current_batch;
	int num_batches;

	void shuffle_data();
	std::vector<Eigen::MatrixXd> one_hot_encode(int label);
};
