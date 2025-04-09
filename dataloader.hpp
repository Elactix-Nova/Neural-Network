#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include "stb_image/stb_image.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
using namespace std;

typedef struct ImageStruct
{
	int channels;
	int height;
	int width;
	string file_path;
	std::shared_ptr<std::vector<Eigen::MatrixXd>> actual_image; 
	string label;
} ImageStruct;

class ImageFolder
{
	public:
		// Initialise image folder with all the images from the nested directory structure
		// Only relevant method of class
		ImageFolder(string folder_root);
		unordered_map<string, int> get_label_counts();
		std::vector<Eigen::MatrixXd> raw_img_to_matrix(unsigned char* img, int channels, int width, int height);
		~ImageFolder(); // Destructor to avoid memory leaks
		string root_folder_path;
		vector<string> labels;
		int num_classes;
		
		// How images are stored:
		// 4 dimensions: 1 - which label, 2 - which image in label,
		// 3(shared ptr) - which channel in image, 
		// 4 - actual matrix containing pixel values for that image
		vector<vector<std::shared_ptr<std::vector<Eigen::MatrixXd>>>> images; // shared ptr makes our life way easier
		
		// Store image metadata by image
		// Not separated by channels obviously, so dimensions only are 1. Label, 2. Image
		vector<vector<ImageStruct>> images_data;
		unordered_map<string, int> label_counts;
};

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
	void shuffle_data();

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

	std::vector<Eigen::MatrixXd> one_hot_encode(int label);
};