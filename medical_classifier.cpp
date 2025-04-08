#include "network.hpp"
#include "dense.hpp"
#include "convolutional.hpp"
#include "reshape.hpp"
#include "activations.hpp"
#include "pooling.hpp"
#include "losses.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include "dataloader.hpp"
using namespace std;

int main()
{
	// Change file paths to path to your dataset
	ImageFolder train_folder = ImageFolder("/Users/id19/Programming/Dev/ML CNN Assignment/test_dataset/train");
	ImageFolder val_folder = ImageFolder("/Users/id19/Programming/Dev/ML CNN Assignment/test_dataset/val");
	DataLoader train_loader = DataLoader(train_folder, 4, true);
	DataLoader val_loader = DataLoader(val_folder, 4, false);
	auto test_batch = train_loader.get_next_batch();
	train_loader.reset();
	auto test = val_loader.get_next_batch();
	cout << test.first << endl;
	val_loader.reset();
	cout << "Loaded successfully" << endl;

	// Now, define network architecture
	// Get number of classes from the training dataset
	int num_classes = train_folder.labels.size();

	// Create layers vector
	std::vector<std::shared_ptr<Layer>> layers;

	/*
	Convolutional(const std::vector<int>& input_shape, 
                    int kernel_size, 
                    int depth, 
                    int stride = 1, 
                    int padding = 0
                    );
	*/

	// For max pooling layer: Default stride is kernel size(2)
	// 300x300 dimension input images
	// First Convolutional Block
	vector<int> input_shape = {1,300,300};
	// Input: 1 channel, Output: 32 channels, 3x3 kernel, 150 x150
	layers.push_back(std::make_shared<Convolutional>(input_shape, 3, 32, 2, 1));  //150 x150, padding = 1 (same)
	layers.push_back(std::make_shared<ReLU>());
	layers.push_back(std::make_shared<MaxPooling>(2, 2));  // 2x2 pooling, default stride is 2
	// Output will be 75 x 75

	// Second Convolutional Block
	vector<int> shape_32 = {32, 75, 75};  // 32 channels from prev layer, 75x75 after pooling
	layers.push_back(std::make_shared<Convolutional>(shape_32, 3, 64, 2, 1));  // 3x3 kernel, 64 filters, 37 x 37
	// 37 x 37
	layers.push_back(std::make_shared<ReLU>());
	layers.push_back(std::make_shared<MaxPooling>(2, 2));
	// 18 x 18

	// Third Convolutional Block
	vector<int> shape_64 = {64, 18, 18};  // 64 channels from prev layer, 18x18 after pooling
	layers.push_back(std::make_shared<Convolutional>(shape_64, 3, 128));  // 3x3 kernel, 128 filters, 16 x 16
	layers.push_back(std::make_shared<ReLU>());
	layers.push_back(std::make_shared<MaxPooling>(2, 2));
	// 8 x 8

	// Fourth Convolutional Block
	vector<int> shape_128 = {128, 8, 8};  // 128 channels from prev layer, 8x8 after pooling
	layers.push_back(std::make_shared<Convolutional>(shape_128, 3, 256));  // 3x3 kernel, 256 filters, 6 x 6
	layers.push_back(std::make_shared<ReLU>());
	layers.push_back(std::make_shared<MaxPooling>(2, 2)); // 3 x 3

	// Fifth and final Convolutional Block
	vector<int> shape_256 = {256, 3, 3};  // 256 channels from prev layer, 3x3 after pooling
	layers.push_back(std::make_shared<Convolutional>(shape_256, 3, 512));  // 3x3 kernel, 512 filters, 1 x 1
	layers.push_back(std::make_shared<ReLU>());
	// No need for GlobalAvgPooling since we already have 1x1 spatial dimensions

	// Flatten the output for dense layers
	// With 512 channels: 512 * 1 * 1 = 512
	layers.push_back(std::make_shared<Reshape>(vector<int>{512, 1, 1}, vector<int>{512, 1, 1}));

	// Dense layers for classification
	// Dense constructor takes input_size, output_size
	layers.push_back(std::make_shared<Dense>(512, 512));
	layers.push_back(std::make_shared<ReLU>());
	layers.push_back(std::make_shared<Dense>(512, 256));
	layers.push_back(std::make_shared<ReLU>());
	layers.push_back(std::make_shared<Dense>(256, 128));
	layers.push_back(std::make_shared<ReLU>());
	// output layer
	layers.push_back(std::make_shared<Dense>(128, num_classes));
	
	// Create network with layers
	Network network(layers);

	// Set loss function
	// network.set_loss<CrossEntropyLoss>();  // This line is causing the error
	cout << "network init done successfully" << endl;

	// Test the network layer by layer
	cout << "\n=== Testing Network Layer by Layer ===\n";
	
	// Create a sample input (1 channel, 300x300)
	std::vector<Eigen::MatrixXd> sample_input(1, Eigen::MatrixXd::Ones(300, 300));
	cout << "Input shape: 1x" << sample_input[0].rows() << "x" << sample_input[0].cols() << endl;
	
	// Process through each layer and print dimensions
	std::vector<Eigen::MatrixXd> current_output = sample_input;
	
	for (size_t i = 0; i < layers.size(); ++i) {
		cout << "\nLayer " << i << ": ";
		
		// Get layer type
		if (dynamic_cast<Convolutional*>(layers[i].get())) {
			cout << "Convolutional";
		} else if (dynamic_cast<ReLU*>(layers[i].get())) {
			cout << "ReLU";
		} else if (dynamic_cast<MaxPooling*>(layers[i].get())) {
			cout << "MaxPooling";
		} else if (dynamic_cast<GlobalAvgPooling*>(layers[i].get())) {
			cout << "GlobalAvgPooling";
		} else if (dynamic_cast<Reshape*>(layers[i].get())) {
			cout << "Reshape";
		} else if (dynamic_cast<Dense*>(layers[i].get())) {
			cout << "Dense";
		} else {
			cout << "Unknown";
		}
		
		// Process through this layer
		current_output = layers[i]->forward(current_output);
		
		// Print output dimensions
		cout << " -> Output shape: " << current_output.size() << " channels";
		if (!current_output.empty()) {
			cout << ", " << current_output[0].rows() << "x" << current_output[0].cols();
		}
		cout << endl;
	}
	
	cout << "\nFinal output shape: " << current_output.size() << " channels";
	if (!current_output.empty()) {
		cout << ", " << current_output[0].rows() << "x" << current_output[0].cols();
	}
	cout << endl;

	// Training code
	cout << "\n=== Starting Training ===\n";
	
	// Training parameters
	int epochs = 50;
	double learning_rate = 0.001;
	bool verbose = true;
	
	// Prepare training data
	std::vector<std::vector<Eigen::MatrixXd>> x_train;
	std::vector<std::vector<Eigen::MatrixXd>> y_train;
	
	// Collect all training data
	while (train_loader.has_next_batch()) {
		auto [batch_inputs, batch_labels] = train_loader.get_next_batch();
		x_train.insert(x_train.end(), batch_inputs.begin(), batch_inputs.end());
		y_train.insert(y_train.end(), batch_labels.begin(), batch_labels.end());
	}
	
	// Train the network
	network.train(x_train, y_train,
		Loss::cross_entropy_loss, Loss::cross_entropy_loss_prime,
		epochs, learning_rate, verbose);
	
	cout << "Training completed successfully" << endl;
	
	return 0;
}