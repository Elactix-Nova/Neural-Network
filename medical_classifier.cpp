#include "network.hpp"
#include "dense.hpp"
#include "convolutional.hpp"
#include "reshape.hpp"
#include "activations.hpp"
#include "pooling.hpp"
#include "losses.hpp"
#include "dataloader.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
using namespace std;

double eval(Network network, DataLoader val_loader, int num_batches = 5){
	val_loader.reset();
	val_loader.shuffle_data(); // Get different batches each time
	double overall_loss = 0;
	double loss;
	for (int i = 0 ; i < num_batches; i++){
		auto [val_x, val_label] = val_loader.get_next_batch();
		for (int sample = 0; sample < val_x.size(); sample++){
			auto pred = network.predict(val_x[sample]);
			loss =  Loss::cross_entropy_loss(val_label[sample], pred);
			overall_loss += loss;
		}
	}
	overall_loss /= num_batches * val_loader.batch_size;
	return overall_loss;
}

// Function to print batch information for debugging
void print_batch_info(const std::vector<std::vector<Eigen::MatrixXd>>& batch_x, 
                     const std::vector<std::vector<Eigen::MatrixXd>>& batch_y) {
    // Print batch dimensions
    cout << "Batch size: " << batch_x.size() << endl;
    if (!batch_x.empty()) {
        cout << "First sample channels: " << batch_x[0].size() << endl;
        if (!batch_x[0].empty()) {
            cout << "First channel dimensions: " << batch_x[0][0].rows() << "x" << batch_x[0][0].cols() << endl;
            
            // Print a small sample of the first channel
            cout << "Sample of first channel (top-left 5x5):" << endl;
            int rows = std::min(5, batch_x[0][0].rows());
            int cols = std::min(5, batch_x[0][0].cols());
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    cout << batch_x[0][0](i, j) << " ";
                }
                cout << endl;
            }
        }
    }
    
    // Print label dimensions
    cout << "Label batch size: " << batch_y.size() << endl;
    if (!batch_y.empty()) {
        cout << "First label channels: " << batch_y[0].size() << endl;
        if (!batch_y[0].empty()) {
            cout << "First label channel dimensions: " << batch_y[0][0].rows() << "x" << batch_y[0][0].cols() << endl;
            cout << "First label values:" << endl;
            for (int i = 0; i < batch_y[0][0].rows(); i++) {
                cout << batch_y[0][0](i, 0) << " ";
            }
            cout << endl;
        }
    }
}

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
	// cout << test.first << endl;
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
	layers.push_back(std::make_shared<Reshape>(vector<int>{512, 1, 1}, vector<int>{1, 512, 1}));

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
	cout << "network init done successfully" << endl;

	// Training code
	cout << "\n=== Starting Training ===\n";
	
	// Training parameters
	int num_epochs = 100;
	double learning_rate = 0.001;
	bool verbose = true;
	
	double epoch_loss = 0;
	for (int epoch = 0; epoch < num_epochs; epoch++){
		cout << "Starting epoch: " << epoch << endl;
		train_loader.reset();
		epoch_loss = 0;
		if (epoch < 10){
			learning_rate = 0.005;
		}
		for (int batch_num = 0; batch_num < train_loader.get_num_batches(); batch_num++){
			cout << "Batch num: " << batch_num << endl;
			auto [batch_x, batch_y] = train_loader.get_next_batch();
			
			// Print batch information for debugging
			print_batch_info(batch_x, batch_y);
			
			cout << "Starting training on batch" << endl;
            network.train(batch_x, batch_y,
                          Loss::cross_entropy_loss,
                          Loss::cross_entropy_loss_prime,
                          1, learning_rate, true);
			cout << "Ending training on batch" << endl;
		}
		cout << "Getting val loss" << endl;
		double val_loss = eval(network, val_loader, 8);
		cout << "Epoch " << epoch << " / " << num_epochs << ": validation loss: " << val_loss << endl;
	}
	
	double final_loss = eval(network, val_loader, val_loader.get_num_batches());
	cout << "Final validation loss: " << final_loss << endl;
	
	cout << "Training completed successfully" << endl;
	
	return 0;
}

/* EARLIER CODE FOR TESTING*/
	// // Process through each layer and print dimensions
	// std::vector<Eigen::MatrixXd> current_output = sample_input;
	
	// for (size_t i = 0; i < layers.size(); ++i) {
	// 	cout << "\nLayer " << i << ": ";
		
	// 	// Get layer type
	// 	if (dynamic_cast<Convolutional*>(layers[i].get())) {
	// 		cout << "Convolutional";
	// 	} else if (dynamic_cast<ReLU*>(layers[i].get())) {
	// 		cout << "ReLU";
	// 	} else if (dynamic_cast<MaxPooling*>(layers[i].get())) {
	// 		cout << "MaxPooling";
	// 	} else if (dynamic_cast<GlobalAvgPooling*>(layers[i].get())) {
	// 		cout << "GlobalAvgPooling";
	// 	} else if (dynamic_cast<Reshape*>(layers[i].get())) {
	// 		cout << "Reshape";
	// 	} else if (dynamic_cast<Dense*>(layers[i].get())) {
	// 		cout << "Dense";
	// 	} else {
	// 		cout << "Unknown";
	// 	}
		
	// 	// Process through this layer
	// 	current_output = layers[i]->forward(current_output);
		
	// 	// Print output dimensions
	// 	cout << " -> Output shape: " << current_output.size() << " channels";
	// 	if (!current_output.empty()) {
	// 		cout << ", " << current_output[0].rows() << "x" << current_output[0].cols();
	// 	}
	// 	cout << endl;
	// }
	
	// cout << "\nFinal output shape: " << current_output.size() << " channels";
	// if (!current_output.empty()) {
	// 	cout << ", " << current_output[0].rows() << "x" << current_output[0].cols();
	// }
	// cout << endl;