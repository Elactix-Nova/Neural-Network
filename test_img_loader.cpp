#include "stb_image/stb_image.h"
#include "image_loader.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
	string dataset_root = "/Users/id19/Programming/Dev/ML CNN Assignment/test_dataset/val";  // adjust to your dataset folder
	ImageFolder dataset(dataset_root);

	cout << "Total classes: " << dataset.num_classes << "\n";

	// Loop through each class
	for (size_t class_idx = 0; class_idx < dataset.images.size(); ++class_idx) {
		string label = dataset.labels[class_idx];
		cout << "\n--- Class: " << label << " ---\n";

		// Loop through each image in this class
		for (size_t img_idx = 0; img_idx < dataset.images[class_idx].size(); ++img_idx) {
			cout << "Image " << img_idx << ":\n";
			std::shared_ptr<Eigen::MatrixXd> image_ptr = dataset.images[class_idx][img_idx];
			const ImageStruct& meta = dataset.images_data[class_idx][img_idx];

			int channels = meta.channels;
			int height = meta.height;
			int width = meta.width;
			string label = meta.label;
			
			cout << "At path: " << meta.file_path << ", " << channels << ", " << height << ", " << width << ", " << label << endl;

			const Eigen::MatrixXd& matrix = *image_ptr;

			// Print pixel values by channel
			double mean;
			for (int c = 0; c < channels; ++c) {
				cout << "Channel " << c << ":\n";
				for (int y = 0; y < height; ++y) {
					for (int x = 0; x < width; ++x) {
						mean += matrix(c, y * width + x);
						// cout << val << " ";
					}
					// cout << "\n";
				}
				mean = mean / (height*width);
				cout << "mean for this channel: " << mean << endl;
			}
		}
	}

	return 0;
}
