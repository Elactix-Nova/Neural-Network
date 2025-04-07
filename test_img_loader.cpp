#include "stb_image/stb_image.h"
#include "image_loader.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
    string dataset_path = "/Users/id19/Programming/Dev/ML CNN Assignment/test_dataset/val";  // or wherever your dataset is

    try {
        ImageFolder folder(dataset_path);

        cout << "\nLoaded " << folder.labels.size() << " classes." << endl;

        for (size_t class_idx = 0; class_idx < folder.labels.size(); ++class_idx) {
            const string& label = folder.labels[class_idx];
            cout << "\nClass " << class_idx << ": " << label
                << " — " << folder.label_counts[label] << " images" << endl;

			if (!folder.images[class_idx].empty()) {
				auto img_ptr = folder.images[class_idx][0]; // shared_ptr<vector<Eigen::MatrixXd>>
				const auto& channels = *img_ptr;
			
				cout << "Image has " << channels.size() << " channels" << endl;
			
				double pixel_sum = 0.0;
				int total_pixels = 0;
			
				for (size_t c = 0; c < channels.size(); ++c) {
					const Eigen::MatrixXd& mat = channels[c];
					int rows = mat.rows(), cols = mat.cols();
			
					pixel_sum += mat.sum();
					total_pixels += rows * cols;
			
					cout << "  Channel " << c << ": " << rows << "x" << cols << endl;
					cout << "  Sample pixel (0,0): " << mat(0, 0) << endl;
				}
			
				double mean_pixel_value = pixel_sum / total_pixels;
				cout << "Mean pixel value across all channels: " << mean_pixel_value << endl;
			}
        }

    } catch (const exception& e) {
        cerr << "Error loading dataset: " << e.what() << endl;
    }

    return 0;
}
