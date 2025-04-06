#include "stb_image/stb_image.h"
#include "image_loader.hpp"
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main()
{
	Final_Matrix rgb_image = read_image("test_images/grayscale.jpeg", true);
	int channels = rgb_image.channels;
	int height = rgb_image.height;
	int width = rgb_image.width;
	Eigen::MatrixXd image_matrix = rgb_image.actual_matrix;
	for (int c = 0; c < channels; ++c) {
		std::cout << "Channel " << c << ":\n";
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				double value = image_matrix(c, y * width + x);
				std::cout << value << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
}