// #include <opencv4/opencv.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#include "image_loader.hpp"
#include <iostream>
#include <string>
#include <vector>
using namespace std;


Final_Matrix read_image(string img_file_path, bool is_grayscale)
{
	int width, height, channels;
	const char* file_name = img_file_path.data();
	// Read the image from memory
	unsigned char *raw_img = stbi_load(file_name, &width, &height, &channels, 0);
	if (raw_img == NULL)
	{
		cout << "Couldn't load the image at file path: " << img_file_path << endl;
		throw std::runtime_error("Image loading failed");;
	}
	cout << "Loaded image with\n" << "width:" << width << "\nheight:" << height << "\nchannels:" << channels << endl;
	// Now, convert to Eigen Matrix
	// Eigen doesn't freaking support 3d matrices, so we make a flattened 2d version
	Eigen::MatrixXd final_image(channels, height * width);
	// vector<Eigen::MatrixXd> final_image;
	// for (int i =0 ; i < channels; i++)
	// {
	// 	final_image[i] = Eigen::MatrixXd();
	// }
	int col;
	int channel;
	int pixel_index;
	for (int row = 0; row < height; row++)
	{
		for (col = 0; col < width; col++)
		{
			pixel_index = (row * width + col) * channels;
			
			for (channel = 0; channel < channels; channel++)
			{
				final_image(channel, row*width + col) = raw_img[pixel_index + channel] / 255.0;
			}
		}
	}
	Final_Matrix fin;
	fin.channels = channels;
	fin.height = height;
	fin.width = width;
	fin.actual_matrix = final_image;
	// Remember, this is a flat 2d matrix
	return fin;
}