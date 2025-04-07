// #include <opencv4/opencv.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#include "image_loader.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>
using namespace std;
namespace fs = std::filesystem;

ImageFolder::~ImageFolder()
{
	// Using shared ptr, this code may cause issues
	// for (auto& class_images : images)
	// {
	// 	for (auto& img_ptr : class_images)
	// 	{
	// 		delete img_ptr;
	// 	}
	// }
}

using ImageChannels = std::vector<Eigen::MatrixXd>;
using ImagePtr = std::shared_ptr<ImageChannels>;

vector<Eigen::MatrixXd> ImageFolder::raw_img_to_matrix(unsigned char* raw_img, int channels, int width, int height)
{
	
	vector<Eigen::MatrixXd> final_image(channels, Eigen::MatrixXd(height, width));
	int col;
	int channel;
	int pixel_index;
	for (channel = 0; channel < channels; channel++)
	{
		for (int row = 0; row < height; row++)
		{
			for (col = 0; col < width; col++)
			{
				pixel_index = (row * width + col) * channels;
				final_image[channel](row,col) = raw_img[pixel_index + channel] / 255.0;
			}
		}
	}
	return final_image;
}


ImageFolder::ImageFolder(string folder_root)
{
	// // Init. member variables
	num_classes = 0;
	root_folder_path = folder_root;
	
	// Temp. variables for convenience
	string abs_path;
	string current_label;
	// string ext;
	int channels, width, height; //temp variables
	
	// Traverse top level directory ('./<ds>/train')
	for (const auto& dir_or_file: fs::directory_iterator(folder_root)){
		
		// Reset current path(i.e. what 'cwd' would return) to base directory of train/val dataset
		fs::current_path(folder_root);
		
		// Check that it is a directory, not a bs file that somehow made it in(e.g. ".DS_store", extremely problematic)
		if (dir_or_file.is_directory())
		{
			
			cout << "Path to dir is:  " << dir_or_file.path() << endl;
			
			// Retrieve current directory name, assuming directory name is label name, like "rose" or "1"
			current_label = dir_or_file.path().filename().string();
			cout << "Current label(adding to labels vector) is:" << current_label << endl;
			
			// Process new label
			num_classes+=1;
			labels.push_back(current_label); // provides index to label mapping
			label_counts[current_label] = 0;
			
			// Add empty nested vector for new labels
			// images.push_back({}); 
			// images_data.push_back({});
			
			// Simpler, less wierd 
			images.emplace_back();
			images_data.emplace_back();
			
			for (const auto& img_file: fs::directory_iterator(dir_or_file.path()))
			{
				// cout << "Path to this img: " << img_file.path().string() << endl;
				// Get extension of current file, verify it's a valid image file
				string ext = img_file.path().extension().string();
				if (img_file.is_regular_file() && ((ext == ".jpg") || (ext == ".png") || (ext == ".jpeg")))
				{
					// cout << "Adding the image at path: " << img_file << endl;
					
					// Read raw image from file path
					unsigned char* curr_img_raw = stbi_load(img_file.path().string().data(), &width, &height, &channels, 0);
					
					if (!curr_img_raw)
					{
						cerr << "Failed to load image: " << img_file.path() << endl;
						continue; // Skip this image and move on
					}
					
					// Convert to Eigen Matrix
					auto raw_vec = raw_img_to_matrix(curr_img_raw, channels, width, height);
					ImagePtr img_ptr = std::make_shared<ImageChannels>(std::move(raw_vec));
					
					// Add image to last added nested vector, which corresponds to current label
					images.back().push_back(img_ptr);
					
					// Add image metadata
					ImageStruct meta;
					meta.channels = channels;
					meta.width = width;
					meta.height = height;
					meta.actual_image = img_ptr;
					meta.file_path = img_file.path().string();
					meta.label = current_label;
					
					images_data.back().push_back(meta);
					
					label_counts[current_label] += 1; // Increment count of images with this label
					
					// Free raw image buffer
					stbi_image_free(curr_img_raw);
				}
			}
		}
	}
}


// Ignore, older function once used for testing
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