#include "dataloader.hpp"
// using namespace std;
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

std::vector<Eigen::MatrixXd> ImageFolder::raw_img_to_matrix(unsigned char* raw_img, int channels, int width, int height)
{
	
	std::vector<Eigen::MatrixXd> final_image(channels, Eigen::MatrixXd(height, width));
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


ImageFolder::ImageFolder(std::string folder_root)
{
	// // Init. member variables
	num_classes = 0;
	root_folder_path = folder_root;
	
	// Temp. variables for convenience
	std::string abs_path;
	std::string current_label;
	// string ext;
	int channels, width, height; //temp variables
	
	// Traverse top level directory ('./<ds>/train')
	for (const auto& dir_or_file: fs::directory_iterator(folder_root)){
		
		// Reset current path(i.e. what 'cwd' would return) to base directory of train/val dataset
		fs::current_path(folder_root);
		
		// Check that it is a directory, not a bs file that somehow made it in(e.g. ".DS_store", extremely problematic)
		if (dir_or_file.is_directory())
		{
			
			std::cout << "Path to dir is:  " << dir_or_file.path() << std::endl;
			
			// Retrieve current directory name, assuming directory name is label name, like "rose" or "1"
			current_label = dir_or_file.path().filename().string();
			std::cout << "Current label(adding to labels vector) is:" << current_label << std::endl;
			
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
				// std::cout << "Path to this img: " << img_file.path().string() << std::endl;
				// Get extension of current file, verify it's a valid image file
				std::string ext = img_file.path().extension().string();
				if (img_file.is_regular_file() && ((ext == ".jpg") || (ext == ".png") || (ext == ".jpeg")))
				{
					// std::cout << "Adding the image at path: " << img_file << std::endl;
					
					// Read raw image from file path
					unsigned char* curr_img_raw = stbi_load(img_file.path().string().data(), &width, &height, &channels, 0);
					
					if (!curr_img_raw)
					{
						std::cerr << "Failed to load image: " << img_file.path() << std::endl;
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
			// Dereference shared_ptr to get actual vector<Eigen::MatrixXd>
			std::vector<Eigen::MatrixXd> input = *img_ptr ;  
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
    if (current_batch >= num_batches) {
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