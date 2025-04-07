#include <Eigen/Dense>
#include <string>
#include <unordered_set>
#include <vector>
#include <memory> // Shared ptr
using namespace std;

/*
Assumed structure of dataset folder
dataset/
├── train/
│   ├── 0/
│   │   ├── img1.png
│   │   ├── img2.png
│   ├── 1/
│   │   ├── img1.png
│   └── ...
├── test/
│   ├── 0/
│   ├── 1/
│   └── ...
*/

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

// Forget these two for now, not using these except for testing
typedef struct final_matrix
{
	int channels;
	int height;
	int width;
	Eigen::MatrixXd actual_matrix;
} Final_Matrix;


// Read image from file as a matrix
Final_Matrix read_image(std::string img_file_path, bool is_grayscale);