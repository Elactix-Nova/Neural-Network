#include <Eigen/Dense>
#include <string>

// /*
// Assumed structure of dataset folder
// dataset/
// ├── train/
// │   ├── 0/
// │   │   ├── img1.png
// │   │   ├── img2.png
// │   ├── 1/
// │   │   ├── img1.png
// │   └── ...
// ├── test/
// │   ├── 0/
// │   ├── 1/
// │   └── ...
// */

// class ImageFolder
// {
	
// }

typedef struct final_matrix
{
	int channels;
	int height;
	int width;
	Eigen::MatrixXd actual_matrix;
} Final_Matrix;

// Read image from file as a matrix
Final_Matrix read_image(std::string img_file_path, bool is_grayscale);