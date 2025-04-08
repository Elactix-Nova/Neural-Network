#include "pooling.hpp"
#include <limits>

// ----------- MaxPooling Implementation --------------

MaxPooling::MaxPooling(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

std::vector<Eigen::MatrixXd> MaxPooling::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;

    int channels = input.size();
    std::vector<Eigen::MatrixXd> output(channels);
    max_row_indices.resize(channels);
    max_col_indices.resize(channels);

    for (int c = 0; c < channels; ++c) {
        int in_rows = input[c].rows();
        int in_cols = input[c].cols();
        int out_rows = (in_rows - kernel_size) / stride + 1;
        int out_cols = (in_cols - kernel_size) / stride + 1;

        output[c] = Eigen::MatrixXd(out_rows, out_cols);
        max_row_indices[c] = Eigen::MatrixXi(out_rows, out_cols);
        max_col_indices[c] = Eigen::MatrixXi(out_rows, out_cols);

        for (int i = 0; i < out_rows; ++i) {
            for (int j = 0; j < out_cols; ++j) {
                double max_val = -std::numeric_limits<double>::infinity();
                int max_row = 0, max_col = 0;

                for (int m = 0; m < kernel_size; ++m) {
                    for (int n = 0; n < kernel_size; ++n) {
                        int row_idx = i * stride + m;
                        int col_idx = j * stride + n;
                        if (input[c](row_idx, col_idx) > max_val) {
                            max_val = input[c](row_idx, col_idx);
                            max_row = row_idx;
                            max_col = col_idx;
                        }
                    }
                }

                output[c](i, j) = max_val;
                max_row_indices[c](i, j) = max_row;
                max_col_indices[c](i, j) = max_col;
            }
        }
    }

    return output;
}

std::vector<Eigen::MatrixXd> MaxPooling::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> input_gradient(input.size());

    for (size_t c = 0; c < input.size(); ++c) {
        input_gradient[c] = Eigen::MatrixXd::Zero(input[c].rows(), input[c].cols());

        for (int i = 0; i < output_gradient[c].rows(); ++i) {
            for (int j = 0; j < output_gradient[c].cols(); ++j) {
                int row_idx = max_row_indices[c](i, j);
                int col_idx = max_col_indices[c](i, j);
                input_gradient[c](row_idx, col_idx) += output_gradient[c](i, j);
            }
        }
    }

    return input_gradient;
}

// ----------- AveragePooling Implementation --------------

AveragePooling::AveragePooling(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

std::vector<Eigen::MatrixXd> AveragePooling::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;

    int channels = input.size();
    std::vector<Eigen::MatrixXd> output(channels);

    for (int c = 0; c < channels; ++c) {
        int in_rows = input[c].rows();
        int in_cols = input[c].cols();
        int out_rows = (in_rows - kernel_size) / stride + 1;
        int out_cols = (in_cols - kernel_size) / stride + 1;

        output[c] = Eigen::MatrixXd(out_rows, out_cols);

        for (int i = 0; i < out_rows; ++i) {
            for (int j = 0; j < out_cols; ++j) {
                double sum = 0.0;

                for (int m = 0; m < kernel_size; ++m) {
                    for (int n = 0; n < kernel_size; ++n) {
                        int row_idx = i * stride + m;
                        int col_idx = j * stride + n;
                        sum += input[c](row_idx, col_idx);
                    }
                }

                output[c](i, j) = sum / (kernel_size * kernel_size);
            }
        }
    }

    return output;
}

std::vector<Eigen::MatrixXd> AveragePooling::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> input_gradient(input.size());

    for (size_t c = 0; c < input.size(); ++c) {
        input_gradient[c] = Eigen::MatrixXd::Zero(input[c].rows(), input[c].cols());

        for (int i = 0; i < output_gradient[c].rows(); ++i) {
            for (int j = 0; j < output_gradient[c].cols(); ++j) {
                double grad = output_gradient[c](i, j) / (kernel_size * kernel_size);

                for (int m = 0; m < kernel_size; ++m) {
                    for (int n = 0; n < kernel_size; ++n) {
                        int row_idx = i * stride + m;
                        int col_idx = j * stride + n;
                        input_gradient[c](row_idx, col_idx) += grad;
                    }
                }
            }
        }
    }

    return input_gradient;
}

// GlobalAvgPooling Implementation
// ------------------------------------------------------------------------------------
GlobalAvgPooling::GlobalAvgPooling(int kernel_size, int stride) 
    : kernel_size(kernel_size), stride(stride) {}

std::vector<Eigen::MatrixXd> GlobalAvgPooling::forward(const std::vector<Eigen::MatrixXd>& input) {
    // Store input shape for backward pass
    input_shape = {static_cast<int>(input.size()), // channels
                  static_cast<int>(input[0].rows()), // height
                  static_cast<int>(input[0].cols())}; // width
    
    // Initialize output with same number of channels but 1x1 size
    std::vector<Eigen::MatrixXd> output(input.size());
    
    for (size_t c = 0; c < input.size(); ++c) {
        // Calculate mean of entire feature map
        output[c] = Eigen::MatrixXd::Zero(1, 1); // Init. output element(for one channel/feature map)
        output[c](0, 0) = input[c].mean();
    }
    
    return output;
}

std::vector<Eigen::MatrixXd> GlobalAvgPooling::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    
    std::vector<Eigen::MatrixXd> input_gradient(input_shape[0]);
    
    for (size_t c = 0; c < input_shape[0]; ++c) {
        // Get the gradient value for this channel
        double grad = output_gradient[c](0, 0);
        
        // Calculate the scaling factor (1/N where N is total number of elements)
        double scale = grad / (input_shape[1] * input_shape[2]);
        
        // Distribute gradient equally to all positions
        input_gradient[c] = Eigen::MatrixXd::Constant(
            input_shape[1], 
            input_shape[2], 
            scale
        );
    }
    
    return input_gradient;
}
