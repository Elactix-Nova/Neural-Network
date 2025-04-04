#include "pooling.hpp"
#include <iostream>

// MaxPooling implementation
MaxPooling::MaxPooling(int kernel_size, int stride)
    : kernel_size(kernel_size),
      stride(stride == -1 ? kernel_size : stride) {}

std::vector<Eigen::MatrixXd> MaxPooling::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    int channels = input.size();
    std::vector<Eigen::MatrixXd> output(channels);
    max_row_indices.resize(channels);
    max_col_indices.resize(channels);

    for (int ch = 0; ch < channels; ++ch) {
        int out_rows = (input[ch].rows() - kernel_size) / stride + 1;
        int out_cols = (input[ch].cols() - kernel_size) / stride + 1;

        output[ch] = Eigen::MatrixXd(out_rows, out_cols);
        max_row_indices[ch] = Eigen::MatrixXi(out_rows, out_cols);
        max_col_indices[ch] = Eigen::MatrixXi(out_rows, out_cols);

        for (int i = 0; i < out_rows; ++i) {
            for (int j = 0; j < out_cols; ++j) {
                Eigen::MatrixXd window = input[ch].block(i * stride, j * stride, kernel_size, kernel_size);
                Eigen::Index maxRow, maxCol;
                double maxVal = window.maxCoeff(&maxRow, &maxCol);

                output[ch](i, j) = maxVal;
                max_row_indices[ch](i, j) = maxRow;
                max_col_indices[ch](i, j) = maxCol;
            }
        }
    }

    return output;
}

std::vector<Eigen::MatrixXd> MaxPooling::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> input_gradient(input.size());

    for (size_t ch = 0; ch < input.size(); ++ch) {
        input_gradient[ch] = Eigen::MatrixXd::Zero(input[ch].rows(), input[ch].cols());

        for (int i = 0; i < output_gradient[ch].rows(); ++i) {
            for (int j = 0; j < output_gradient[ch].cols(); ++j) {
                int max_i = i * stride + max_row_indices[ch](i, j);
                int max_j = j * stride + max_col_indices[ch](i, j);
                input_gradient[ch](max_i, max_j) += output_gradient[ch](i, j);
            }
        }
    }

    return input_gradient;
}

// AveragePooling implementation
AveragePooling::AveragePooling(int kernel_size, int stride)
    : kernel_size(kernel_size),
      stride(stride == -1 ? kernel_size : stride) {}

std::vector<Eigen::MatrixXd> AveragePooling::forward(const std::vector<Eigen::MatrixXd>& input) {
    this->input = input;
    int channels = input.size();
    std::vector<Eigen::MatrixXd> output(channels);

    for (int ch = 0; ch < channels; ++ch) {
        int out_rows = (input[ch].rows() - kernel_size) / stride + 1;
        int out_cols = (input[ch].cols() - kernel_size) / stride + 1;

        output[ch] = Eigen::MatrixXd(out_rows, out_cols);

        for (int i = 0; i < out_rows; ++i) {
            for (int j = 0; j < out_cols; ++j) {
                Eigen::MatrixXd window = input[ch].block(i * stride, j * stride, kernel_size, kernel_size);
                output[ch](i, j) = window.mean();
            }
        }
    }

    // std::cout << "Channels " << output.size() << " Height " << output[0].rows() << " Width " << output[0].cols() << std::endl;
    return output;
}

std::vector<Eigen::MatrixXd> AveragePooling::backward(const std::vector<Eigen::MatrixXd>& output_gradient, double learning_rate) {
    std::vector<Eigen::MatrixXd> input_gradient(input.size());

    for (size_t ch = 0; ch < input.size(); ++ch) {
        input_gradient[ch] = Eigen::MatrixXd::Zero(input[ch].rows(), input[ch].cols());

        for (int i = 0; i < output_gradient[ch].rows(); ++i) {
            for (int j = 0; j < output_gradient[ch].cols(); ++j) {
                input_gradient[ch].block(i * stride, j * stride, kernel_size, kernel_size)
                    .array() += output_gradient[ch](i, j) / (kernel_size * kernel_size);
            }
        }
    }

    // std::cout << "Channels " << input_gradient.size() << " Height " << input_gradient[0].rows() << " Width " << input_gradient[0].cols() << std::endl;
    return input_gradient;
}
