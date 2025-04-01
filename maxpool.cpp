#include <vector>
#include <utility>
#include <limits>

using namespace std;

class MaxPooling {
public:
    int kernel_size;
    int stride; 

    vector<vector<vector<pair<int, int>>>> max_indices;
    vector<vector<vector<double>>> last_input;

    MaxPooling(int kernel_size, int stride = -1) : kernel_size(kernel_size) {
        this->stride = (stride == -1) ? kernel_size : stride;
    }

    vector<vector<vector<double>>> forward(const vector<vector<vector<double>>>& input) {
        last_input = input;
        int depth = input.size();
        int height = input[0].size();
        int width = input[0][0].size();

        int out_height = (height - kernel_size) / stride + 1;
        int out_width = (width - kernel_size) / stride + 1;

        vector<vector<vector<double>>> output(depth, vector<vector<double>>(out_height, vector<double>(out_width, 0.0)));
        max_indices = vector<vector<vector<pair<int, int>>>>(
            depth, vector<vector<pair<int, int>>>(out_height, vector<pair<int, int>>(out_width, {0, 0}))
        );

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    double max_val = -numeric_limits<double>::infinity();
                    int max_row = -1, max_col = -1;
                    for (int m = 0; m < kernel_size; m++) {
                        for (int n = 0; n < kernel_size; n++) {
                            int cur_row = i * stride + m;
                            int cur_col = j * stride + n;
                            double cur_val = input[d][cur_row][cur_col];
                            if (cur_val > max_val) {
                                max_val = cur_val;
                                max_row = cur_row;
                                max_col = cur_col;
                            }
                        }
                    }
                    output[d][i][j] = max_val;
                    max_indices[d][i][j] = {max_row, max_col};
                }
            }
        }
        return output;
    }

    vector<vector<vector<double>>> backward(const vector<vector<vector<double>>>& grad_output) {
        int depth = last_input.size();
        int height = last_input[0].size();
        int width = last_input[0][0].size();

        // Initialize the gradient for the input with zeros.
        vector<vector<vector<double>>> grad_input(depth, vector<vector<double>>(height, vector<double>(width, 0.0)));
        int out_height = grad_output[0].size();
        int out_width = grad_output[0][0].size();

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    pair<int, int> max_pos = max_indices[d][i][j];
                    grad_input[d][max_pos.first][max_pos.second] += grad_output[d][i][j];
                }
            }
        }
        return grad_input;
    }
};
