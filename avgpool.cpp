#include <iostream>
#include <vector>

using namespace std;

class AveragePooling {
public:
    int kernel_size;
    int stride;
    vector<vector<vector<double>>> last_input;
 
    AveragePooling(int kernel_size, int stride = -1) : kernel_size(kernel_size) {
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


        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    double sum = 0.0;
                    // Sum over the window.
                    for (int m = 0; m < kernel_size; m++) {
                        for (int n = 0; n < kernel_size; n++) {
                            int cur_row = i * stride + m;
                            int cur_col = j * stride + n;
                            sum += input[d][cur_row][cur_col];
                        }
                    }
           
                    output[d][i][j] = sum / (kernel_size * kernel_size);
                }
            }
        }
        return output;
    }

    vector<vector<vector<double>>> backward(const vector<vector<vector<double>>>& grad_output) {
        int depth = last_input.size();
        int height = last_input[0].size();
        int width = last_input[0][0].size();

        vector<vector<vector<double>>> grad_input(depth, vector<vector<double>>(height, vector<double>(width, 0.0)));
        int out_height = grad_output[0].size();
        int out_width = grad_output[0][0].size();

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    double grad = grad_output[d][i][j] / (kernel_size * kernel_size);
                    for (int m = 0; m < kernel_size; m++) {
                        for (int n = 0; n < kernel_size; n++) {
                            int cur_row = i * stride + m;
                            int cur_col = j * stride + n;
                            grad_input[d][cur_row][cur_col] += grad;
                        }
                    }
                }
            }
        }
        return grad_input;
    }
};
