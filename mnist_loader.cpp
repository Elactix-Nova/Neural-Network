#include "network.hpp"
#include "dense.hpp"
#include "convolutional.hpp"
#include "reshape.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include "dataloader.hpp"
#include "pooling.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>

// Function to load MNIST images
std::vector<Eigen::MatrixXd> load_mnist_images(const std::string& path, int num_images) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open image file " + path);

    file.ignore(16);  // Skip MNIST header (always present)

    std::vector<Eigen::MatrixXd> images;
    for (int i = 0; i < num_images; ++i) {
        Eigen::MatrixXd image(28, 28);
        for (int r = 0; r < 28; ++r)
            for (int c = 0; c < 28; ++c) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                image(r, c) = static_cast<double>(pixel) / 255.0;
            }
        images.push_back(image);
    }
    file.close();
    return images;
}

// Function to load MNIST labels
std::vector<int> load_mnist_labels(const std::string& path, int num_labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open label file " + path);

    file.ignore(8);  // Skip MNIST header (always present)

    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }
    file.close();
    return labels;
}

int main() {
    // Load MNIST data
    int num_train = 600;  // Using smaller set for quicker demonstration
    auto train_images = load_mnist_images("train-images.idx3-ubyte", num_train);
    auto train_labels = load_mnist_labels("train-labels.idx1-ubyte", num_train);

    DataLoader train_loader(train_images, train_labels, 100, 10, true);

    // Define CNN
    std::vector<std::shared_ptr<Layer>> layers = {
        std::make_shared<Convolutional>(std::vector<int>{1,28,28}, 3, 5, 1, 0),
        std::make_shared<AveragePooling>(6,4),
        std::make_shared<Sigmoid>(),
        std::make_shared<Reshape>(std::vector<int>{5,6,6}, std::vector<int>{1,5*6*6,1}),
        std::make_shared<Dense>(5*6*6, 36),
        std::make_shared<Sigmoid>(),
        std::make_shared<Dense>(36, 10),
        std::make_shared<Softmax>()
    };

    Network network(layers);

    // Train the network
    int epochs = 100;
    double learning_rate = 0.1;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        train_loader.reset();
        double epoch_loss = 0;

        for (int b = 0; b < train_loader.get_num_batches(); ++b) {
            auto [batch_x, batch_y] = train_loader.get_next_batch();
            network.train(batch_x, batch_y,
                          Loss::cross_entropy_loss,
                          Loss::cross_entropy_loss_prime,
                          1, learning_rate, false);

            for (size_t i = 0; i < batch_x.size(); ++i) {
                auto prediction = network.predict(batch_x[i]);
                epoch_loss += Loss::cross_entropy_loss(batch_y[i], prediction);
                std::cout << "Loss " << epoch_loss << std::endl; 
            }
        }

        epoch_loss /= train_loader.get_num_batches();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << epoch_loss << std::endl;
    }

    // Simple evaluation on training set
    int correct = 0;
    for (int i = 0; i < num_train; ++i) {
        auto output = network.predict({train_images[i]});
        int predicted_label;
        output[0].col(0).maxCoeff(&predicted_label);
        if (predicted_label == train_labels[i]) correct++;
    }

    std::cout << "Training accuracy: " << (double(correct) / num_train) * 100 << "%" << std::endl;

    return 0;
}